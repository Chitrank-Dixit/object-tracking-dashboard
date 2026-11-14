"""Custom dataset fine-tuning pipeline for YOLO models with Roboflow integration.

Provides functions to download remote datasets, validate configurations, execute
Ultralytics YOLO training runs, copy best weights to models/, and expose a CLI entrypoint.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from src.schemas import TrainingConfig, TrainingResult

logger = logging.getLogger(__name__)


def download_roboflow_dataset(
    config: TrainingConfig,
    target_dir: str | Path = "data",
) -> Path:
    """Download a remote dataset from Roboflow in YOLOv8 format.

    Args:
        config: TrainingConfig containing Roboflow API key, workspace, project, and version.
        target_dir: Directory where the dataset will be saved.

    Returns:
        Resolved Path to the downloaded data.yaml file.

    Raises:
        RuntimeError: If Roboflow authentication or download fails.
        FileNotFoundError: If the downloaded dataset does not contain data.yaml.
    """
    if not (
        config.roboflow_api_key
        and config.roboflow_workspace
        and config.roboflow_project
        and config.roboflow_version
    ):
        raise ValueError("Complete Roboflow parameters required to initiate dataset download")

    try:
        from roboflow import Roboflow  # type: ignore[import-untyped]

        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Connecting to Roboflow workspace '%s', project '%s', version %d",
            config.roboflow_workspace,
            config.roboflow_project,
            config.roboflow_version,
        )

        rf = Roboflow(api_key=config.roboflow_api_key)
        workspace = rf.workspace(config.roboflow_workspace)
        project = workspace.project(config.roboflow_project)
        version = project.version(config.roboflow_version)

        dataset = version.download(
            model_format=config.roboflow_model_format,
            location=str(target_path / config.roboflow_project),
        )

        location_path = (
            Path(dataset.location)
            if hasattr(dataset, "location")
            else target_path / config.roboflow_project
        )
        data_yaml = location_path / "data.yaml"

        if not data_yaml.is_file():
            raise FileNotFoundError(
                f"Roboflow download did not produce a valid data.yaml at {location_path}"
            )

        logger.info("Roboflow dataset downloaded successfully: %s", data_yaml)
        return data_yaml.resolve()

    except FileNotFoundError:
        raise
    except Exception as exc:
        logger.error("Roboflow download failed: %s", exc)
        raise RuntimeError(f"Roboflow dataset download failed: {exc}") from exc


def resolve_dataset_path(config: TrainingConfig) -> Path:
    """Resolve and verify dataset path from local specification or Roboflow download.

    Args:
        config: Validated TrainingConfig instance.

    Returns:
        Resolved Path to data.yaml.

    Raises:
        FileNotFoundError: If local dataset path cannot be located.
    """
    if config.data:
        data_path = Path(config.data)
        if data_path.is_file():
            return data_path.resolve()
        elif data_path.is_dir() and (data_path / "data.yaml").is_file():
            return (data_path / "data.yaml").resolve()
        else:
            raise FileNotFoundError(f"Local dataset path not found: {config.data}")

    return download_roboflow_dataset(config)


def run_training(
    config: TrainingConfig,
    model: Any | None = None,
) -> TrainingResult:
    """Execute YOLO fine-tuning pipeline or dry-run validation.

    Args:
        config: Validated TrainingConfig hyperparameters and paths.
        model: Optional pre-instantiated YOLO model for testing and mocking.

    Returns:
        Validated TrainingResult summarizing the fine-tuning outcome.

    Raises:
        FileNotFoundError: If dataset or exported best weights cannot be located.
        RuntimeError: If training fails during execution.
    """
    # 1. Resolve and validate dataset manifest
    dataset_path = resolve_dataset_path(config)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {dataset_path}")

    # 2. Ensure output directory exists
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Handle Dry-Run validation
    if config.dry_run:
        logger.info(
            "Dry-run validation successful for model '%s' with dataset '%s'",
            config.model_name,
            dataset_path,
        )
        # Verify base model identifier or file can be loaded if not mocked
        if model is None:
            _ = YOLO(config.model_name)

        return TrainingResult(
            success=True,
            model_name=config.model_name,
            dataset_path=str(dataset_path),
            epochs_completed=0,
            best_weight_path=None,
            metrics={},
            dry_run=True,
            training_time_seconds=0.0,
        )

    # 4. Initialize YOLO model
    yolo_model = model if model is not None else YOLO(config.model_name)

    start_time = time.perf_counter()

    try:
        logger.info(
            "Starting fine-tuning: %s on %s for %d epochs (batch=%d, imgsz=%d, device=%s)",
            config.model_name,
            dataset_path,
            config.epochs,
            config.batch_size,
            config.imgsz,
            config.device,
        )

        results = yolo_model.train(
            data=str(dataset_path),
            epochs=config.epochs,
            batch=config.batch_size,
            imgsz=config.imgsz,
            device=config.device,
            workers=config.workers,
            patience=config.patience,
            optimizer=config.optimizer,
            lr0=config.lr0,
            lrf=config.lrf,
            seed=config.seed,
            project=config.project,
            name=config.name,
            exist_ok=config.exist_ok,
            resume=config.resume,
            verbose=True,
        )
    except Exception as exc:
        logger.error("YOLO training run encountered an error: %s", exc)
        raise

    elapsed_time = time.perf_counter() - start_time

    # 5. Locate and persist best weights to models/
    run_save_dir = Path(getattr(results, "save_dir", Path(config.project) / config.name))
    best_weights_src = run_save_dir / "weights" / "best.pt"

    if not best_weights_src.is_file():
        # Fallback to direct project/name path
        best_weights_src = Path(config.project) / config.name / "weights" / "best.pt"

    if not best_weights_src.is_file():
        raise FileNotFoundError(
            f"Training finished but best weights were not generated at {best_weights_src}"
        )

    dest_weight_path = output_dir / f"{config.name}_best.pt"
    shutil.copy2(best_weights_src, dest_weight_path)
    logger.info("Best weights successfully exported to %s", dest_weight_path)

    # Extract metrics dict if present
    metrics: dict[str, float] = {}
    if hasattr(results, "results_dict") and isinstance(results.results_dict, dict):
        metrics = {
            k: float(v) for k, v in results.results_dict.items() if isinstance(v, (int, float))
        }

    return TrainingResult(
        success=True,
        model_name=config.model_name,
        dataset_path=str(dataset_path),
        epochs_completed=config.epochs,
        best_weight_path=str(dest_weight_path.resolve()),
        metrics=metrics,
        dry_run=False,
        training_time_seconds=elapsed_time,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build command-line interface argument parser for fine-tuning pipeline."""
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLO object detection models with local or Roboflow datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model and dataset
    parser.add_argument(
        "--model",
        dest="model_name",
        default="yolov8n.pt",
        help="Path or name of base YOLO weights (e.g., yolov8n.pt, yolov8s.pt)",
    )
    parser.add_argument(
        "--data",
        dest="data",
        default=None,
        help="Path to local dataset YAML file or directory",
    )

    # Roboflow arguments
    parser.add_argument(
        "--roboflow-api-key",
        dest="roboflow_api_key",
        default=None,
        help="Roboflow private API key",
    )
    parser.add_argument(
        "--roboflow-workspace",
        dest="roboflow_workspace",
        default=None,
        help="Roboflow workspace URL slug",
    )
    parser.add_argument(
        "--roboflow-project",
        dest="roboflow_project",
        default=None,
        help="Roboflow project slug",
    )
    parser.add_argument(
        "--roboflow-version",
        dest="roboflow_version",
        type=int,
        default=None,
        help="Roboflow dataset version integer",
    )

    # Hyperparameters
    parser.add_argument(
        "--epochs",
        dest="epochs",
        type=int,
        default=50,
        help="Total training epochs",
    )
    parser.add_argument(
        "--batch",
        dest="batch_size",
        type=int,
        default=16,
        help="Batch size per step",
    )
    parser.add_argument(
        "--imgsz",
        dest="imgsz",
        type=int,
        default=640,
        help="Input image resolution in pixels",
    )
    parser.add_argument(
        "--device",
        dest="device",
        default="cpu",
        help="Compute target device ('cpu', '0', 'cuda')",
    )
    parser.add_argument(
        "--workers",
        dest="workers",
        type=int,
        default=2,
        help="Worker threads for data loading",
    )
    parser.add_argument(
        "--lr0",
        dest="lr0",
        type=float,
        default=0.01,
        help="Initial learning rate",
    )

    # Run management
    parser.add_argument(
        "--project",
        dest="project",
        default="runs/train",
        help="Runs directory for training output",
    )
    parser.add_argument(
        "--name",
        dest="name",
        default="exp",
        help="Run identifier name",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default="models",
        help="Target directory for exported best.pt weights",
    )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Verify config and data paths without running training loop",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI execution entrypoint.

    Args:
        argv: Optional list of argument strings. Defaults to sys.argv[1:].

    Returns:
        0 on success, non-zero on failure.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    # Convert non-None parsed arguments to dictionary
    kwargs = {k: v for k, v in vars(args).items() if v is not None}

    try:
        config = TrainingConfig(**kwargs)
        result = run_training(config)
        logger.info(
            "Pipeline finished successfully (dry_run=%s, best_weight=%s)",
            result.dry_run,
            result.best_weight_path,
        )
        return 0
    except Exception as exc:
        logger.error("Training pipeline failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
