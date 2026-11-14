"""Comprehensive unit and integration test suite for fine-tuning pipeline in src.train.

Implements all 25 test scenarios (TC-TR-01 through TC-TR-25) specified in Milestone 4:
- TC-TR-01 to TC-TR-10: Pydantic v2 TrainingConfig and TrainingResult validation.
- TC-TR-11 to TC-TR-13: Roboflow download orchestration and error handling.
- TC-TR-14 to TC-TR-20: run_training execution, dry-run, weight export, and error mapping.
- TC-TR-21 to TC-TR-25: CLI argument parsing and main() execution entrypoint.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from src.schemas import TrainingConfig, TrainingResult
from src.train import (
    build_parser,
    download_roboflow_dataset,
    main,
    resolve_dataset_path,
    run_training,
)

# ==============================================================================
# TrainingConfig and TrainingResult Validation Tests (TC-TR-01 to TC-TR-10)
# ==============================================================================


@pytest.mark.unit
def test_tc_tr_01_training_config_nominal_local() -> None:
    """TC-TR-01: Verify nominal TrainingConfig initialization with local dataset."""
    config = TrainingConfig(data="data/custom.yaml")

    assert config.data == "data/custom.yaml"
    assert config.model_name == "yolov8n.pt"
    assert config.epochs == 50
    assert config.batch_size == 16
    assert config.imgsz == 640
    assert config.device == "cpu"
    assert config.workers == 2
    assert config.patience == 10
    assert config.optimizer == "auto"
    assert config.lr0 == 0.01
    assert config.lrf == 0.01
    assert config.seed == 42
    assert config.project == "runs/train"
    assert config.name == "exp"
    assert config.output_dir == "models"
    assert config.dry_run is False
    assert config.exist_ok is True
    assert config.resume is False
    assert config.roboflow_api_key is None
    assert config.roboflow_workspace is None
    assert config.roboflow_project is None
    assert config.roboflow_version is None
    assert config.roboflow_model_format == "yolov8"


@pytest.mark.unit
def test_tc_tr_02_training_config_nominal_roboflow() -> None:
    """TC-TR-02: Verify nominal TrainingConfig initialization with Roboflow credentials."""
    config = TrainingConfig(
        roboflow_api_key="rf_key_sec_12345",
        roboflow_workspace="smart-retail",
        roboflow_project="store-aisle-items",
        roboflow_version=3,
        epochs=100,
        batch_size=32,
        imgsz=1280,
    )

    assert config.data is None
    assert config.roboflow_api_key == "rf_key_sec_12345"
    assert config.roboflow_workspace == "smart-retail"
    assert config.roboflow_project == "store-aisle-items"
    assert config.roboflow_version == 3
    assert config.roboflow_model_format == "yolov8"
    assert config.epochs == 100
    assert config.batch_size == 32
    assert config.imgsz == 1280


@pytest.mark.unit
def test_tc_tr_03_training_config_missing_dataset_source_raises() -> None:
    """TC-TR-03: Verify TrainingConfig rejects initialization without any dataset source."""
    with pytest.raises(ValidationError) as exc_info:
        TrainingConfig()

    error_msg = str(exc_info.value)
    assert (
        "Either a local dataset path ('data') or complete Roboflow credentials must be specified"
        in error_msg
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "incomplete_rf",
    [
        {"roboflow_api_key": "api_key_only"},
        {"roboflow_workspace": "workspace_only"},
        {"roboflow_project": "project_only"},
        {"roboflow_version": 2},
        {"roboflow_api_key": "key", "roboflow_workspace": "ws"},
        {"roboflow_api_key": "key", "roboflow_workspace": "ws", "roboflow_project": "proj"},
        {"roboflow_workspace": "ws", "roboflow_project": "proj", "roboflow_version": 1},
    ],
)
def test_tc_tr_04_training_config_incomplete_roboflow_credentials_raises(
    incomplete_rf: dict[str, object],
) -> None:
    """TC-TR-04: Verify TrainingConfig rejects incomplete Roboflow credential combinations."""
    with pytest.raises(ValidationError) as exc_info:
        TrainingConfig(**incomplete_rf)

    error_msg = str(exc_info.value)
    assert "Roboflow download requires all four parameters" in error_msg


@pytest.mark.unit
@pytest.mark.parametrize("invalid_epochs", [0, -1, -50, 1001, 2000])
def test_tc_tr_05_training_config_invalid_epochs_raises(invalid_epochs: int) -> None:
    """TC-TR-05: Verify TrainingConfig enforces epochs bounds between 1 and 1000."""
    with pytest.raises(ValidationError) as exc_info:
        TrainingConfig(data="data.yaml", epochs=invalid_epochs)

    error_msg = str(exc_info.value)
    assert "epochs" in error_msg
    assert (
        "greater_than_equal" in error_msg
        or "less_than_equal" in error_msg
        or "ge" in error_msg
        or "le" in error_msg
    )


@pytest.mark.unit
@pytest.mark.parametrize("invalid_batch", [0, -1, -16])
def test_tc_tr_06a_training_config_invalid_batch_size_raises(invalid_batch: int) -> None:
    """TC-TR-06a: Verify TrainingConfig rejects batch_size less than 1."""
    with pytest.raises(ValidationError) as exc_info:
        TrainingConfig(data="data.yaml", batch_size=invalid_batch)

    error_msg = str(exc_info.value)
    assert "batch_size" in error_msg


@pytest.mark.unit
@pytest.mark.parametrize("invalid_imgsz", [0, -1, 31, 16])
def test_tc_tr_06b_training_config_invalid_imgsz_raises(invalid_imgsz: int) -> None:
    """TC-TR-06b: Verify TrainingConfig rejects imgsz less than 32 pixels."""
    with pytest.raises(ValidationError) as exc_info:
        TrainingConfig(data="data.yaml", imgsz=invalid_imgsz)

    error_msg = str(exc_info.value)
    assert "imgsz" in error_msg


@pytest.mark.unit
@pytest.mark.parametrize("invalid_lr", [0.0, -0.01, 1.01, 2.0])
def test_tc_tr_07_training_config_invalid_learning_rates_raises(invalid_lr: float) -> None:
    """TC-TR-07: Verify TrainingConfig enforces lr0 and lrf bounds within (0.0, 1.0]."""
    with pytest.raises(ValidationError) as exc_info_lr0:
        TrainingConfig(data="data.yaml", lr0=invalid_lr)
    assert "lr0" in str(exc_info_lr0.value)

    with pytest.raises(ValidationError) as exc_info_lrf:
        TrainingConfig(data="data.yaml", lrf=invalid_lr)
    assert "lrf" in str(exc_info_lrf.value)


@pytest.mark.unit
@pytest.mark.parametrize(
    "field_name",
    ["model_name", "output_dir", "project", "name", "device", "optimizer"],
)
@pytest.mark.parametrize("invalid_value", ["", "   ", "\t\n"])
def test_tc_tr_08_training_config_whitespace_fields_raises(
    field_name: str,
    invalid_value: str,
) -> None:
    """TC-TR-08: Verify TrainingConfig rejects empty or whitespace-only string fields."""
    kwargs = {"data": "data.yaml", field_name: invalid_value}
    with pytest.raises(ValidationError) as exc_info:
        TrainingConfig(**kwargs)

    error_msg = str(exc_info.value)
    assert (
        "cannot be empty or whitespace" in error_msg
        or "string_too_short" in error_msg
        or field_name in error_msg
    )


@pytest.mark.unit
def test_tc_tr_09_training_config_and_result_frozen_immutability() -> None:
    """TC-TR-09: Verify TrainingConfig and TrainingResult enforce frozen immutability."""
    config = TrainingConfig(data="data.yaml")
    with pytest.raises(ValidationError):
        config.epochs = 100  # type: ignore[misc]

    result = TrainingResult(
        model_name="yolov8n.pt",
        dataset_path="data.yaml",
        epochs_completed=5,
    )
    with pytest.raises(ValidationError):
        result.epochs_completed = 10  # type: ignore[misc]


@pytest.mark.unit
def test_tc_tr_10_training_config_and_result_extra_forbidden_and_validation() -> None:
    """TC-TR-10: Verify extra fields are forbidden and TrainingResult bounds are enforced."""
    # Extra field forbidden in TrainingConfig
    with pytest.raises(ValidationError) as exc_info_cfg:
        TrainingConfig(data="data.yaml", unexpected_extra_attr=123)  # type: ignore[call-arg]
    assert "extra_forbidden" in str(exc_info_cfg.value)

    # Extra field forbidden in TrainingResult
    with pytest.raises(ValidationError) as exc_info_res:
        TrainingResult(
            model_name="yolov8n.pt",
            dataset_path="data.yaml",
            epochs_completed=10,
            unexpected_result_attr="bad",  # type: ignore[call-arg]
        )
    assert "extra_forbidden" in str(exc_info_res.value)

    # Negative epochs_completed rejected in TrainingResult
    with pytest.raises(ValidationError):
        TrainingResult(
            model_name="yolov8n.pt",
            dataset_path="data.yaml",
            epochs_completed=-1,
        )

    # Negative training_time_seconds rejected in TrainingResult
    with pytest.raises(ValidationError):
        TrainingResult(
            model_name="yolov8n.pt",
            dataset_path="data.yaml",
            epochs_completed=10,
            training_time_seconds=-0.5,
        )


# ==============================================================================
# Roboflow Dataset Download Tests (TC-TR-11 to TC-TR-13)
# ==============================================================================


@pytest.mark.unit
def test_tc_tr_11_download_roboflow_nominal(tmp_path: Path) -> None:
    """TC-TR-11: Verify nominal Roboflow dataset download and path resolution."""
    config = TrainingConfig(
        roboflow_api_key="mock_api_key",
        roboflow_workspace="mock_workspace",
        roboflow_project="mock_project",
        roboflow_version=1,
    )

    project_dir = tmp_path / "mock_project"
    project_dir.mkdir(parents=True)
    yaml_file = project_dir / "data.yaml"
    yaml_file.write_text("names: ['car', 'person']\nnc: 2\n", encoding="utf-8")

    mock_dataset = MagicMock()
    mock_dataset.location = str(project_dir)

    mock_version = MagicMock()
    mock_version.download.return_value = mock_dataset

    mock_project = MagicMock()
    mock_project.version.return_value = mock_version

    mock_workspace = MagicMock()
    mock_workspace.project.return_value = mock_project

    mock_rf_instance = MagicMock()
    mock_rf_instance.workspace.return_value = mock_workspace

    with patch("roboflow.Roboflow", return_value=mock_rf_instance) as mock_rf_cls:
        resolved_path = download_roboflow_dataset(config, target_dir=tmp_path)

        assert resolved_path == yaml_file.resolve()
        mock_rf_cls.assert_called_once_with(api_key="mock_api_key")
        mock_rf_instance.workspace.assert_called_once_with("mock_workspace")
        mock_workspace.project.assert_called_once_with("mock_project")
        mock_project.version.assert_called_once_with(1)
        mock_version.download.assert_called_once_with(
            model_format="yolov8",
            location=str(tmp_path / "mock_project"),
        )


@pytest.mark.unit
def test_tc_tr_12_download_roboflow_api_failure_raises_runtime_error(tmp_path: Path) -> None:
    """TC-TR-12: Verify Roboflow API authentication/network failures raise RuntimeError."""
    config = TrainingConfig(
        roboflow_api_key="bad_api_key",
        roboflow_workspace="mock_ws",
        roboflow_project="mock_proj",
        roboflow_version=1,
    )

    with patch("roboflow.Roboflow", side_effect=Exception("HTTP 401 Unauthorized")):
        with pytest.raises(RuntimeError) as exc_info:
            download_roboflow_dataset(config, target_dir=tmp_path)

        assert "Roboflow dataset download failed: HTTP 401 Unauthorized" in str(exc_info.value)


@pytest.mark.unit
def test_tc_tr_13_download_roboflow_missing_data_yaml_raises_file_not_found(tmp_path: Path) -> None:
    """TC-TR-13: Verify download without data.yaml manifest raises FileNotFoundError."""
    config = TrainingConfig(
        roboflow_api_key="mock_api_key",
        roboflow_workspace="mock_ws",
        roboflow_project="empty_proj",
        roboflow_version=1,
    )

    empty_dir = tmp_path / "empty_proj"
    empty_dir.mkdir(parents=True)
    # Deliberately do not create data.yaml

    mock_dataset = MagicMock()
    mock_dataset.location = str(empty_dir)

    mock_version = MagicMock()
    mock_version.download.return_value = mock_dataset

    mock_project = MagicMock()
    mock_project.version.return_value = mock_version

    mock_workspace = MagicMock()
    mock_workspace.project.return_value = mock_project

    mock_rf_instance = MagicMock()
    mock_rf_instance.workspace.return_value = mock_workspace

    with patch("roboflow.Roboflow", return_value=mock_rf_instance):
        with pytest.raises(FileNotFoundError) as exc_info:
            download_roboflow_dataset(config, target_dir=tmp_path)

        assert "did not produce a valid data.yaml" in str(exc_info.value)


# ==============================================================================
# run_training Execution Tests (TC-TR-14 to TC-TR-20)
# ==============================================================================


@pytest.mark.unit
def test_tc_tr_14_run_training_dry_run_nominal(tmp_path: Path) -> None:
    """TC-TR-14: Verify dry-run mode validates data paths and skips training execution."""
    data_file = tmp_path / "data.yaml"
    data_file.write_text("names: ['car']\nnc: 1\n", encoding="utf-8")

    config = TrainingConfig(
        data=str(data_file),
        output_dir=str(tmp_path / "models"),
        dry_run=True,
    )

    mock_model = MagicMock()
    result = run_training(config, model=mock_model)

    assert result.success is True
    assert result.dry_run is True
    assert result.epochs_completed == 0
    assert result.best_weight_path is None
    assert result.metrics == {}
    assert result.training_time_seconds == 0.0
    mock_model.train.assert_not_called()


@pytest.mark.unit
def test_tc_tr_15_run_training_dry_run_missing_data_raises(tmp_path: Path) -> None:
    """TC-TR-15: Verify dry-run raises FileNotFoundError when dataset is missing."""
    config = TrainingConfig(
        data=str(tmp_path / "non_existent_manifest.yaml"),
        dry_run=True,
    )

    mock_model = MagicMock()
    with pytest.raises(FileNotFoundError) as exc_info:
        run_training(config, model=mock_model)

    assert "Local dataset path not found" in str(exc_info.value)
    mock_model.train.assert_not_called()


@pytest.mark.unit
def test_tc_tr_16_run_training_nominal_flow(tmp_path: Path) -> None:
    """TC-TR-16: Verify nominal training flow exports best.pt to target models directory."""
    data_file = tmp_path / "data.yaml"
    data_file.write_text("names: ['person', 'dog']\nnc: 2\n", encoding="utf-8")

    project_dir = tmp_path / "runs" / "train"
    models_dir = tmp_path / "models"

    config = TrainingConfig(
        data=str(data_file),
        project=str(project_dir),
        name="custom_detector",
        output_dir=str(models_dir),
        epochs=15,
        batch_size=8,
    )

    # Prepare mock training output directory with weights/best.pt
    run_save_dir = project_dir / "custom_detector"
    weights_dir = run_save_dir / "weights"
    weights_dir.mkdir(parents=True)
    best_weights_src = weights_dir / "best.pt"
    best_weights_src.write_text("mock binary weights content", encoding="utf-8")

    mock_results = MagicMock()
    mock_results.save_dir = str(run_save_dir)
    mock_results.results_dict = {
        "metrics/precision(B)": 0.88,
        "metrics/recall(B)": 0.82,
        "metrics/mAP50(B)": 0.85,
    }

    mock_model = MagicMock()
    mock_model.train.return_value = mock_results

    result = run_training(config, model=mock_model)

    expected_exported_weight = models_dir / "custom_detector_best.pt"

    assert result.success is True
    assert result.dry_run is False
    assert result.epochs_completed == 15
    assert result.best_weight_path == str(expected_exported_weight.resolve())
    assert expected_exported_weight.is_file()
    assert expected_exported_weight.read_text(encoding="utf-8") == "mock binary weights content"
    assert result.metrics["metrics/mAP50(B)"] == 0.85
    assert result.training_time_seconds >= 0.0


@pytest.mark.unit
def test_tc_tr_17_run_training_creates_output_dir(tmp_path: Path) -> None:
    """TC-TR-17: Verify missing output directory is created automatically."""
    data_file = tmp_path / "data.yaml"
    data_file.write_text("names: ['obj']\nnc: 1\n", encoding="utf-8")

    deep_nested_output = tmp_path / "nested" / "sub" / "output_models"
    assert not deep_nested_output.exists()

    config = TrainingConfig(
        data=str(data_file),
        output_dir=str(deep_nested_output),
        dry_run=True,
    )

    mock_model = MagicMock()
    run_training(config, model=mock_model)

    assert deep_nested_output.is_dir()


@pytest.mark.unit
def test_tc_tr_18_run_training_missing_best_pt_raises_file_not_found(tmp_path: Path) -> None:
    """TC-TR-18: Verify missing best.pt artifact after training raises FileNotFoundError."""
    data_file = tmp_path / "data.yaml"
    data_file.write_text("names: ['obj']\nnc: 1\n", encoding="utf-8")

    config = TrainingConfig(
        data=str(data_file),
        project=str(tmp_path / "runs"),
        name="missing_weights_exp",
        output_dir=str(tmp_path / "models"),
    )

    mock_results = MagicMock()
    mock_results.save_dir = str(tmp_path / "runs" / "missing_weights_exp")

    mock_model = MagicMock()
    mock_model.train.return_value = mock_results

    with pytest.raises(FileNotFoundError) as exc_info:
        run_training(config, model=mock_model)

    assert "best weights were not generated" in str(exc_info.value)


@pytest.mark.unit
def test_tc_tr_19_run_training_hyperparameter_mapping(tmp_path: Path) -> None:
    """TC-TR-19: Verify complete hyperparameter mapping to yolo_model.train() call."""
    data_file = tmp_path / "data.yaml"
    data_file.write_text("names: ['obj']\nnc: 1\n", encoding="utf-8")

    project_dir = tmp_path / "runs"
    weights_dir = project_dir / "hyperparam_exp" / "weights"
    weights_dir.mkdir(parents=True)
    (weights_dir / "best.pt").write_text("dummy", encoding="utf-8")

    config = TrainingConfig(
        data=str(data_file),
        model_name="yolov8s.pt",
        epochs=35,
        batch_size=32,
        imgsz=512,
        device="cpu",
        workers=4,
        patience=7,
        optimizer="AdamW",
        lr0=0.005,
        lrf=0.02,
        seed=123,
        project=str(project_dir),
        name="hyperparam_exp",
        output_dir=str(tmp_path / "models"),
        exist_ok=True,
        resume=False,
    )

    mock_results = MagicMock()
    mock_results.save_dir = str(project_dir / "hyperparam_exp")

    mock_model = MagicMock()
    mock_model.train.return_value = mock_results

    run_training(config, model=mock_model)

    mock_model.train.assert_called_once_with(
        data=str(data_file.resolve()),
        epochs=35,
        batch=32,
        imgsz=512,
        device="cpu",
        workers=4,
        patience=7,
        optimizer="AdamW",
        lr0=0.005,
        lrf=0.02,
        seed=123,
        project=str(project_dir),
        name="hyperparam_exp",
        exist_ok=True,
        resume=False,
        verbose=True,
    )


@pytest.mark.unit
def test_tc_tr_20_run_training_trainer_exception_propagates(tmp_path: Path) -> None:
    """TC-TR-20: Verify trainer execution exceptions propagate unsuppressed."""
    data_file = tmp_path / "data.yaml"
    data_file.write_text("names: ['obj']\nnc: 1\n", encoding="utf-8")

    config = TrainingConfig(
        data=str(data_file),
        output_dir=str(tmp_path / "models"),
    )

    mock_model = MagicMock()
    mock_model.train.side_effect = RuntimeError("CUDA device out of memory during backward pass")

    with pytest.raises(RuntimeError) as exc_info:
        run_training(config, model=mock_model)

    assert "CUDA device out of memory" in str(exc_info.value)


# ==============================================================================
# CLI Argument Parsing & main() Tests (TC-TR-21 to TC-TR-25)
# ==============================================================================


@pytest.mark.unit
def test_tc_tr_21_build_parser_nominal_flags() -> None:
    """TC-TR-21: Verify CLI argument parser correctly parses nominal training flags."""
    parser = build_parser()
    args = parser.parse_args(
        [
            "--model",
            "yolov8m.pt",
            "--data",
            "datasets/coco.yaml",
            "--epochs",
            "40",
            "--batch",
            "8",
            "--imgsz",
            "320",
            "--device",
            "0",
            "--workers",
            "4",
            "--lr0",
            "0.002",
            "--project",
            "custom_runs",
            "--name",
            "run_01",
            "--output-dir",
            "saved_weights",
            "--dry-run",
        ]
    )

    assert args.model_name == "yolov8m.pt"
    assert args.data == "datasets/coco.yaml"
    assert args.epochs == 40
    assert args.batch_size == 8
    assert args.imgsz == 320
    assert args.device == "0"
    assert args.workers == 4
    assert args.lr0 == 0.002
    assert args.project == "custom_runs"
    assert args.name == "run_01"
    assert args.output_dir == "saved_weights"
    assert args.dry_run is True


@pytest.mark.unit
def test_tc_tr_22_build_parser_roboflow_flags() -> None:
    """TC-TR-22: Verify CLI argument parser handles Roboflow credentials and flags."""
    parser = build_parser()
    args = parser.parse_args(
        [
            "--roboflow-api-key",
            "api_key_secret",
            "--roboflow-workspace",
            "ws_slug",
            "--roboflow-project",
            "proj_slug",
            "--roboflow-version",
            "4",
        ]
    )

    assert args.roboflow_api_key == "api_key_secret"
    assert args.roboflow_workspace == "ws_slug"
    assert args.roboflow_project == "proj_slug"
    assert args.roboflow_version == 4


@pytest.mark.unit
def test_tc_tr_23_main_dry_run_nominal(tmp_path: Path) -> None:
    """TC-TR-23: Verify main entrypoint exits with code 0 on successful dry-run."""
    data_file = tmp_path / "data.yaml"
    data_file.write_text("names: ['obj']\nnc: 1\n", encoding="utf-8")

    with patch("src.train.YOLO"):
        exit_code = main(["--data", str(data_file), "--dry-run"])

    assert exit_code == 0


@pytest.mark.unit
def test_tc_tr_24_main_validation_error_returns_1(tmp_path: Path) -> None:
    """TC-TR-24: Verify main entrypoint exits with code 1 when configuration validation fails."""
    data_file = tmp_path / "data.yaml"
    data_file.write_text("names: ['obj']\nnc: 1\n", encoding="utf-8")

    exit_code = main(["--data", str(data_file), "--epochs", "0"])
    assert exit_code == 1


@pytest.mark.unit
def test_tc_tr_25_main_runtime_error_returns_1(tmp_path: Path) -> None:
    """TC-TR-25: Verify main entrypoint exits with code 1 on runtime error (e.g., missing data)."""
    exit_code = main(["--data", str(tmp_path / "non_existent_dataset.yaml")])
    assert exit_code == 1


# ==============================================================================
# Helper & Dataset Path Resolution Tests
# ==============================================================================


@pytest.mark.unit
def test_resolve_dataset_path_directory_containing_data_yaml(tmp_path: Path) -> None:
    """Verify resolve_dataset_path resolves data.yaml inside a provided directory."""
    data_dir = tmp_path / "custom_dataset"
    data_dir.mkdir(parents=True)
    yaml_file = data_dir / "data.yaml"
    yaml_file.write_text("names: ['sample']\nnc: 1\n", encoding="utf-8")

    config = TrainingConfig(data=str(data_dir))
    resolved = resolve_dataset_path(config)

    assert resolved == yaml_file.resolve()
