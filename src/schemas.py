"""Domain data contracts and validation schemas for Object Tracking Dashboard.

All models adhere strictly to Pydantic v2 standards with frozen immutability
and strict attribute validation.
"""

from __future__ import annotations

import json
import math
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class DetectorConfig(BaseModel):
    """Configuration contract for YOLO detector and ByteTrack/BoT-SORT tracker."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    model_path: str = Field(
        default="yolov8n.pt",
        min_length=1,
        description="Path or model identifier for YOLO weights",
    )
    confidence_threshold: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Minimum detection confidence score threshold (0.0 to 1.0)",
    )
    iou_threshold: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="Non-maximum suppression IoU threshold (0.0 to 1.0)",
    )
    target_classes: list[int] | None = Field(
        default=None,
        description="Optional list of COCO class indices to filter detections. None tracks all.",
    )
    tracker_type: Literal["bytetrack.yaml", "botsort.yaml"] = Field(
        default="bytetrack.yaml",
        description="Ultralytics tracker configuration file",
    )

    @field_validator("model_path")
    @classmethod
    def validate_model_path(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("model_path cannot be an empty or whitespace string")
        return v.strip()

    @field_validator("target_classes")
    @classmethod
    def validate_target_classes(cls, v: list[int] | None) -> list[int] | None:
        if v is not None:
            if not v:
                raise ValueError(
                    "target_classes cannot be an empty list; use None to track all classes"
                )
            for class_id in v:
                if class_id < 0:
                    raise ValueError(
                        f"Class IDs must be non-negative integers; received {class_id}"
                    )
        return v


class TrackedObject(BaseModel):
    """Schema representing an individual detected and tracked object in a frame."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    track_id: int = Field(ge=0, description="Unique persistent identifier assigned by tracker")
    class_id: int = Field(
        ge=0, description="Integer class identifier corresponding to model ontology"
    )
    class_name: str = Field(min_length=1, description="Human-readable category name")
    confidence: float = Field(ge=0.0, le=1.0, description="Detection confidence score")
    bbox_xyxy: tuple[float, float, float, float] = Field(
        description="Bounding box coordinates in pixels as (x1, y1, x2, y2)"
    )
    centroid: tuple[float, float] | None = Field(
        default=None,
        description="Center point (cx, cy). Automatically computed from bbox_xyxy if omitted.",
    )

    @field_validator("class_name")
    @classmethod
    def validate_class_name(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("class_name cannot be empty or whitespace")
        return v.strip()

    @field_validator("bbox_xyxy")
    @classmethod
    def validate_bbox(
        cls, v: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = v
        if any(c < 0.0 for c in (x1, y1, x2, y2)):
            raise ValueError(
                f"Bounding box coordinates must be non-negative: ({x1}, {y1}, {x2}, {y2})"
            )
        if x1 >= x2:
            raise ValueError(
                f"Invalid bounding box: x1 ({x1}) must be strictly less than x2 ({x2})"
            )
        if y1 >= y2:
            raise ValueError(
                f"Invalid bounding box: y1 ({y1}) must be strictly less than y2 ({y2})"
            )
        return v

    @model_validator(mode="after")
    def compute_or_validate_centroid(self) -> TrackedObject:
        x1, y1, x2, y2 = self.bbox_xyxy
        expected_cx = (x1 + x2) / 2.0
        expected_cy = (y1 + y2) / 2.0

        if self.centroid is None:
            object.__setattr__(self, "centroid", (expected_cx, expected_cy))
        else:
            cx, cy = self.centroid
            if not (
                math.isclose(cx, expected_cx, abs_tol=1e-2)
                and math.isclose(cy, expected_cy, abs_tol=1e-2)
            ):
                raise ValueError(
                    f"Centroid ({cx:.2f}, {cy:.2f}) does not match bbox center "
                    f"({expected_cx:.2f}, {expected_cy:.2f}) within tolerance"
                )
        return self


class FrameResult(BaseModel):
    """Payload representing tracking inference and timing results for a single video frame."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    frame_index: int = Field(ge=0, description="Zero-indexed monotonic frame sequence number")
    inference_ms: float = Field(
        ge=0.0, description="Ultralytics inference duration in milliseconds"
    )
    fps: float = Field(
        ge=0.0, description="Calculated instantaneous processing rate in frames per second"
    )
    objects: list[TrackedObject] = Field(
        default_factory=list,
        description="Collection of tracked objects present in the current frame",
    )


class CountingLine(BaseModel):
    """Virtual tripwire line defined by start and end points for directional crossing detection."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    name: str = Field(min_length=1, description="Unique human-readable identifier for the line")
    start_point: tuple[int, int] = Field(description="Starting coordinate (x, y) in integer pixels")
    end_point: tuple[int, int] = Field(description="Ending coordinate (x, y) in integer pixels")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("CountingLine name cannot be empty or whitespace")
        return v.strip()

    @field_validator("start_point", "end_point")
    @classmethod
    def validate_non_negative_coords(cls, v: tuple[int, int]) -> tuple[int, int]:
        x, y = v
        if x < 0 or y < 0:
            raise ValueError(f"Line coordinates must be non-negative: ({x}, {y})")
        return v

    @model_validator(mode="after")
    def validate_non_degenerate_line(self) -> CountingLine:
        if self.start_point == self.end_point:
            raise ValueError(
                f"Line '{self.name}' is degenerate: start_point {self.start_point} and "
                f"end_point {self.end_point} cannot be identical"
            )
        return self


class CountingZone(BaseModel):
    """Virtual region-of-interest polygon for point-in-polygon occupancy analytics."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    name: str = Field(min_length=1, description="Unique human-readable identifier for the zone")
    polygon: list[tuple[int, int]] = Field(
        min_length=3,
        description="Ordered sequence of polygon vertices (x, y) defining the region",
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("CountingZone name cannot be empty or whitespace")
        return v.strip()

    @field_validator("polygon")
    @classmethod
    def validate_polygon_vertices(cls, v: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if len(v) < 3:
            raise ValueError(f"Polygon must have at least 3 vertices; received {len(v)}")
        for x, y in v:
            if x < 0 or y < 0:
                raise ValueError(f"Polygon vertex coordinates must be non-negative: ({x}, {y})")
        unique_vertices = set(v)
        if len(unique_vertices) < 3:
            raise ValueError(
                f"Polygon must contain at least 3 unique vertices; found {len(unique_vertices)}"
            )
        return v

    @model_validator(mode="after")
    def validate_non_collinear_polygon(self) -> CountingZone:
        points = self.polygon
        x0, y0 = points[0]
        x1, y1 = points[1]
        dx0 = x1 - x0
        dy0 = y1 - y0

        has_non_collinear = False
        for i in range(2, len(points)):
            xi, yi = points[i]
            cross_product = dx0 * (yi - y0) - dy0 * (xi - x0)
            if cross_product != 0:
                has_non_collinear = True
                break

        if not has_non_collinear:
            raise ValueError(
                f"Polygon '{self.name}' is degenerate: all vertices are collinear (zero area)"
            )
        return self


class AnalyticsSummary(BaseModel):
    """Aggregated spatial metrics reflecting cumulative crossings and real-time occupancies."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    total_in: int = Field(
        default=0, ge=0, description="Cumulative count of objects entering boundary"
    )
    total_out: int = Field(
        default=0, ge=0, description="Cumulative count of objects exiting boundary"
    )
    zone_occupancy: dict[str, int] = Field(
        default_factory=dict,
        description="Mapping of zone name to current object occupancy count",
    )
    class_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Mapping of class name to current active tracked object count",
    )

    @field_validator("zone_occupancy", "class_counts")
    @classmethod
    def validate_counts_non_negative(cls, v: dict[str, int]) -> dict[str, int]:
        for key, count in v.items():
            if count < 0:
                raise ValueError(
                    f"Occupancy and class counts cannot be negative. Key '{key}' has count {count}"
                )
        return v


class LineCrossingEvent(BaseModel):
    """Immutable event record representing a directional line crossing."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    track_id: int = Field(ge=0, description="Tracker ID of the crossing entity")
    line_name: str = Field(min_length=1, description="Name of the line crossed")
    direction: Literal["IN", "OUT"] = Field(
        description="Direction of crossing relative to vector A->B"
    )
    frame_index: int = Field(ge=0, description="Frame sequence index when crossing was confirmed")
    class_name: str = Field(min_length=1, description="Object class category name")
    timestamp_ms: float = Field(
        default=0.0, ge=0.0, description="Elapsed timestamp in milliseconds"
    )


class SpatialAnalyticsConfig(BaseModel):
    """Configuration contract for SpatialAnalytics engine thresholds and buffers."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    debounce_distance: float = Field(
        default=3.0,
        ge=0.0,
        description="Pixel hysteresis deadband margin from tripwire line",
    )
    cooldown_frames: int = Field(
        default=5,
        ge=0,
        description="Minimum frames between successive crossings for the same ID",
    )
    max_disappeared_frames: int = Field(
        default=60,
        ge=1,
        description="Maximum inactive frames before pruning track state",
    )
    trajectory_history_length: int = Field(
        default=30,
        ge=2,
        description="Maximum historical centroid positions to retain per track",
    )


class TrainingConfig(BaseModel):
    """Configuration contract for YOLO fine-tuning and Roboflow dataset ingestion."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    model_name: str = Field(
        default="yolov8n.pt",
        min_length=1,
        description="Path or identifier for base YOLO weights (e.g., yolov8n.pt, yolov8s.pt)",
    )
    data: str | None = Field(
        default=None,
        description="Path to local dataset YAML file or directory containing data.yaml",
    )

    roboflow_api_key: str | None = Field(
        default=None,
        description="Roboflow private API key for dataset access",
    )
    roboflow_workspace: str | None = Field(
        default=None,
        description="Roboflow workspace URL slug",
    )
    roboflow_project: str | None = Field(
        default=None,
        description="Roboflow project URL slug",
    )
    roboflow_version: int | None = Field(
        default=None,
        ge=1,
        description="Roboflow dataset version integer number",
    )
    roboflow_model_format: str = Field(
        default="yolov8",
        min_length=1,
        description="Export format expected from Roboflow download",
    )

    epochs: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Total training epochs",
    )
    batch_size: int = Field(
        default=16,
        ge=1,
        description="Batch size per training step",
    )
    imgsz: int = Field(
        default=640,
        ge=32,
        description="Input square image resolution in pixels",
    )
    device: str = Field(
        default="cpu",
        min_length=1,
        description="Compute target device ('cpu', '0', 'cuda')",
    )
    workers: int = Field(
        default=2,
        ge=0,
        description="Number of worker threads for DataLoader",
    )
    patience: int = Field(
        default=10,
        ge=0,
        description="Early stopping patience in epochs",
    )
    optimizer: str = Field(
        default="auto",
        min_length=1,
        description="Optimization algorithm ('auto', 'SGD', 'Adam', 'AdamW')",
    )
    lr0: float = Field(
        default=0.01,
        gt=0.0,
        le=1.0,
        description="Initial learning rate",
    )
    lrf: float = Field(
        default=0.01,
        gt=0.0,
        le=1.0,
        description="Final learning rate fraction relative to lr0",
    )
    seed: int = Field(
        default=42,
        ge=0,
        description="Deterministic random seed",
    )

    project: str = Field(
        default="runs/train",
        min_length=1,
        description="Directory for training run outputs and tensorboard logs",
    )
    name: str = Field(
        default="exp",
        min_length=1,
        description="Subdirectory name for run artifacts",
    )
    output_dir: str = Field(
        default="models",
        min_length=1,
        description="Directory where best exported weights are saved",
    )
    dry_run: bool = Field(
        default=False,
        description="If True, validates config and data paths without running actual training",
    )
    exist_ok: bool = Field(
        default=True,
        description="Whether to overwrite existing experiment directory",
    )
    resume: bool = Field(
        default=False,
        description="Whether to resume training from the last saved checkpoint",
    )

    @field_validator("model_name", "output_dir", "project", "name", "device", "optimizer")
    @classmethod
    def validate_non_empty_strings(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field cannot be empty or whitespace")
        return v.strip()

    @model_validator(mode="after")
    def validate_dataset_source(self) -> TrainingConfig:
        has_local_data = self.data is not None and bool(self.data.strip())
        rf_fields = (
            self.roboflow_api_key,
            self.roboflow_workspace,
            self.roboflow_project,
            self.roboflow_version,
        )
        has_any_rf = any(f is not None for f in rf_fields)
        has_all_rf = all(f is not None for f in rf_fields)

        if not has_local_data and not has_all_rf:
            if has_any_rf:
                raise ValueError(
                    "Roboflow download requires all four parameters: roboflow_api_key, "
                    "roboflow_workspace, roboflow_project, and roboflow_version."
                )
            raise ValueError(
                "Either a local dataset path ('data') or complete Roboflow credentials must be specified."
            )

        return self


class TrainingResult(BaseModel):
    """Immutable result contract summarizing YOLO fine-tuning outcomes."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    success: bool = Field(
        default=True,
        description="Whether fine-tuning pipeline completed without errors",
    )
    model_name: str = Field(
        min_length=1,
        description="Base model name or weights file utilized",
    )
    dataset_path: str = Field(
        min_length=1,
        description="Resolved path to the dataset YAML file",
    )
    epochs_completed: int = Field(
        ge=0,
        description="Number of completed training epochs (0 for dry-run)",
    )
    best_weight_path: str | None = Field(
        default=None,
        description="Path to the exported best model weights in models/ directory",
    )
    metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Evaluation metrics captured at the conclusion of training",
    )
    dry_run: bool = Field(
        default=False,
        description="Whether execution was performed in dry-run validation mode",
    )
    training_time_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Total wall-clock duration of training in seconds",
    )


DEFAULT_LINES_JSON: str = json.dumps(
    [
        {
            "name": "Tripwire-1",
            "start_point": [100, 300],
            "end_point": [540, 300],
        }
    ],
    indent=2,
)

DEFAULT_ZONES_JSON: str = json.dumps(
    [
        {
            "name": "Zone-A",
            "polygon": [
                [150, 150],
                [450, 150],
                [450, 350],
                [150, 350],
            ],
        }
    ],
    indent=2,
)


class DashboardConfig(BaseModel):
    """Runtime configuration contract for Streamlit live analytics dashboard."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    model_path: str = Field(
        default="yolov8n.pt",
        min_length=1,
        description="Path or identifier for YOLO weights",
    )
    confidence_threshold: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Detection confidence score threshold (0.0 to 1.0)",
    )
    iou_threshold: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="Non-maximum suppression IoU threshold (0.0 to 1.0)",
    )
    tracker_type: Literal["bytetrack.yaml", "botsort.yaml"] = Field(
        default="bytetrack.yaml",
        description="Tracking algorithm configuration file",
    )
    target_classes: list[int] | None = Field(
        default=None,
        description="COCO class indices to track. None tracks all classes.",
    )
    source_type: Literal["Sample Video", "Upload Video", "Webcam / RTSP"] = Field(
        default="Sample Video",
        description="Input video source category",
    )
    source_path: str = Field(
        default="data/sample_video.mp4",
        min_length=1,
        description="Resolved video source path or RTSP stream URL",
    )
    lines: list[CountingLine] = Field(
        default_factory=list,
        description="Active tripwire counting lines",
    )
    zones: list[CountingZone] = Field(
        default_factory=list,
        description="Active polygon counting zones",
    )
    draw_boxes: bool = Field(default=True, description="Render bounding boxes and IDs")
    draw_trajectories: bool = Field(default=True, description="Render centroid trajectories")
    draw_lines: bool = Field(default=True, description="Render tripwire lines")
    draw_zones: bool = Field(default=True, description="Render polygon zones")
    draw_hud: bool = Field(default=True, description="Render HUD summary card")

    @field_validator("model_path", "source_path")
    @classmethod
    def validate_non_empty_paths(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Path cannot be empty or whitespace")
        return v.strip()
