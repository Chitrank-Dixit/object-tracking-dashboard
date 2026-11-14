# Project Specification: Real-Time Object Detection & Tracking System

## 1. Objective
Build a containerized, real-time object detection, multi-object tracking, and spatial analytics dashboard using PyTorch, Ultralytics (YOLO26/YOLOv8), ByteTrack, OpenCV, Pydantic, and Streamlit.

## 2. Tech Stack & Constraints
- **Runtime:** Python 3.11+ inside Podman (`Containerfile` + `compose.yaml`).
- **Package Manager:** `uv` (`pyproject.toml` + `uv.lock`).
- **Validation:** Pydantic v2 (`src/schemas.py`).
- **Linting/Formatting:** Ruff (`ruff check`, `ruff format`).
- **Testing:** Pytest (`tests/`).
- **CV/ML:** `ultralytics`, `torch`, `opencv-python-headless`, `roboflow`.
- **UI:** `streamlit`.

## 3. Core Architecture & Data Contracts (`src/schemas.py`)
All module boundaries must exchange validated Pydantic models:
- `DetectorConfig`: `model_path` (str), `confidence_threshold` (float, 0.0–1.0), `iou_threshold` (float, 0.0–1.0), `target_classes` (list[int] | None), `tracker_type` (Literal["bytetrack.yaml", "botsort.yaml"]).
- `TrackedObject`: `track_id` (int), `class_id` (int), `class_name` (str), `confidence` (float), `bbox_xyxy` (tuple[float, float, float, float]), `centroid` (tuple[float, float]).
- `FrameResult`: `frame_index` (int), `inference_ms` (float), `fps` (float), `objects` (list[TrackedObject]).
- `CountingLine`: `name` (str), `start_point` (tuple[int, int]), `end_point` (tuple[int, int]).
- `CountingZone`: `name` (str), `polygon` (list[tuple[int, int]]).
- `AnalyticsSummary`: `total_in` (int), `total_out` (int), `zone_occupancy` (dict[str, int]), `class_counts` (dict[str, int]).

## 4. Milestones & Acceptance Criteria

### Milestone 1: Containerized Foundation & Schemas
- **Deliverables:** `pyproject.toml`, `Containerfile`, `compose.yaml`, `src/schemas.py`, `tests/test_schemas.py`.
- **Acceptance Criteria:** `podman compose build` succeeds; `pytest` validates Pydantic schema constraints (rejecting invalid confidence scores or malformed polygons); `ruff` passes.

### Milestone 2: Detection & ByteTrack Inference Engine (`src/tracker.py`)
- **Deliverables:** `ObjectTracker` class wrapping Ultralytics `.track(persist=True, tracker="bytetrack.yaml")` and returning `FrameResult`.
- **Acceptance Criteria:** Processes `data/sample_video.mp4` without GUI dependencies; maintains persistent `track_id` across frames; unit tests verify output parsing using mocked/synthetic frames.

### Milestone 3: Line-Crossing & Zone-Counting Analytics (`src/counter.py`)
- **Deliverables:** `SpatialAnalytics` engine computing vector cross-product line crossings (directional IN/OUT per `track_id`, preventing double-counts) and point-in-polygon zone occupancy.
- **Acceptance Criteria:** Synthetic trajectory tests in `tests/test_counter.py` prove accurate IN/OUT counts, zero duplicate triggers for stationary/jittering IDs, and accurate zone occupancy counts.

### Milestone 4: Custom Dataset Fine-Tuning Pipeline (`src/train.py`)
- **Deliverables:** CLI-driven fine-tuning script supporting Roboflow dataset download and Ultralytics YOLO training, saving best weights to `models/`.
- **Acceptance Criteria:** Dry-run/smoke test validates config parsing and dataset path resolution without requiring full GPU training during CI.

### Milestone 5: Streamlit Live Analytics Dashboard (`app.py`)
- **Deliverables:** Interactive Streamlit UI supporting sample video upload/selection and live webcam stream, real-time bounding box + trajectory + line/zone overlay rendering, sidebar controls (confidence, model selector, line/zone coordinates), and live metrics (FPS, IN/OUT counts, zone occupancy).
- **Acceptance Criteria:** Runs cleanly at `http://localhost:8501` via `podman compose up`; frame loop handles start/stop gracefully without crashing or leaking memory.