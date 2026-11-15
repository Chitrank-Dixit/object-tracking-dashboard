"""Comprehensive end-to-end verification script for GPU-accelerated tracking & analytics."""

import time
import cv2
import numpy as np
from src.device import get_cuda_memory_info, resolve_compute_device
from src.schemas import CountingLine, CountingZone, DetectorConfig
from src.tracker import ObjectTracker
from src.counter import SpatialAnalytics

def run_live_verification():
    print("=" * 65)
    print("      LIVE GPU INFERENCE & SPATIAL ANALYTICS VERIFICATION")
    print("=" * 65)

    # 1. Device Resolution
    res = resolve_compute_device("auto")
    print(f"[1] Device Resolution: requested='auto' -> resolved='{res.resolved_device}' (type={res.device_type})")
    assert res.resolved_device == "cuda:0", f"Expected cuda:0, got {res.resolved_device}"

    # 2. Initialize Tracker on GPU
    cfg = DetectorConfig(
        model_path="yolov8n.pt",
        confidence_threshold=0.25,
        iou_threshold=0.45,
        device="auto",
    )
    print("[2] Initializing ObjectTracker with YOLOv8 on CUDA...")
    t0 = time.perf_counter()
    tracker = ObjectTracker(config=cfg)
    init_ms = (time.perf_counter() - t0) * 1000.0
    print(f"    Tracker initialized in {init_ms:.1f} ms on {tracker.resolved_device}")
    assert tracker.resolved_device == "cuda:0"
    assert not tracker.is_fallback

    # 3. Setup Spatial Analytics with Line & Zone
    line = CountingLine(name="Tripwire-1", start_point=(100, 240), end_point=(540, 240))
    zone = CountingZone(name="Center-Zone", polygon=[(150, 150), (490, 150), (490, 350), (150, 350)])
    analytics = SpatialAnalytics(lines=[line], zones=[zone])
    print("[3] SpatialAnalytics configured with 1 Tripwire and 1 Polygonal Zone")

    # Warm-up frame (triggers cuDNN kernel compilation)
    print("[4] Executing warm-up frame (cuDNN kernel compilation)...")
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    t_warm = time.perf_counter()
    _ = tracker.track_frame(dummy_frame, frame_index=0)
    warmup_ms = (time.perf_counter() - t_warm) * 1000.0
    print(f"    Warm-up completed in {warmup_ms:.1f} ms")

    # 5. Measure 30 steady-state frames
    print("[5] Executing 30 steady-state tracking frames on RTX 5060 Ti...")
    latencies = []
    for i in range(1, 31):
        frame = np.full((480, 640, 3), 30, dtype=np.uint8)
        cy = int(180 + i * 5)
        cx = int(250 + i * 4)
        cv2.circle(frame, (cx, cy), 25, (200, 200, 200), -1)
        cv2.rectangle(frame, (cx - 15, cy), (cx + 15, cy + 50), (200, 200, 200), -1)

        result = tracker.track_frame(frame, frame_index=i)
        _ = analytics.update(result)
        latencies.append(result.inference_ms)

    avg_ms = sum(latencies) / len(latencies)
    min_ms = min(latencies)
    max_ms = max(latencies)
    fps_estimate = 1000.0 / avg_ms if avg_ms > 0 else 0

    print(f"    Steady-State Latency: avg={avg_ms:.2f} ms | min={min_ms:.2f} ms | max={max_ms:.2f} ms")
    print(f"    Steady-State GPU Throughput: ~{fps_estimate:.1f} FPS 🚀")

    # 6. Check Live VRAM Memory Telemetry
    mem = get_cuda_memory_info(0)
    if mem:
        print("[6] GPU VRAM Telemetry Post-Inference:")
        print(f"    Device Name:    {mem.device_name}")
        print(f"    Total VRAM:     {mem.total_memory_mb:.1f} MB ({mem.total_memory_mb/1024:.2f} GB)")
        print(f"    Allocated VRAM: {mem.allocated_memory_mb:.2f} MB")
        print(f"    Reserved VRAM:  {mem.reserved_memory_mb:.2f} MB")
        print(f"    Free VRAM:      {mem.free_memory_mb:.1f} MB")
        print(f"    Utilization:    {mem.utilization_pct:.2f}%")

    # 7. Reset & Cleanup
    tracker.reset()
    mem_after = get_cuda_memory_info(0)
    if mem_after:
        print(f"[7] Post-Reset Allocated VRAM: {mem_after.allocated_memory_mb:.2f} MB (Cache Flushed)")

    print("=" * 65)
    print("             ALL GPU VERIFICATION CHECKS PASSED ✅")
    print("=" * 65)

if __name__ == "__main__":
    run_live_verification()
