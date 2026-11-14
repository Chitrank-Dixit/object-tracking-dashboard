"""Comprehensive test suite for Line-Crossing & Zone-Counting Analytics (src/counter.py).

Implements all 20 test cases (TC-CNT-01 through TC-CNT-20) verifying SpatialAnalytics:
- TC-CNT-01: Nominal Single Crossing IN (Left -> Right)
- TC-CNT-02: Nominal Single Crossing OUT (Right -> Left)
- TC-CNT-03: Vertical Line Crossings (handles vertical vectors without division-by-zero)
- TC-CNT-04: Diagonal Line Crossings
- TC-CNT-05: Crossing Infinite Line Extension (Near-Miss outside segment endpoints)
- TC-CNT-06: Stationary Object on Tripwire (0 triggers over 30 frames)
- TC-CNT-07: Jitter Rejection Across Line (Sub-pixel noise within deadband suppresses duplicate triggers)
- TC-CNT-08: U-Turn Near Tripwire (approaches, turns back without crossing)
- TC-CNT-09: Movement Parallel to Line (constant cross product sign, 0 triggers)
- TC-CNT-10: Consecutive Round-Trip (IN then OUT with cooldown, 1 IN and 1 OUT)
- TC-CNT-11: Multiple Concurrent Tracks (isolated state machines across different IDs)
- TC-CNT-12: Convex Zone Occupancy (inside and vertex counted, outside excluded)
- TC-CNT-13: Non-Convex (Concave L-Shaped) Zone (notch point excluded, leg points counted)
- TC-CNT-14: Zone Dynamic Entry, Dwell, Exit (sequence: 0 -> 1 -> 1 -> 1 -> 0)
- TC-CNT-15: Multiple Overlapping / Disjoint Zones (point in overlap counted in both)
- TC-CNT-16: Real-Time Metrics & Class Breakdown (AnalyticsSummary with total_in, zone_occupancy, class_counts)
- TC-CNT-17: Memory Lifecycle / Inactivity GC (stale tracks pruned after max_disappeared_frames)
- TC-CNT-18: Headless Visualization Immutability (annotate_analytics returns new array; input frame is unmutated)
- TC-CNT-19: Malformed Frame & Contract Validation (invalid frame arrays raise ValueError)
- TC-CNT-20: Engine Reset (reset() zeroes counts, empties events and states)
"""

from __future__ import annotations

import numpy as np
import pytest

from src.counter import (
    SpatialAnalytics,
    _cross_2d,
    _on_segment,
    _segments_intersect,
)
from src.schemas import (
    AnalyticsSummary,
    CountingLine,
    CountingZone,
    FrameResult,
    SpatialAnalyticsConfig,
    TrackedObject,
)


def _make_frame_result(
    frame_index: int,
    objects_data: list[tuple[int, str, float, float]],
    fps: float = 30.0,
    inference_ms: float = 12.0,
) -> FrameResult:
    """Helper to synthesize deterministic FrameResult instances.

    Args:
        frame_index: Sequence frame number.
        objects_data: List of tuples (track_id, class_name, centroid_x, centroid_y).
        fps: Optional frame rate.
        inference_ms: Optional inference duration.
    """
    tracked_objects: list[TrackedObject] = []
    for track_id, class_name, cx, cy in objects_data:
        box_w, box_h = 20.0, 40.0
        x1 = cx - box_w / 2.0
        y1 = cy - box_h / 2.0
        x2 = cx + box_w / 2.0
        y2 = cy + box_h / 2.0
        tracked_objects.append(
            TrackedObject(
                track_id=track_id,
                class_id=0 if class_name == "person" else 1,
                class_name=class_name,
                confidence=0.90,
                bbox_xyxy=(x1, y1, x2, y2),
                centroid=(cx, cy),
            )
        )
    return FrameResult(
        frame_index=frame_index,
        inference_ms=inference_ms,
        fps=fps,
        objects=tracked_objects,
    )


# ==============================================================================
# Geometric Primitives Unit Tests
# ==============================================================================


@pytest.mark.unit
def test_geometry_cross_2d_and_segment_intersection() -> None:
    """Verify geometric intersection and cross-product primitives."""
    p0 = (0.0, 0.0)
    p1 = (10.0, 0.0)
    # Right of directed vector (0,0) -> (10,0) in image coords (y down) is y > 0
    assert _cross_2d(p0, p1, (5.0, 5.0)) > 0.0
    # Left is y < 0
    assert _cross_2d(p0, p1, (5.0, -5.0)) < 0.0
    # Collinear
    assert _cross_2d(p0, p1, (5.0, 0.0)) == 0.0

    # Strict intersection
    assert _segments_intersect((0.0, 5.0), (10.0, 5.0), (5.0, 0.0), (5.0, 10.0)) is True
    # Parallel disjoint
    assert _segments_intersect((0.0, 0.0), (10.0, 0.0), (0.0, 5.0), (10.0, 5.0)) is False
    # Touching boundary
    assert _segments_intersect((0.0, 0.0), (10.0, 0.0), (5.0, 0.0), (5.0, 5.0)) is True
    # On segment helper
    assert _on_segment((0.0, 0.0), (10.0, 0.0), (5.0, 0.0)) is True
    assert _on_segment((0.0, 0.0), (10.0, 0.0), (15.0, 0.0)) is False


# ==============================================================================
# TC-CNT-01: Nominal Single Crossing IN (Left -> Right)
# ==============================================================================


@pytest.mark.unit
def test_tc_cnt_01_nominal_single_crossing_in() -> None:
    """TC-CNT-01: Verify track moving from LEFT to RIGHT of tripwire triggers IN crossing."""
    # Line directed from (200, 400) to (200, 100)
    # Vector points up; left of this vector is x < 200, right is x > 200
    line = CountingLine(name="EntryGate", start_point=(200, 400), end_point=(200, 100))
    analytics = SpatialAnalytics(lines=[line])

    # Frame 0: Track 1 starts at x=150 (LEFT side)
    fr0 = _make_frame_result(0, [(1, "person", 150.0, 250.0)])
    summary0 = analytics.update(fr0)
    assert summary0.total_in == 0
    assert summary0.total_out == 0

    # Frame 1: Track 1 moves to x=250 (RIGHT side), crossing x=200
    fr1 = _make_frame_result(1, [(1, "person", 250.0, 250.0)])
    summary1 = analytics.update(fr1)

    assert summary1.total_in == 1
    assert summary1.total_out == 0
    assert analytics.get_line_counts("EntryGate") == {"EntryGate": {"in": 1, "out": 0}}

    events = analytics.get_events()
    assert len(events) == 1
    ev = events[0]
    assert ev.track_id == 1
    assert ev.line_name == "EntryGate"
    assert ev.direction == "IN"
    assert ev.frame_index == 1
    assert ev.class_name == "person"


# ==============================================================================
# TC-CNT-02: Nominal Single Crossing OUT (Right -> Left)
# ==============================================================================


@pytest.mark.unit
def test_tc_cnt_02_nominal_single_crossing_out() -> None:
    """TC-CNT-02: Verify track moving from RIGHT to LEFT of tripwire triggers OUT crossing."""
    line = CountingLine(name="ExitGate", start_point=(200, 400), end_point=(200, 100))
    analytics = SpatialAnalytics(lines=[line])

    # Frame 0: Track 1 starts at x=250 (RIGHT side)
    fr0 = _make_frame_result(0, [(1, "car", 250.0, 250.0)])
    summary0 = analytics.update(fr0)
    assert summary0.total_in == 0
    assert summary0.total_out == 0

    # Frame 1: Track 1 moves to x=150 (LEFT side), crossing x=200
    fr1 = _make_frame_result(1, [(1, "car", 150.0, 250.0)])
    summary1 = analytics.update(fr1)

    assert summary1.total_in == 0
    assert summary1.total_out == 1
    assert analytics.get_line_counts("ExitGate") == {"ExitGate": {"in": 0, "out": 1}}

    events = analytics.get_events()
    assert len(events) == 1
    ev = events[0]
    assert ev.track_id == 1
    assert ev.line_name == "ExitGate"
    assert ev.direction == "OUT"
    assert ev.frame_index == 1
    assert ev.class_name == "car"


# ==============================================================================
# TC-CNT-03: Vertical Line Crossings
# ==============================================================================


@pytest.mark.unit
def test_tc_cnt_03_vertical_line_crossings() -> None:
    """TC-CNT-03: Verify vertical line handles vertical vectors without division-by-zero."""
    # Perfectly vertical line: dx = 0
    vert_line = CountingLine(name="VertLine", start_point=(300, 500), end_point=(300, 100))
    analytics = SpatialAnalytics(lines=[vert_line])

    # In crossing: (200, 300) -> (400, 300)
    fr0 = _make_frame_result(0, [(1, "person", 200.0, 300.0)])
    analytics.update(fr0)
    fr1 = _make_frame_result(1, [(1, "person", 400.0, 300.0)])
    summary1 = analytics.update(fr1)

    assert summary1.total_in == 1
    assert summary1.total_out == 0

    # Out crossing with another ID: (400, 300) -> (200, 300)
    fr2 = _make_frame_result(2, [(2, "bicycle", 400.0, 300.0)])
    analytics.update(fr2)
    fr3 = _make_frame_result(3, [(2, "bicycle", 200.0, 300.0)])
    summary3 = analytics.update(fr3)

    assert summary3.total_in == 1
    assert summary3.total_out == 1


# ==============================================================================
# TC-CNT-04: Diagonal Line Crossings
# ==============================================================================


@pytest.mark.unit
def test_tc_cnt_04_diagonal_line_crossings() -> None:
    """TC-CNT-04: Verify crossing across a 45-degree diagonal line segment."""
    # Diagonal line from (100, 300) to (300, 100)
    diag_line = CountingLine(name="DiagLine", start_point=(100, 300), end_point=(300, 100))
    analytics = SpatialAnalytics(lines=[diag_line])

    # (150, 150) is on LEFT side; (250, 250) is on RIGHT side
    fr0 = _make_frame_result(0, [(1, "person", 150.0, 150.0)])
    analytics.update(fr0)

    # Frame 1: Crosses through midpoint (200, 200) to (250, 250)
    fr1 = _make_frame_result(1, [(1, "person", 250.0, 250.0)])
    summary = analytics.update(fr1)

    assert summary.total_in == 1
    assert summary.total_out == 0
    assert len(analytics.get_events()) == 1
    assert analytics.get_events()[0].direction == "IN"


# ==============================================================================
# TC-CNT-05: Crossing Infinite Line Extension (Near-Miss outside segment endpoints)
# ==============================================================================


@pytest.mark.unit
def test_tc_cnt_05_crossing_infinite_extension_near_miss() -> None:
    """TC-CNT-05: Movement crossing the infinite line extension outside endpoints triggers 0."""
    # Line segment defined strictly between y=100 and y=300 at x=200
    line = CountingLine(name="ShortSegment", start_point=(200, 300), end_point=(200, 100))
    analytics = SpatialAnalytics(lines=[line])

    # Object crosses x=200 at y=50 (outside [100, 300])
    fr0 = _make_frame_result(0, [(1, "person", 150.0, 50.0)])
    analytics.update(fr0)
    fr1 = _make_frame_result(1, [(1, "person", 250.0, 50.0)])
    summary = analytics.update(fr1)

    assert summary.total_in == 0
    assert summary.total_out == 0
    assert len(analytics.get_events()) == 0


# ==============================================================================
# TC-CNT-06: Stationary Object on Tripwire
# ==============================================================================


@pytest.mark.unit
def test_tc_cnt_06_stationary_object_on_tripwire() -> None:
    """TC-CNT-06: Stationary object resting near or on tripwire fires 0 triggers over 30 frames."""
    line = CountingLine(name="Gate", start_point=(200, 400), end_point=(200, 100))
    analytics = SpatialAnalytics(lines=[line])

    for frame_idx in range(30):
        # Resting at x=200.0, y=250.0
        fr = _make_frame_result(frame_idx, [(1, "person", 200.0, 250.0)])
        summary = analytics.update(fr)

    assert summary.total_in == 0
    assert summary.total_out == 0
    assert len(analytics.get_events()) == 0


# ==============================================================================
# TC-CNT-07: Jitter Rejection Across Line
# ==============================================================================


@pytest.mark.unit
def test_tc_cnt_07_jitter_rejection_across_line() -> None:
    """TC-CNT-07: Sub-pixel noise within debounce deadband suppresses duplicate triggers."""
    line = CountingLine(name="Gate", start_point=(200, 400), end_point=(200, 100))
    # Debounce deadband is 3.0 pixels
    config = SpatialAnalyticsConfig(debounce_distance=3.0)
    analytics = SpatialAnalytics(lines=[line], config=config)

    # Object jitters between x=199.0 and x=201.0 (both within 3.0px of x=200)
    for frame_idx in range(20):
        cx = 199.0 if frame_idx % 2 == 0 else 201.0
        fr = _make_frame_result(frame_idx, [(1, "person", cx, 250.0)])
        summary = analytics.update(fr)

    assert summary.total_in == 0
    assert summary.total_out == 0
    assert len(analytics.get_events()) == 0


# ==============================================================================
# TC-CNT-08: U-Turn Near Tripwire
# ==============================================================================


@pytest.mark.unit
def test_tc_cnt_08_uturn_near_tripwire() -> None:
    """TC-CNT-08: Object approaches tripwire, turns around without crossing, 0 triggers."""
    line = CountingLine(name="Gate", start_point=(200, 400), end_point=(200, 100))
    analytics = SpatialAnalytics(lines=[line])

    positions = [(100.0, 250.0), (150.0, 250.0), (195.0, 250.0), (140.0, 250.0), (90.0, 250.0)]
    for frame_idx, (cx, cy) in enumerate(positions):
        fr = _make_frame_result(frame_idx, [(1, "person", cx, cy)])
        summary = analytics.update(fr)

    assert summary.total_in == 0
    assert summary.total_out == 0
    assert len(analytics.get_events()) == 0


# ==============================================================================
# TC-CNT-09: Movement Parallel to Line
# ==============================================================================


@pytest.mark.unit
def test_tc_cnt_09_movement_parallel_to_line() -> None:
    """TC-CNT-09: Object moving parallel to tripwire maintains constant side, 0 triggers."""
    line = CountingLine(name="Gate", start_point=(200, 400), end_point=(200, 100))
    analytics = SpatialAnalytics(lines=[line])

    # Trajectory moves vertically along x=150 (parallel to x=200)
    for frame_idx in range(10):
        cy = 120.0 + frame_idx * 25.0
        fr = _make_frame_result(frame_idx, [(1, "car", 150.0, cy)])
        summary = analytics.update(fr)

    assert summary.total_in == 0
    assert summary.total_out == 0
    assert len(analytics.get_events()) == 0


# ==============================================================================
# TC-CNT-10: Consecutive Round-Trip (IN then OUT with cooldown)
# ==============================================================================


@pytest.mark.unit
def test_tc_cnt_10_consecutive_round_trip_with_cooldown() -> None:
    """TC-CNT-10: Verify IN followed by OUT respects cooldown frames window."""
    line = CountingLine(name="Gate", start_point=(200, 400), end_point=(200, 100))
    config = SpatialAnalyticsConfig(cooldown_frames=5)
    analytics = SpatialAnalytics(lines=[line], config=config)

    # Frame 0: Start at x=150 (LEFT)
    analytics.update(_make_frame_result(0, [(1, "person", 150.0, 250.0)]))

    # Frame 1: Cross to x=250 (RIGHT) -> IN crossing confirmed at frame 1
    summary1 = analytics.update(_make_frame_result(1, [(1, "person", 250.0, 250.0)]))
    assert summary1.total_in == 1
    assert summary1.total_out == 0

    # Frame 2: Immediate cross back to x=150 -> cooldown suppressed! (elapsed 1 < 5)
    summary2 = analytics.update(_make_frame_result(2, [(1, "person", 150.0, 250.0)]))
    assert summary2.total_in == 1
    assert summary2.total_out == 0

    # Object returns to right side and dwells until frame 6
    analytics.update(_make_frame_result(3, [(1, "person", 250.0, 250.0)]))
    analytics.update(_make_frame_result(4, [(1, "person", 250.0, 250.0)]))
    analytics.update(_make_frame_result(5, [(1, "person", 250.0, 250.0)]))
    analytics.update(_make_frame_result(6, [(1, "person", 250.0, 250.0)]))

    # Frame 7: Cross back to x=150 (LEFT) -> cooldown satisfied (7 - 1 = 6 >= 5)
    summary7 = analytics.update(_make_frame_result(7, [(1, "person", 150.0, 250.0)]))
    assert summary7.total_in == 1
    assert summary7.total_out == 1

    events = analytics.get_events()
    assert len(events) == 2
    assert events[0].direction == "IN"
    assert events[1].direction == "OUT"


# ==============================================================================
# TC-CNT-11: Multiple Concurrent Tracks
# ==============================================================================


@pytest.mark.unit
def test_tc_cnt_11_multiple_concurrent_tracks() -> None:
    """TC-CNT-11: Verify isolated state machines across different concurrent track IDs."""
    line = CountingLine(name="Gate", start_point=(200, 400), end_point=(200, 100))
    analytics = SpatialAnalytics(lines=[line])

    # Frame 0 setup:
    # Track 1: person at (150, 200) [LEFT]
    # Track 2: car at (250, 300) [RIGHT]
    # Track 3: bicycle at (120, 150) [LEFT]
    # Track 4: person at (200, 250) [ON LINE]
    fr0 = _make_frame_result(
        0,
        [
            (1, "person", 150.0, 200.0),
            (2, "car", 250.0, 300.0),
            (3, "bicycle", 120.0, 150.0),
            (4, "person", 200.0, 250.0),
        ],
    )
    analytics.update(fr0)

    # Frame 1:
    # Track 1: moves to (250, 200) -> Crosses IN
    # Track 2: moves to (150, 300) -> Crosses OUT
    # Track 3: moves to (120, 250) -> Moves parallel (no crossing)
    # Track 4: stays at (200, 250) -> Stationary (no crossing)
    fr1 = _make_frame_result(
        1,
        [
            (1, "person", 250.0, 200.0),
            (2, "car", 150.0, 300.0),
            (3, "bicycle", 120.0, 250.0),
            (4, "person", 200.0, 250.0),
        ],
    )
    summary = analytics.update(fr1)

    assert summary.total_in == 1
    assert summary.total_out == 1
    assert summary.class_counts == {"person": 2, "car": 1, "bicycle": 1}

    events = analytics.get_events()
    assert len(events) == 2
    ev_ids = {ev.track_id: ev.direction for ev in events}
    assert ev_ids[1] == "IN"
    assert ev_ids[2] == "OUT"


# ==============================================================================
# TC-CNT-12: Convex Zone Occupancy
# ==============================================================================


@pytest.mark.unit
def test_tc_cnt_12_convex_zone_occupancy() -> None:
    """TC-CNT-12: Verify inside point and vertex are counted; outside excluded."""
    zone = CountingZone(
        name="ParkingArea",
        polygon=[(100, 100), (300, 100), (300, 300), (100, 300)],
    )
    analytics = SpatialAnalytics(zones=[zone])

    # Track 1 at (200, 200): strictly inside
    # Track 2 at (100, 100): on vertex (cv2.pointPolygonTest >= 0)
    # Track 3 at (50, 50): strictly outside
    fr = _make_frame_result(
        0,
        [
            (1, "car", 200.0, 200.0),
            (2, "person", 100.0, 100.0),
            (3, "truck", 50.0, 50.0),
        ],
    )
    summary = analytics.update(fr)

    assert summary.zone_occupancy == {"ParkingArea": 2}
    occupants = analytics.get_zone_occupants("ParkingArea")
    assert set(occupants["ParkingArea"]) == {1, 2}


# ==============================================================================
# TC-CNT-13: Non-Convex (Concave L-Shaped) Zone
# ==============================================================================


@pytest.mark.unit
def test_tc_cnt_13_non_convex_l_shaped_zone() -> None:
    """TC-CNT-13: Concave L-shaped zone excludes notch point while including leg points."""
    # Concave polygon: inverted L-shape
    # Bounding box is [100, 300] x [100, 300], notch is (200..300, 200..300)
    zone = CountingZone(
        name="LZone",
        polygon=[(100, 100), (300, 100), (300, 200), (200, 200), (200, 300), (100, 300)],
    )
    analytics = SpatialAnalytics(zones=[zone])

    fr = _make_frame_result(
        0,
        [
            (1, "person", 150.0, 150.0),  # Inside top-left leg
            (2, "person", 250.0, 150.0),  # Inside top-right leg
            (3, "person", 250.0, 250.0),  # In the notch! Inside bbox but outside polygon
            (4, "car", 50.0, 50.0),  # Completely outside
        ],
    )
    summary = analytics.update(fr)

    assert summary.zone_occupancy == {"LZone": 2}
    occupants = analytics.get_zone_occupants("LZone")["LZone"]
    assert set(occupants) == {1, 2}
    assert 3 not in occupants


# ==============================================================================
# TC-CNT-14: Zone Dynamic Entry, Dwell, Exit
# ==============================================================================


@pytest.mark.unit
def test_tc_cnt_14_zone_entry_dwell_exit_lifecycle() -> None:
    """TC-CNT-14: Sequence: entry (0->1), dwell (1->1), exit (1->0)."""
    zone = CountingZone(
        name="Lobby",
        polygon=[(100, 100), (300, 100), (300, 300), (100, 300)],
    )
    analytics = SpatialAnalytics(zones=[zone])

    trajectory = [
        (50.0, 50.0),  # Frame 0: Outside
        (150.0, 150.0),  # Frame 1: Enters
        (200.0, 200.0),  # Frame 2: Dwells
        (250.0, 250.0),  # Frame 3: Dwells
        (350.0, 350.0),  # Frame 4: Exits
    ]

    occupancy_counts: list[int] = []
    for frame_idx, (cx, cy) in enumerate(trajectory):
        fr = _make_frame_result(frame_idx, [(1, "person", cx, cy)])
        s = analytics.update(fr)
        occupancy_counts.append(s.zone_occupancy["Lobby"])

    assert occupancy_counts == [0, 1, 1, 1, 0]


# ==============================================================================
# TC-CNT-15: Multiple Overlapping / Disjoint Zones
# ==============================================================================


@pytest.mark.unit
def test_tc_cnt_15_overlapping_and_disjoint_zones() -> None:
    """TC-CNT-15: Point in intersection counted in both; disjoint points counted exclusively."""
    zone_a = CountingZone(
        name="ZoneA",
        polygon=[(100, 100), (300, 100), (300, 300), (100, 300)],
    )
    zone_b = CountingZone(
        name="ZoneB",
        polygon=[(200, 200), (400, 200), (400, 400), (200, 400)],
    )
    analytics = SpatialAnalytics(zones=[zone_a, zone_b])

    # Track 1 at (250, 250): inside both ZoneA and ZoneB
    # Track 2 at (150, 150): inside ZoneA only
    # Track 3 at (350, 350): inside ZoneB only
    fr = _make_frame_result(
        0,
        [
            (1, "person", 250.0, 250.0),
            (2, "person", 150.0, 150.0),
            (3, "car", 350.0, 350.0),
        ],
    )
    summary = analytics.update(fr)

    assert summary.zone_occupancy == {"ZoneA": 2, "ZoneB": 2}
    all_occupants = analytics.get_zone_occupants()
    assert set(all_occupants["ZoneA"]) == {1, 2}
    assert set(all_occupants["ZoneB"]) == {1, 3}


# ==============================================================================
# TC-CNT-16: Real-Time Metrics & Class Breakdown
# ==============================================================================


@pytest.mark.unit
def test_tc_cnt_16_real_time_metrics_and_class_breakdown() -> None:
    """TC-CNT-16: AnalyticsSummary exposes valid total_in, total_out, zone_occupancy, and class_counts."""
    line = CountingLine(name="Line1", start_point=(200, 400), end_point=(200, 100))
    zone = CountingZone(name="Zone1", polygon=[(0, 0), (500, 0), (500, 500), (0, 500)])
    analytics = SpatialAnalytics(lines=[line], zones=[zone])

    fr0 = _make_frame_result(
        0,
        [
            (1, "person", 150.0, 250.0),
            (2, "person", 160.0, 260.0),
            (3, "car", 100.0, 100.0),
            (4, "bicycle", 120.0, 120.0),
        ],
    )
    analytics.update(fr0)

    # Frame 1: Track 1 crosses IN
    fr1 = _make_frame_result(
        1,
        [
            (1, "person", 250.0, 250.0),
            (2, "person", 160.0, 260.0),
            (3, "car", 100.0, 100.0),
            (4, "bicycle", 120.0, 120.0),
        ],
    )
    summary = analytics.update(fr1)

    assert isinstance(summary, AnalyticsSummary)
    assert summary.total_in == 1
    assert summary.total_out == 0
    assert summary.zone_occupancy == {"Zone1": 4}
    assert summary.class_counts == {"person": 2, "car": 1, "bicycle": 1}

    # Verify frozen immutability
    with pytest.raises((TypeError, ValueError)):
        summary.total_in = 99  # type: ignore[misc]


# ==============================================================================
# TC-CNT-17: Memory Lifecycle / Inactivity GC
# ==============================================================================


@pytest.mark.unit
def test_tc_cnt_17_memory_lifecycle_inactivity_gc() -> None:
    """TC-CNT-17: Inactive tracks pruned after exceeding max_disappeared_frames."""
    config = SpatialAnalyticsConfig(max_disappeared_frames=5)
    analytics = SpatialAnalytics(config=config)

    # Track 1 and Track 2 appear at frame 0
    analytics.update(_make_frame_result(0, [(1, "person", 100.0, 100.0), (2, "car", 200.0, 200.0)]))
    assert 1 in analytics.get_trajectories()
    assert 2 in analytics.get_trajectories()

    # Track 2 is seen again at frame 3; Track 1 disappears
    analytics.update(_make_frame_result(3, [(2, "car", 210.0, 210.0)]))

    # At frame 7: Track 1 disappeared for 7 frames (7 > 5) -> pruned!
    # Track 2 last seen at frame 3; 7 - 3 = 4 <= 5 -> retained!
    analytics.update(_make_frame_result(7, [(2, "car", 220.0, 220.0)]))

    trajectories = analytics.get_trajectories()
    assert 1 not in trajectories
    assert 2 in trajectories


# ==============================================================================
# TC-CNT-18: Headless Visualization Immutability
# ==============================================================================


@pytest.mark.unit
def test_tc_cnt_18_headless_visualization_immutability(
    sample_synthetic_frame: np.ndarray,
) -> None:
    """TC-CNT-18: annotate_analytics returns a new array copy; input frame is unmutated."""
    line = CountingLine(name="Line1", start_point=(100, 400), end_point=(100, 100))
    zone = CountingZone(name="Zone1", polygon=[(50, 50), (250, 50), (250, 250), (50, 250)])
    analytics = SpatialAnalytics(lines=[line], zones=[zone])

    # Run one update to populate HUD and trajectory state
    analytics.update(_make_frame_result(0, [(1, "person", 80.0, 200.0)]))
    analytics.update(_make_frame_result(1, [(1, "person", 120.0, 200.0)]))

    input_frame = sample_synthetic_frame.copy()
    original_copy = input_frame.copy()

    annotated = analytics.annotate_analytics(
        input_frame,
        draw_lines=True,
        draw_zones=True,
        draw_trajectories=True,
        draw_hud=True,
    )

    # Check distinct object and frame properties
    assert annotated is not input_frame
    assert annotated.shape == input_frame.shape
    assert annotated.dtype == np.uint8

    # Input frame must remain 100% unaltered
    np.testing.assert_array_equal(input_frame, original_copy)

    # Output annotated frame must contain drawn pixels (not identical to black frame)
    assert not np.array_equal(annotated, input_frame)


# ==============================================================================
# TC-CNT-19: Malformed Frame & Contract Validation
# ==============================================================================


@pytest.mark.unit
def test_tc_cnt_19_malformed_frame_and_contract_validation() -> None:
    """TC-CNT-19: Invalid frames and nonexistent queries raise ValueError / KeyError."""
    analytics = SpatialAnalytics()

    # Malformed frames
    with pytest.raises(ValueError, match="Frame must be a numpy.ndarray"):
        analytics.annotate_analytics("not_an_array")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Frame must be a 3D array"):
        analytics.annotate_analytics(np.zeros((480, 640), dtype=np.uint8))  # 2D

    with pytest.raises(ValueError, match="Frame must be a 3D array"):
        analytics.annotate_analytics(np.zeros((480, 640, 4), dtype=np.uint8))  # 4 channels

    with pytest.raises(ValueError, match="Frame dimensions must be non-zero"):
        analytics.annotate_analytics(np.zeros((0, 640, 3), dtype=np.uint8))  # 0 height

    with pytest.raises(ValueError, match="Frame dtype must be uint8"):
        analytics.annotate_analytics(np.zeros((480, 640, 3), dtype=np.float32))  # float32

    # Querying nonexistent lines/zones
    with pytest.raises(KeyError, match="Line 'Nonexistent' not configured"):
        analytics.get_line_counts("Nonexistent")

    with pytest.raises(KeyError, match="Zone 'Nonexistent' not configured"):
        analytics.get_zone_occupants("Nonexistent")


# ==============================================================================
# TC-CNT-20: Engine Reset
# ==============================================================================


@pytest.mark.unit
def test_tc_cnt_20_engine_reset() -> None:
    """TC-CNT-20: reset() restores all counters, occupant lists, events, and states to zero."""
    line = CountingLine(name="Gate", start_point=(200, 400), end_point=(200, 100))
    zone = CountingZone(name="Area", polygon=[(100, 100), (300, 100), (300, 300), (100, 300)])
    analytics = SpatialAnalytics(lines=[line], zones=[zone])

    # Populate state
    analytics.update(_make_frame_result(0, [(1, "person", 150.0, 250.0)]))
    analytics.update(_make_frame_result(1, [(1, "person", 250.0, 250.0)]))

    summary_before = analytics.get_summary()
    assert summary_before.total_in == 1
    assert len(analytics.get_events()) == 1
    assert len(analytics.get_trajectories()) == 1

    # Execute reset
    analytics.reset()

    summary_after = analytics.get_summary()
    assert summary_after.total_in == 0
    assert summary_after.total_out == 0
    assert summary_after.zone_occupancy == {"Area": 0}
    assert summary_after.class_counts == {}
    assert analytics.get_events() == []
    assert analytics.get_trajectories() == {}
    assert analytics.get_line_counts() == {"Gate": {"in": 0, "out": 0}}
    assert analytics.get_zone_occupants() == {"Area": []}
