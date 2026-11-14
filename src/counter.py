"""Line-crossing detection and polygonal zone occupancy analytics engine.

Provides the SpatialAnalytics class for computing directional tripwire crossings,
anti-double-count debouncing, point-in-polygon zone occupancy, and headless visual HUDs.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from src.schemas import (
    AnalyticsSummary,
    CountingLine,
    CountingZone,
    FrameResult,
    LineCrossingEvent,
    SpatialAnalyticsConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class _TrackSpatialState:
    """Internal mutable state tracker for an individual tracked entity."""

    last_confirmed_side: dict[str, str] = field(default_factory=dict)
    last_crossing_frame: dict[str, int] = field(default_factory=dict)
    last_seen_frame: int = 0
    trajectory: deque[tuple[float, float]] = field(default_factory=deque)


def _cross_2d(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p: tuple[float, float],
) -> float:
    """Compute 2D cross product of vector (p1 - p0) and vector (p - p0).

    Returns:
        Positive if p is on the right side of directed segment p0 -> p1,
        negative if on the left side, zero if collinear.
    """
    return (p1[0] - p0[0]) * (p[1] - p0[1]) - (p1[1] - p0[1]) * (p[0] - p0[0])


def _segments_intersect(
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    p4: tuple[float, float],
) -> bool:
    """Determine whether line segment (p1, p2) and segment (p3, p4) intersect.

    Args:
        p1, p2: Endpoints of the first line segment (e.g. trajectory segment).
        p3, p4: Endpoints of the second line segment (e.g. tripwire line).

    Returns:
        True if the two line segments strictly intersect, False otherwise.
    """
    # Straddle test 1: points p1, p2 relative to line p3->p4
    d1 = _cross_2d(p3, p4, p1)
    d2 = _cross_2d(p3, p4, p2)

    # Straddle test 2: points p3, p4 relative to line p1->p2
    d3 = _cross_2d(p1, p2, p3)
    d4 = _cross_2d(p1, p2, p4)

    # Segments intersect if endpoints of each segment lie on opposite sides of the other segment
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and (
        (d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)
    ):
        return True

    # Check collinear / touching boundary cases
    if d1 == 0 and _on_segment(p3, p4, p1):
        return True
    if d2 == 0 and _on_segment(p3, p4, p2):
        return True
    if d3 == 0 and _on_segment(p1, p2, p3):
        return True
    return bool(d4 == 0 and _on_segment(p1, p2, p4))


def _on_segment(
    a: tuple[float, float],
    b: tuple[float, float],
    p: tuple[float, float],
) -> bool:
    """Check if point p lies on segment ab (assuming p is collinear with ab)."""
    return min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])


class SpatialAnalytics:
    """Spatial analytics engine computing line crossings and zone occupancies."""

    def __init__(
        self,
        lines: list[CountingLine] | None = None,
        zones: list[CountingZone] | None = None,
        config: SpatialAnalyticsConfig | None = None,
    ) -> None:
        """Initialize spatial analytics with lines, zones, and configuration.

        Args:
            lines: Collection of CountingLine tripwires to monitor.
            zones: Collection of CountingZone regions of interest to monitor.
            config: Threshold and buffer configuration settings.
        """
        self.lines: list[CountingLine] = list(lines) if lines else []
        self.zones: list[CountingZone] = list(zones) if zones else []
        self.config: SpatialAnalyticsConfig = (
            config if config is not None else SpatialAnalyticsConfig()
        )

        # Pre-compile OpenCV polygon contours for fast point-in-polygon queries
        self._zone_contours: dict[str, np.ndarray] = {
            zone.name: np.array(zone.polygon, dtype=np.int32).reshape((-1, 1, 2))
            for zone in self.zones
        }

        # Internal cumulative state
        self._total_in: int = 0
        self._total_out: int = 0
        self._line_in: dict[str, int] = {line.name: 0 for line in self.lines}
        self._line_out: dict[str, int] = {line.name: 0 for line in self.lines}
        self._zone_occupants: dict[str, list[int]] = {zone.name: [] for zone in self.zones}
        self._class_counts: dict[str, int] = {}
        self._events: list[LineCrossingEvent] = []

        # Track state storage: track_id -> _TrackSpatialState
        self._track_states: dict[int, _TrackSpatialState] = {}

    def _validate_frame(self, frame: Any) -> None:
        """Validate that input frame meets format, dimension, and dtype requirements."""
        if not isinstance(frame, np.ndarray):
            raise ValueError(f"Frame must be a numpy.ndarray; received {type(frame).__name__}")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"Frame must be a 3D array with shape (H, W, 3); received shape {frame.shape}"
            )
        if frame.shape[0] == 0 or frame.shape[1] == 0 or frame.size == 0:
            raise ValueError(f"Frame dimensions must be non-zero; received shape {frame.shape}")
        if frame.dtype != np.uint8:
            raise ValueError(f"Frame dtype must be uint8; received {frame.dtype}")

    def update(self, frame_result: FrameResult) -> AnalyticsSummary:
        """Update spatial tracking state with current frame detection results.

        Args:
            frame_result: Current frame inference results from ObjectTracker.

        Returns:
            Validated immutable AnalyticsSummary reflecting cumulative state.
        """
        frame_idx = frame_result.frame_index

        # 1. Update active class distribution
        current_class_counts: dict[str, int] = {}
        for obj in frame_result.objects:
            current_class_counts[obj.class_name] = current_class_counts.get(obj.class_name, 0) + 1
        self._class_counts = current_class_counts

        # 2. Process Point-in-Polygon Zone Occupancy
        for zone in self.zones:
            contour = self._zone_contours[zone.name]
            occupants: list[int] = []
            for obj in frame_result.objects:
                if obj.centroid is not None:
                    cx, cy = obj.centroid
                    # pointPolygonTest returns >= 0 for points inside or on the contour boundary
                    if cv2.pointPolygonTest(contour, (float(cx), float(cy)), False) >= 0:
                        occupants.append(obj.track_id)
            self._zone_occupants[zone.name] = occupants

        # 3. Process Directional Line Crossings
        for obj in frame_result.objects:
            if obj.centroid is None:
                continue

            curr_pos = obj.centroid
            track_id = obj.track_id

            if track_id not in self._track_states:
                self._track_states[track_id] = _TrackSpatialState(
                    trajectory=deque(maxlen=self.config.trajectory_history_length),
                    last_seen_frame=frame_idx,
                )

            track_state = self._track_states[track_id]
            track_state.last_seen_frame = frame_idx

            if len(track_state.trajectory) > 0:
                prev_pos = track_state.trajectory[-1]

                for line in self.lines:
                    self._check_line_crossing(
                        line=line,
                        track_id=track_id,
                        class_name=obj.class_name,
                        prev_pos=prev_pos,
                        curr_pos=curr_pos,
                        frame_idx=frame_idx,
                        track_state=track_state,
                    )
            else:
                # First time seeing this track: initialize side for each line
                for line in self.lines:
                    side = self._compute_confirmed_side(line, curr_pos)
                    if side != "NEUTRAL":
                        track_state.last_confirmed_side[line.name] = side

            track_state.trajectory.append(curr_pos)

        # 4. Garbage Collect Stale Inactive Tracks
        self._prune_disappeared_tracks(frame_idx)

        return self.get_summary()

    def _compute_confirmed_side(
        self,
        line: CountingLine,
        pt: tuple[float, float],
    ) -> str:
        """Compute confirmed side ('LEFT', 'RIGHT', or 'NEUTRAL') of point relative to line."""
        a = (float(line.start_point[0]), float(line.start_point[1]))
        b = (float(line.end_point[0]), float(line.end_point[1]))

        line_len = math.hypot(b[0] - a[0], b[1] - a[1])
        if line_len == 0.0:
            return "NEUTRAL"

        cp = _cross_2d(a, b, pt)
        signed_dist = cp / line_len

        if signed_dist > self.config.debounce_distance:
            return "RIGHT"
        elif signed_dist < -self.config.debounce_distance:
            return "LEFT"
        else:
            return "NEUTRAL"

    def _check_line_crossing(
        self,
        line: CountingLine,
        track_id: int,
        class_name: str,
        prev_pos: tuple[float, float],
        curr_pos: tuple[float, float],
        frame_idx: int,
        track_state: _TrackSpatialState,
    ) -> None:
        """Evaluate and record a line crossing for an individual track movement."""
        a = (float(line.start_point[0]), float(line.start_point[1]))
        b = (float(line.end_point[0]), float(line.end_point[1]))

        # Calculate current confirmed side
        curr_side = self._compute_confirmed_side(line, curr_pos)
        prev_confirmed = track_state.last_confirmed_side.get(line.name, "NEUTRAL")

        # Check cooldown constraint
        last_crossing = track_state.last_crossing_frame.get(line.name, -99999)
        cooldown_ok = (frame_idx - last_crossing) >= self.config.cooldown_frames

        # Check segment-segment intersection
        segments_cross = _segments_intersect(prev_pos, curr_pos, a, b)

        if segments_cross and cooldown_ok and curr_side != "NEUTRAL":
            # Valid crossing detected
            if prev_confirmed == "LEFT" and curr_side == "RIGHT":
                # Direction IN
                direction = "IN"
                self._total_in += 1
                self._line_in[line.name] += 1
                track_state.last_crossing_frame[line.name] = frame_idx
                track_state.last_confirmed_side[line.name] = "RIGHT"

                self._events.append(
                    LineCrossingEvent(
                        track_id=track_id,
                        line_name=line.name,
                        direction=direction,
                        frame_index=frame_idx,
                        class_name=class_name,
                    )
                )
            elif prev_confirmed == "RIGHT" and curr_side == "LEFT":
                # Direction OUT
                direction = "OUT"
                self._total_out += 1
                self._line_out[line.name] += 1
                track_state.last_crossing_frame[line.name] = frame_idx
                track_state.last_confirmed_side[line.name] = "LEFT"

                self._events.append(
                    LineCrossingEvent(
                        track_id=track_id,
                        line_name=line.name,
                        direction=direction,
                        frame_index=frame_idx,
                        class_name=class_name,
                    )
                )
        elif curr_side != "NEUTRAL":
            # Update confirmed side when clearly on one side without crossing
            track_state.last_confirmed_side[line.name] = curr_side

    def _prune_disappeared_tracks(self, current_frame_index: int) -> None:
        """Prune inactive track states exceeding max_disappeared_frames."""
        stale_ids = [
            tid
            for tid, state in self._track_states.items()
            if (current_frame_index - state.last_seen_frame) > self.config.max_disappeared_frames
        ]
        for tid in stale_ids:
            del self._track_states[tid]

    def get_summary(self) -> AnalyticsSummary:
        """Return current cumulative AnalyticsSummary."""
        zone_counts = {name: len(occupants) for name, occupants in self._zone_occupants.items()}
        return AnalyticsSummary(
            total_in=self._total_in,
            total_out=self._total_out,
            zone_occupancy=zone_counts,
            class_counts=dict(self._class_counts),
        )

    def get_line_counts(self, line_name: str | None = None) -> dict[str, dict[str, int]]:
        """Return breakdown of in/out counts per line."""
        if line_name:
            if line_name not in self._line_in:
                raise KeyError(f"Line '{line_name}' not configured")
            return {
                line_name: {
                    "in": self._line_in[line_name],
                    "out": self._line_out[line_name],
                }
            }
        return {
            line.name: {
                "in": self._line_in[line.name],
                "out": self._line_out[line.name],
            }
            for line in self.lines
        }

    def get_zone_occupants(self, zone_name: str | None = None) -> dict[str, list[int]]:
        """Return list of track IDs currently inside specified zone(s)."""
        if zone_name:
            if zone_name not in self._zone_occupants:
                raise KeyError(f"Zone '{zone_name}' not configured")
            return {zone_name: list(self._zone_occupants[zone_name])}
        return {name: list(occupants) for name, occupants in self._zone_occupants.items()}

    def get_events(self) -> list[LineCrossingEvent]:
        """Return copy of historical crossing events log."""
        return list(self._events)

    def get_trajectories(self) -> dict[int, list[tuple[float, float]]]:
        """Return active centroid trajectory paths."""
        return {tid: list(state.trajectory) for tid, state in self._track_states.items()}

    def annotate_analytics(
        self,
        frame: np.ndarray,
        summary: AnalyticsSummary | None = None,
        draw_lines: bool = True,
        draw_zones: bool = True,
        draw_trajectories: bool = True,
        draw_hud: bool = True,
    ) -> np.ndarray:
        """Render lines, zones, HUD overlay, and trajectories onto a frame copy.

        Args:
            frame: Input BGR image frame (H, W, 3).
            summary: Optional AnalyticsSummary to display. Defaults to current state.
            draw_lines: Whether to draw tripwire counting lines.
            draw_zones: Whether to draw polygonal counting zones.
            draw_trajectories: Whether to draw centroid trajectory trails.
            draw_hud: Whether to render the summary metrics HUD overlay card.

        Returns:
            New annotated copy of the frame as np.ndarray.
        """
        self._validate_frame(frame)
        annotated = frame.copy()
        current_summary = summary if summary is not None else self.get_summary()

        # 1. Render Polygonal Zones with Alpha Blending
        if draw_zones and self.zones:
            overlay = annotated.copy()
            for zone in self.zones:
                pts = np.array(zone.polygon, dtype=np.int32).reshape((-1, 1, 2))
                count = len(self._zone_occupants.get(zone.name, []))

                # Fill semi-transparent polygon (Purple/Indigo hue)
                cv2.fillPoly(overlay, [pts], (180, 50, 100))
                # Draw solid boundary
                cv2.polylines(annotated, [pts], isClosed=True, color=(220, 100, 160), thickness=2)

                # Label at first vertex
                v0 = zone.polygon[0]
                label = f"{zone.name}: {count}"
                cv2.putText(
                    annotated,
                    label,
                    (v0[0] + 5, v0[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            # Blend overlay at 30% alpha
            cv2.addWeighted(overlay, 0.3, annotated, 0.7, 0, annotated)

        # 2. Render Counting Tripwires
        if draw_lines and self.lines:
            for line in self.lines:
                pt1 = (int(line.start_point[0]), int(line.start_point[1]))
                pt2 = (int(line.end_point[0]), int(line.end_point[1]))

                in_cnt = self._line_in.get(line.name, 0)
                out_cnt = self._line_out.get(line.name, 0)

                # Draw main tripwire line (Cyan)
                cv2.line(annotated, pt1, pt2, (255, 255, 0), 2, cv2.LINE_AA)

                # Draw endpoint markers
                cv2.circle(annotated, pt1, 4, (0, 255, 255), -1)
                cv2.circle(annotated, pt2, 4, (0, 255, 255), -1)

                # Midpoint for label badge
                mx = (pt1[0] + pt2[0]) // 2
                my = (pt1[1] + pt2[1]) // 2

                badge_text = f"{line.name} | IN:{in_cnt} OUT:{out_cnt}"
                cv2.putText(
                    annotated,
                    badge_text,
                    (mx + 5, my - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        # 3. Render Trajectory Trails
        if draw_trajectories and self._track_states:
            for state in self._track_states.values():
                if len(state.trajectory) >= 2:
                    pts = np.array(list(state.trajectory), dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(
                        annotated,
                        [pts],
                        isClosed=False,
                        color=(0, 200, 255),
                        thickness=1,
                        lineType=cv2.LINE_AA,
                    )

        # 4. Render Analytics Summary HUD Card
        if draw_hud:
            h, w = annotated.shape[:2]
            hud_w, hud_h = 240, 95
            hud_x1, hud_y1 = 10, 10
            hud_x2, hud_y2 = hud_x1 + hud_w, hud_y1 + hud_h

            # Semi-transparent dark background
            hud_overlay = annotated.copy()
            cv2.rectangle(hud_overlay, (hud_x1, hud_y1), (hud_x2, hud_y2), (20, 20, 20), -1)
            cv2.addWeighted(hud_overlay, 0.7, annotated, 0.3, 0, annotated)

            # Border
            cv2.rectangle(annotated, (hud_x1, hud_y1), (hud_x2, hud_y2), (80, 80, 80), 1)

            # Text content
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                annotated,
                "SPATIAL ANALYTICS",
                (hud_x1 + 8, hud_y1 + 18),
                font,
                0.45,
                (0, 255, 200),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                f"Total IN:  {current_summary.total_in}",
                (hud_x1 + 8, hud_y1 + 38),
                font,
                0.45,
                (100, 255, 100),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                f"Total OUT: {current_summary.total_out}",
                (hud_x1 + 8, hud_y1 + 56),
                font,
                0.45,
                (100, 150, 255),
                1,
                cv2.LINE_AA,
            )

            active_zones_str = (
                ", ".join(f"{k}:{v}" for k, v in current_summary.zone_occupancy.items()) or "None"
            )
            cv2.putText(
                annotated,
                f"Zones: {active_zones_str}",
                (hud_x1 + 8, hud_y1 + 74),
                font,
                0.38,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

            classes_str = (
                ", ".join(f"{k}:{v}" for k, v in current_summary.class_counts.items()) or "None"
            )
            cv2.putText(
                annotated,
                f"Objects: {classes_str}",
                (hud_x1 + 8, hud_y1 + 88),
                font,
                0.35,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )

        return annotated

    def reset(self) -> None:
        """Reset all cumulative counts, debounce states, and histories."""
        self._total_in = 0
        self._total_out = 0
        self._line_in = {line.name: 0 for line in self.lines}
        self._line_out = {line.name: 0 for line in self.lines}
        self._zone_occupants = {zone.name: [] for zone in self.zones}
        self._class_counts = {}
        self._events.clear()
        self._track_states.clear()
