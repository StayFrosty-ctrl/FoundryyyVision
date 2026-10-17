"""Multi-object tracking with IoU + center-distance association and velocity estimation."""

from __future__ import annotations

import math

from foundryvision.core.calibration import PlanarFieldCalibrator
from foundryvision.core.schema import BoundingBox, Detection, ObjectCategory, Tracklet


class MultiObjectTracker:
    """ByteTrack/OC-SORT inspired real-time tracker for athletes and high-speed balls."""

    def __init__(
        self,
        iou_threshold: float = 0.2,
        max_center_distance_px: float = 140.0,
        max_age: int = 15,
        history_len: int = 60,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_center_distance_px = max_center_distance_px
        self.max_age = max_age
        self.history_len = history_len
        self._next_id = 1
        self._tracks: dict[int, Tracklet] = {}

    def update(
        self,
        detections: list[Detection],
        frame_index: int,
        fps: float = 30.0,
        calibrator: PlanarFieldCalibrator | None = None,
    ) -> list[Tracklet]:
        for trk in self._tracks.values():
            trk.age += 1
            trk.time_since_update += 1

        unmatched_dets = list(range(len(detections)))
        unmatched_trks = list(self._tracks.keys())
        matches: list[tuple[int, int]] = []

        # Greedy bipartite matching by category + combined IoU & center distance score
        candidate_pairs: list[tuple[float, int, int]] = []
        for tid in unmatched_trks:
            trk = self._tracks[tid]
            for did in unmatched_dets:
                det = detections[did]
                if det.category != trk.category:
                    continue
                score = self._association_score(trk.bbox, det.bbox, trk.category)
                if score > 0.0:
                    candidate_pairs.append((score, tid, did))

        candidate_pairs.sort(key=lambda item: item[0], reverse=True)
        used_trks: set[int] = set()
        used_dets: set[int] = set()
        for _, tid, did in candidate_pairs:
            if tid in used_trks or did in used_dets:
                continue
            matches.append((tid, did))
            used_trks.add(tid)
            used_dets.add(did)

        for tid, did in matches:
            self._update_track(self._tracks[tid], detections[did], frame_index, fps, calibrator)

        for did in unmatched_dets:
            if did not in used_dets:
                self._spawn_track(detections[did], frame_index, calibrator)

        expired = [
            tid for tid, trk in self._tracks.items() if trk.time_since_update > self.max_age
        ]
        for tid in expired:
            del self._tracks[tid]

        return [
            trk for trk in self._tracks.values() if trk.time_since_update == 0
        ]

    def _association_score(
        self, prev_box: BoundingBox, new_box: BoundingBox, category: ObjectCategory
    ) -> float:
        iou = prev_box.iou(new_box)
        cx1, cy1 = prev_box.center
        cx2, cy2 = new_box.center
        dist = math.hypot(cx2 - cx1, cy2 - cy1)
        max_dist = (
            self.max_center_distance_px * 2.2
            if category in (ObjectCategory.BALL, ObjectCategory.BAT)
            else self.max_center_distance_px
        )
        if iou < self.iou_threshold and dist > max_dist:
            return 0.0
        dist_score = max(0.0, 1.0 - dist / max_dist)
        return 0.65 * iou + 0.35 * dist_score

    def _update_track(
        self,
        trk: Tracklet,
        det: Detection,
        frame_index: int,
        fps: float,
        calibrator: PlanarFieldCalibrator | None,
    ) -> None:
        prev_cx, prev_cy = trk.bbox.center
        new_cx, new_cy = det.bbox.center
        dt_frames = max(1, trk.time_since_update)
        trk.velocity_px = ((new_cx - prev_cx) / dt_frames, (new_cy - prev_cy) / dt_frames)
        trk.bbox = det.bbox
        trk.confidence = det.confidence
        trk.hits += 1
        trk.time_since_update = 0
        if det.keypoints:
            trk.keypoints = det.keypoints
        if det.metadata.get("team_id"):
            trk.identity.team_id = str(det.metadata["team_id"])
        if det.metadata.get("jersey_number") is not None:
            trk.identity.jersey_number = int(det.metadata["jersey_number"])
        if det.metadata.get("role"):
            trk.identity.role = str(det.metadata["role"])

        trk.history.append((frame_index, det.bbox))
        if len(trk.history) > self.history_len:
            trk.history.pop(0)

        if calibrator and calibrator.is_calibrated:
            anchor = (
                det.bbox.center
                if det.category == ObjectCategory.BALL
                else det.bbox.bottom_center
            )
            field_coord = calibrator.project_point(anchor)
            if field_coord is not None:
                if trk.field_history:
                    prev_frame, prev_coord = trk.field_history[-1]
                    dt_sec = max(1e-3, (frame_index - prev_frame) / max(1.0, fps))
                    dist_units = math.hypot(
                        field_coord.x - prev_coord.x, field_coord.y - prev_coord.y
                    )
                    feet_per_sec = (
                        dist_units * 3.0 / dt_sec
                        if field_coord.unit == "yards"
                        else dist_units / dt_sec
                    )
                    trk.speed_mph = round(feet_per_sec * 0.681818, 2)
                trk.field_position = field_coord
                trk.field_history.append((frame_index, field_coord))
                if len(trk.field_history) > self.history_len:
                    trk.field_history.pop(0)

    def _spawn_track(
        self,
        det: Detection,
        frame_index: int,
        calibrator: PlanarFieldCalibrator | None,
    ) -> Tracklet:
        tid = self._next_id
        self._next_id += 1
        field_pos = None
        field_hist = []
        if calibrator and calibrator.is_calibrated:
            anchor = (
                det.bbox.center
                if det.category == ObjectCategory.BALL
                else det.bbox.bottom_center
            )
            field_pos = calibrator.project_point(anchor)
            if field_pos is not None:
                field_hist.append((frame_index, field_pos))

        trk = Tracklet(
            track_id=tid,
            category=det.category,
            bbox=det.bbox,
            confidence=det.confidence,
            keypoints=dict(det.keypoints),
            field_position=field_pos,
            history=[(frame_index, det.bbox)],
            field_history=field_hist,
        )
        if det.metadata.get("team_id"):
            trk.identity.team_id = str(det.metadata["team_id"])
        if det.metadata.get("jersey_number") is not None:
            trk.identity.jersey_number = int(det.metadata["jersey_number"])
        if det.metadata.get("role"):
            trk.identity.role = str(det.metadata["role"])
        self._tracks[tid] = trk
        return trk
