"""Unit and integration tests for FoundryyyVision core and Baseball/American Football plugins."""

from __future__ import annotations

import math
import unittest

from foundryvision.cli import run_simulation
from foundryvision.core.calibration import PlanarFieldCalibrator
from foundryvision.core.identity import IdentityResolver
from foundryvision.core.ingest import FrameRingBuffer
from foundryvision.core.schema import (
    BoundingBox,
    Detection,
    ObjectCategory,
    Point2D,
    SportType,
)
from foundryvision.core.tracking import MultiObjectTracker
from foundryvision.sports.football import NFL_FIELD_LANDMARKS_YARDS


def test_planar_homography_calibration() -> None:
    calibrator = PlanarFieldCalibrator(NFL_FIELD_LANDMARKS_YARDS, unit="yards")
    observed = {
        "own_20_left_sideline": Point2D(300.0, 0.0),
        "own_20_right_sideline": Point2D(300.0, 533.3),
        "midfield_left_sideline": Point2D(600.0, 0.0),
        "midfield_right_sideline": Point2D(600.0, 533.3),
    }
    assert calibrator.update_from_keypoints(observed) is True
    projected = calibrator.project_point((450.0, 266.65))
    assert projected is not None
    assert math.isclose(projected.x, 45.0, abs_tol=0.1)
    assert math.isclose(projected.y, 26.665, abs_tol=0.1)


def test_multi_object_tracker_and_identity_resolution() -> None:
    tracker = MultiObjectTracker()
    resolver = IdentityResolver(roster={("offense", 15): "Patrick Mahomes"})

    for frame_idx in range(3):
        det = Detection(
            bbox=BoundingBox(100.0 + frame_idx * 10.0, 200.0, 120.0 + frame_idx * 10.0, 250.0),
            category=ObjectCategory.PLAYER,
            confidence=0.95,
            metadata={"team_id": "offense", "jersey_number": 15 if frame_idx != 1 else 16},
        )
        tracks = tracker.update([det], frame_index=frame_idx, fps=30.0)
        resolver.update(tracks)

    assert len(tracks) == 1
    trk = tracks[0]
    assert trk.track_id == 1
    assert trk.hits == 3
    assert trk.identity.jersey_number == 15
    assert trk.identity.athlete_id == "Patrick Mahomes"


def test_baseball_end_to_end_simulation() -> None:
    summary = run_simulation(SportType.BASEBALL, frames=15)
    assert summary["sport"] == "baseball"
    assert summary["calibrated"] is True
    event_types = [e["event_type"] for e in summary["events"]]
    assert "pitch_release" in event_types
    assert "bat_swing" in event_types
    assert "ball_in_play_contact" in event_types
    report = summary["scouting_report"]
    assert report["pitches_tracked"] == 1
    assert report["swings_tracked"] == 1
    assert report["contacts_tracked"] == 1
    assert report["avg_exit_velocity_mph"] > 0


def test_football_end_to_end_simulation() -> None:
    summary = run_simulation(SportType.FOOTBALL, frames=16)
    assert summary["sport"] == "football"
    assert summary["calibrated"] is True
    event_types = [e["event_type"] for e in summary["events"]]
    assert "ball_snap" in event_types
    assert "pass_attempt" in event_types
    assert "play_whistle" in event_types
    report = summary["scouting_report"]
    assert report["plays_completed"] == 1
    assert report["pass_rate"] == 1.0
    assert report["formation_tendencies"]["shotgun"] == 1


def test_frame_ring_buffer_slice() -> None:
    buf = FrameRingBuffer(maxlen=5)
    from foundryvision.core.schema import FramePacket

    for idx in range(7):
        buf.push(FramePacket(frame_index=idx, timestamp_sec=idx * 0.1, width=100, height=100))

    assert len(buf) == 5
    sliced = buf.slice_by_frame(3, 5)
    assert [p.frame_index for p in sliced] == [3, 4, 5]


class PipelineTestSuite(unittest.TestCase):
    def test_planar_homography_calibration(self) -> None:
        test_planar_homography_calibration()

    def test_multi_object_tracker_and_identity_resolution(self) -> None:
        test_multi_object_tracker_and_identity_resolution()

    def test_baseball_end_to_end_simulation(self) -> None:
        test_baseball_end_to_end_simulation()

    def test_football_end_to_end_simulation(self) -> None:
        test_football_end_to_end_simulation()

    def test_frame_ring_buffer_slice(self) -> None:
        test_frame_ring_buffer_slice()


if __name__ == "__main__":
    unittest.main()
