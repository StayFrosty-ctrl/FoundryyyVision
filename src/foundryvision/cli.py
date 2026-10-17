"""Command-line interface and synthetic play simulators for Baseball and American Football."""

from __future__ import annotations

import argparse
import json

from foundryvision.api import build_pipeline_summary
from foundryvision.core.identity import IdentityResolver
from foundryvision.core.pipeline import VisionPipeline
from foundryvision.core.schema import (
    BoundingBox,
    Detection,
    FramePacket,
    ObjectCategory,
    Point2D,
    SportType,
)
from foundryvision.sports.baseball import BaseballPlugin
from foundryvision.sports.football import FootballPlugin


def simulate_baseball_sequence(num_frames: int = 15) -> list[FramePacket]:
    """Generate a synthetic pitch -> swing -> bat-ball contact frame sequence."""
    packets: list[FramePacket] = []
    landmarks = {
        "home_plate": Point2D(500.0, 800.0),
        "first_base": Point2D(800.0, 500.0),
        "second_base": Point2D(500.0, 200.0),
        "third_base": Point2D(200.0, 500.0),
    }
    for idx in range(num_frames):
        ball_y = 515.0 + idx * 28.0 if idx <= 8 else 739.0 - (idx - 8) * 36.0
        bat_x = 460.0 + (idx - 6) * 24.0 if 6 <= idx <= 9 else 460.0
        dets = [
            Detection(
                bbox=BoundingBox(485.0, 480.0, 515.0, 540.0),
                category=ObjectCategory.PLAYER,
                confidence=0.96,
                metadata={"team_id": "MIL", "jersey_number": 39, "role": "pitcher"},
            ),
            Detection(
                bbox=BoundingBox(470.0, 750.0, 500.0, 810.0),
                category=ObjectCategory.PLAYER,
                confidence=0.95,
                metadata={"team_id": "CHC", "jersey_number": 27, "role": "batter"},
            ),
            Detection(
                bbox=BoundingBox(495.0, ball_y - 5.0, 505.0, ball_y + 5.0),
                category=ObjectCategory.BALL,
                confidence=0.91,
            ),
            Detection(
                bbox=BoundingBox(bat_x - 15.0, 765.0, bat_x + 15.0, 785.0),
                category=ObjectCategory.BAT,
                confidence=0.89,
            ),
        ]
        packets.append(
            FramePacket(
                frame_index=idx,
                timestamp_sec=round(idx / 60.0, 4),
                width=1920,
                height=1080,
                fps=60.0,
                field_keypoints=landmarks,
                detections=dets,
            )
        )
    return packets


def simulate_football_sequence(num_frames: int = 16) -> list[FramePacket]:
    """Generate a synthetic pre-snap -> snap -> pass attempt -> whistle frame sequence."""
    packets: list[FramePacket] = []
    landmarks = {
        "own_20_left_sideline": Point2D(300.0, 0.0),
        "own_20_right_sideline": Point2D(300.0, 533.3),
        "midfield_left_sideline": Point2D(600.0, 0.0),
        "midfield_right_sideline": Point2D(600.0, 533.3),
    }
    for idx in range(num_frames):
        if idx < 3:
            dx = 0.0
        elif idx < 11:
            dx = (idx - 2) * 14.0
        else:
            dx = 8 * 14.0

        qb_x = 450.0 - (idx - 2) * 4.0 if 3 <= idx < 7 else 434.0
        ball_x = qb_x if idx < 6 else 434.0 + (idx - 5) * 32.0
        dets = [
            Detection(
                bbox=BoundingBox(qb_x - 10.0, 250.0, qb_x + 10.0, 280.0),
                category=ObjectCategory.PLAYER,
                confidence=0.97,
                metadata={"team_id": "offense", "jersey_number": 15, "role": "QB"},
            ),
            Detection(
                bbox=BoundingBox(500.0 + dx - 10.0, 90.0, 500.0 + dx + 10.0, 120.0),
                category=ObjectCategory.PLAYER,
                confidence=0.96,
                metadata={"team_id": "offense", "jersey_number": 11, "role": "WR"},
            ),
            Detection(
                bbox=BoundingBox(525.0 + dx - 10.0, 95.0, 525.0 + dx + 10.0, 125.0),
                category=ObjectCategory.PLAYER,
                confidence=0.94,
                metadata={"team_id": "defense", "jersey_number": 24, "role": "CB"},
            ),
            Detection(
                bbox=BoundingBox(ball_x - 4.0, 260.0, ball_x + 4.0, 268.0),
                category=ObjectCategory.BALL,
                confidence=0.90,
            ),
        ]
        packets.append(
            FramePacket(
                frame_index=idx,
                timestamp_sec=round(idx / 30.0, 4),
                width=1920,
                height=1080,
                fps=30.0,
                field_keypoints=landmarks,
                detections=dets,
            )
        )
    return packets


def run_simulation(sport: SportType, frames: int = 16) -> dict:
    if sport == SportType.BASEBALL:
        plugin = BaseballPlugin()
        roster = {("MIL", 39): "Corbin Burnes", ("CHC", 27): "Seiya Suzuki"}
        sequence = simulate_baseball_sequence(frames)
    else:
        plugin = FootballPlugin()
        roster = {
            ("offense", 15): "Patrick Mahomes",
            ("offense", 11): "WR1",
            ("defense", 24): "Derek Stingley",
        }
        sequence = simulate_football_sequence(frames)

    pipeline = VisionPipeline(
        sport_plugin=plugin,
        identity_resolver=IdentityResolver(roster=roster),
    )
    for pkt in sequence:
        pipeline.step(pkt)

    return build_pipeline_summary(pipeline)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="foundryvision",
        description="FoundryyyVision Real-Time Sports Tracking & Scouting Engine",
    )
    parser.add_argument(
        "--sport",
        choices=[SportType.BASEBALL.value, SportType.FOOTBALL.value],
        default=SportType.FOOTBALL.value,
        help="Target sport specialization plugin",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=16,
        help="Number of frames to simulate",
    )
    args = parser.parse_args(argv)
    summary = run_simulation(SportType(args.sport), frames=args.frames)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
