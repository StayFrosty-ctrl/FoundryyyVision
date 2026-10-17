"""American Football specialization module for All-22/Sideline tracking, formation & play recognition, and scouting."""

from __future__ import annotations

from collections import Counter
import math
from typing import Any

from foundryvision.core.schema import (
    FramePacket,
    ObjectCategory,
    SportEvent,
    SportType,
    Tracklet,
)
from foundryvision.sports.base import SportPlugin


# Standardized NFL field coordinates in yards: X in [0, 120] (endzone to endzone), Y in [0, 53.33] (sideline to sideline)
NFL_FIELD_LANDMARKS_YARDS: dict[str, tuple[float, float]] = {
    "own_20_left_sideline": (30.0, 0.0),
    "own_20_right_sideline": (30.0, 53.33),
    "midfield_left_sideline": (60.0, 0.0),
    "midfield_right_sideline": (60.0, 53.33),
    "opp_20_left_sideline": (90.0, 0.0),
    "opp_20_right_sideline": (90.0, 53.33),
}


class FootballPlugin(SportPlugin):
    """Real-time American Football play segmentation, formation analysis, route/play classification, and scouting."""

    sport = SportType.FOOTBALL

    def __init__(
        self,
        snap_motion_threshold_px: float = 8.5,
        whistle_min_frames: int = 4,
    ) -> None:
        self.snap_motion_threshold_px = snap_motion_threshold_px
        self.whistle_min_frames = whistle_min_frames

        self._in_play: bool = False
        self._low_motion_streak: int = 0
        self._line_of_scrimmage_x: float | None = None
        self._current_formation: str = "unknown"
        self._current_play_type: str | None = None
        self._plays_completed: int = 0
        self._formation_counts: Counter[str] = Counter()
        self._play_type_counts: Counter[str] = Counter()

    def reference_landmarks(self) -> dict[str, tuple[float, float]]:
        return dict(NFL_FIELD_LANDMARKS_YARDS)

    def coordinate_unit(self) -> str:
        return "yards"

    def process_frame(self, packet: FramePacket) -> list[SportEvent]:
        events: list[SportEvent] = []
        players = [t for t in packet.tracks if t.category == ObjectCategory.PLAYER]
        balls = [t for t in packet.tracks if t.category == ObjectCategory.BALL]

        avg_player_speed_px = (
            sum(math.hypot(*p.velocity_px) for p in players) / len(players)
            if players
            else 0.0
        )

        # 1. Pre-snap analysis: Line of Scrimmage (LOS) and Offensive Formation
        if not self._in_play and players:
            los_x = self._estimate_line_of_scrimmage(players)
            if los_x is not None:
                self._line_of_scrimmage_x = los_x
            self._current_formation = self._classify_offensive_formation(
                players, self._line_of_scrimmage_x
            )

        # 2. Snap detection (transition from pre-snap set to post-snap burst)
        if not self._in_play and len(players) >= 2 and avg_player_speed_px >= self.snap_motion_threshold_px:
            self._in_play = True
            self._low_motion_streak = 0
            self._current_play_type = None
            self._formation_counts[self._current_formation] += 1
            events.append(
                SportEvent(
                    event_type="ball_snap",
                    sport=self.sport,
                    frame_index=packet.frame_index,
                    timestamp_sec=packet.timestamp_sec,
                    confidence=0.91,
                    involved_track_ids=[p.track_id for p in players],
                    details={
                        "formation": self._current_formation,
                        "line_of_scrimmage_yd": self._line_of_scrimmage_x,
                    },
                )
            )

        # 3. Post-snap play type classification (Pass vs. Run)
        if self._in_play and self._current_play_type is None:
            ball_trk = balls[0] if balls else None
            if ball_trk is not None and (
                ball_trk.speed_mph >= 22.0 or math.hypot(*ball_trk.velocity_px) >= 18.0
            ):
                self._current_play_type = "pass"
                self._play_type_counts["pass"] += 1
                events.append(
                    SportEvent(
                        event_type="pass_attempt",
                        sport=self.sport,
                        frame_index=packet.frame_index,
                        timestamp_sec=packet.timestamp_sec,
                        confidence=0.89,
                        involved_track_ids=[ball_trk.track_id],
                        details={"formation": self._current_formation},
                    )
                )
            elif self._in_play and self._CheckDownfieldRun(players):
                self._current_play_type = "run"
                self._play_type_counts["run"] += 1
                events.append(
                    SportEvent(
                        event_type="rush_attempt",
                        sport=self.sport,
                        frame_index=packet.frame_index,
                        timestamp_sec=packet.timestamp_sec,
                        confidence=0.84,
                        involved_track_ids=[p.track_id for p in players],
                        details={"formation": self._current_formation},
                    )
                )

        # 4. Play whistle / tackle termination
        if self._in_play:
            if avg_player_speed_px < self.snap_motion_threshold_px * 0.45:
                self._low_motion_streak += 1
            else:
                self._low_motion_streak = 0

            if self._low_motion_streak >= self.whistle_min_frames:
                play_kind = self._current_play_type or "run"
                if self._current_play_type is None:
                    self._play_type_counts[play_kind] += 1
                self._plays_completed += 1
                events.append(
                    SportEvent(
                        event_type="play_whistle",
                        sport=self.sport,
                        frame_index=packet.frame_index,
                        timestamp_sec=packet.timestamp_sec,
                        confidence=0.87,
                        involved_track_ids=[p.track_id for p in players],
                        details={
                            "play_type": play_kind,
                            "formation": self._current_formation,
                        },
                    )
                )
                self._in_play = False
                self._low_motion_streak = 0

        packet.sport_analytics["football"] = {
            "in_play": self._in_play,
            "formation": self._current_formation,
            "line_of_scrimmage_yd": self._line_of_scrimmage_x,
            "active_play_type": self._current_play_type,
            "plays_completed": self._plays_completed,
        }
        return events

    def _estimate_line_of_scrimmage(self, players: list[Tracklet]) -> float | None:
        offense_x = [
            p.field_position.x
            for p in players
            if p.identity.team_id == "offense" and p.field_position is not None
        ]
        if offense_x:
            return round(max(offense_x), 2)
        all_x = [p.field_position.x for p in players if p.field_position is not None]
        if all_x:
            return round(sum(all_x) / len(all_x), 2)
        return None

    def _classify_offensive_formation(
        self, players: list[Tracklet], los_x: float | None
    ) -> str:
        """Heuristic formation classifier (Shotgun / Empty / Under Center) from pre-snap spacing."""
        offense = [p for p in players if p.identity.team_id in ("offense", None)]
        if not offense:
            return "unknown"

        wide_players = [
            p
            for p in offense
            if p.field_position is not None
            and (p.field_position.y <= 14.0 or p.field_position.y >= 39.3)
        ]
        deep_backfield = [
            p
            for p in offense
            if p.field_position is not None
            and los_x is not None
            and (los_x - p.field_position.x) >= 4.0
        ]
        if len(wide_players) >= 4 and len(deep_backfield) <= 1:
            return "empty_spread"
        if deep_backfield:
            return "shotgun"
        return "under_center"

    def _CheckDownfieldRun(self, players: list[Tracklet]) -> bool:
        if self._line_of_scrimmage_x is None:
            return False
        for p in players:
            if (
                p.identity.team_id == "offense"
                and p.field_position is not None
                and (p.field_position.x - self._line_of_scrimmage_x) >= 3.5
            ):
                return True
        return False

    def generate_scouting_report(self) -> dict[str, Any]:
        total_plays = sum(self._play_type_counts.values())
        pass_rate = (
            round(self._play_type_counts["pass"] / total_plays, 3)
            if total_plays > 0
            else 0.0
        )
        run_rate = (
            round(self._play_type_counts["run"] / total_plays, 3)
            if total_plays > 0
            else 0.0
        )
        return {
            "sport": self.sport.value,
            "plays_completed": self._plays_completed,
            "formation_tendencies": dict(self._formation_counts),
            "play_type_counts": dict(self._play_type_counts),
            "pass_rate": pass_rate,
            "run_rate": run_rate,
        }
