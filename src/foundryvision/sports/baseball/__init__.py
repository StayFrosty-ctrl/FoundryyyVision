"""Baseball specialization module inspired by BaseballCV and Statcast-style tracking."""

from __future__ import annotations

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


# Standard MLB 90-foot diamond landmarks with Home Plate at (0, 0)
MLB_DIAMOND_LANDMARKS_FEET: dict[str, tuple[float, float]] = {
    "home_plate": (0.0, 0.0),
    "first_base": (63.64, 63.64),
    "second_base": (0.0, 127.28),
    "third_base": (-63.64, 63.64),
    "pitchers_plate": (0.0, 60.5),
}


class BaseballPlugin(SportPlugin):
    """Real-time baseball tracking, pitch/swing/contact detection, and player scouting."""

    sport = SportType.BASEBALL

    def __init__(
        self,
        pitch_speed_threshold_mph: float = 55.0,
        swing_velocity_threshold_px: float = 18.0,
        contact_distance_px: float = 45.0,
    ) -> None:
        self.pitch_speed_threshold_mph = pitch_speed_threshold_mph
        self.swing_velocity_threshold_px = swing_velocity_threshold_px
        self.contact_distance_px = contact_distance_px

        self._pitch_active: bool = False
        self._swing_detected_in_pitch: bool = False
        self._contact_detected_in_pitch: bool = False
        self._pitch_velocities_mph: list[float] = []
        self._exit_velocities_mph: list[float] = []
        self._swings_count: int = 0
        self._contacts_count: int = 0
        self._pitches_count: int = 0

    def reference_landmarks(self) -> dict[str, tuple[float, float]]:
        return dict(MLB_DIAMOND_LANDMARKS_FEET)

    def coordinate_unit(self) -> str:
        return "feet"

    def process_frame(self, packet: FramePacket) -> list[SportEvent]:
        events: list[SportEvent] = []
        balls = [t for t in packet.tracks if t.category == ObjectCategory.BALL]
        bats = [t for t in packet.tracks if t.category == ObjectCategory.BAT]
        gloves = [t for t in packet.tracks if t.category == ObjectCategory.GLOVE]
        players = [t for t in packet.tracks if t.category == ObjectCategory.PLAYER]

        roles = self._assign_player_roles(players)
        ball_trk = balls[0] if balls else None
        bat_trk = bats[0] if bats else None
        glove_trk = gloves[0] if gloves else None

        # 1. Pitch release & velocity tracking (ball moving toward home plate: vy > 0)
        if ball_trk is not None:
            vy = ball_trk.velocity_px[1]
            if (
                not self._pitch_active
                and not self._contact_detected_in_pitch
                and vy >= 12.0
            ):
                self._pitch_active = True
                self._swing_detected_in_pitch = False
                self._pitches_count += 1
                pitch_mph = round(abs(vy) * 3.4, 1)
                self._pitch_velocities_mph.append(pitch_mph)
                events.append(
                    SportEvent(
                        event_type="pitch_release",
                        sport=self.sport,
                        frame_index=packet.frame_index,
                        timestamp_sec=packet.timestamp_sec,
                        confidence=0.9,
                        involved_track_ids=[ball_trk.track_id],
                        details={"estimated_velocity_mph": pitch_mph},
                    )
                )

        # 2. Bat swing detection
        if bat_trk is not None:
            bat_speed_px = math.hypot(*bat_trk.velocity_px)
            if (
                self._pitch_active
                and not self._swing_detected_in_pitch
                and bat_speed_px >= self.swing_velocity_threshold_px
            ):
                self._swing_detected_in_pitch = True
                self._swings_count += 1
                events.append(
                    SportEvent(
                        event_type="bat_swing",
                        sport=self.sport,
                        frame_index=packet.frame_index,
                        timestamp_sec=packet.timestamp_sec,
                        confidence=0.88,
                        involved_track_ids=[bat_trk.track_id],
                        details={"bat_speed_px_per_frame": round(bat_speed_px, 2)},
                    )
                )

        # 3. Bat-ball contact (ball-in-play) or Glove catch (pitch completion)
        if self._pitch_active and ball_trk is not None and bat_trk is not None:
            bx, by = ball_trk.bbox.center
            tx, ty = bat_trk.bbox.center
            dist = math.hypot(bx - tx, by - ty)
            if (
                self._swing_detected_in_pitch
                and not self._contact_detected_in_pitch
                and dist <= self.contact_distance_px
            ):
                self._contact_detected_in_pitch = True
                self._contacts_count += 1
                exit_velo = round(math.hypot(*ball_trk.velocity_px) * 3.7, 1)
                self._exit_velocities_mph.append(exit_velo)
                events.append(
                    SportEvent(
                        event_type="ball_in_play_contact",
                        sport=self.sport,
                        frame_index=packet.frame_index,
                        timestamp_sec=packet.timestamp_sec,
                        confidence=0.92,
                        involved_track_ids=[ball_trk.track_id, bat_trk.track_id],
                        details={"estimated_exit_velocity_mph": exit_velo},
                    )
                )
                self._pitch_active = False

        if self._pitch_active and ball_trk is not None and glove_trk is not None:
            bx, by = ball_trk.bbox.center
            gx, gy = glove_trk.bbox.center
            if math.hypot(bx - gx, by - gy) <= self.contact_distance_px:
                events.append(
                    SportEvent(
                        event_type="catcher_glove_receive",
                        sport=self.sport,
                        frame_index=packet.frame_index,
                        timestamp_sec=packet.timestamp_sec,
                        confidence=0.86,
                        involved_track_ids=[ball_trk.track_id, glove_trk.track_id],
                        details={"swing_occurred": self._swing_detected_in_pitch},
                    )
                )
                self._pitch_active = False
                self._contact_detected_in_pitch = False

        packet.sport_analytics["baseball"] = {
            "pitch_active": self._pitch_active,
            "assigned_roles": roles,
            "pitches_seen": self._pitches_count,
            "swings": self._swings_count,
            "contacts": self._contacts_count,
        }
        return events

    def _assign_player_roles(self, players: list[Tracklet]) -> dict[int, str]:
        """Assign pitcher, batter, catcher, or fielder roles from field geometry or detector metadata."""
        roles: dict[int, str] = {}
        for p in players:
            if p.identity.role:
                roles[p.track_id] = p.identity.role
                continue
            if p.field_position is not None:
                fx, fy = p.field_position.x, p.field_position.y
                if math.hypot(fx, fy - 60.5) <= 10.0:
                    p.identity.role = "pitcher"
                elif math.hypot(fx, fy) <= 8.0 and fy < 0:
                    p.identity.role = "catcher"
                elif math.hypot(fx, fy) <= 8.0:
                    p.identity.role = "batter"
                else:
                    p.identity.role = "fielder"
                roles[p.track_id] = p.identity.role
        return roles

    def generate_scouting_report(self) -> dict[str, Any]:
        avg_pitch = (
            round(sum(self._pitch_velocities_mph) / len(self._pitch_velocities_mph), 1)
            if self._pitch_velocities_mph
            else 0.0
        )
        max_pitch = max(self._pitch_velocities_mph, default=0.0)
        avg_exit = (
            round(sum(self._exit_velocities_mph) / len(self._exit_velocities_mph), 1)
            if self._exit_velocities_mph
            else 0.0
        )
        whiff_rate = (
            round(
                (self._swings_count - self._contacts_count) / self._swings_count,
                3,
            )
            if self._swings_count > 0
            else 0.0
        )
        return {
            "sport": self.sport.value,
            "pitches_tracked": self._pitches_count,
            "avg_pitch_velocity_mph": avg_pitch,
            "max_pitch_velocity_mph": max_pitch,
            "swings_tracked": self._swings_count,
            "contacts_tracked": self._contacts_count,
            "whiff_rate": whiff_rate,
            "avg_exit_velocity_mph": avg_exit,
        }
