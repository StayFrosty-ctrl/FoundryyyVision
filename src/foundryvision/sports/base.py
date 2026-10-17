"""Base interface for sport-specific intelligence and scouting plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from foundryvision.core.schema import FramePacket, SportEvent, SportType


class SportPlugin(ABC):
    """Contract implemented by Baseball and American Football domain modules."""

    sport: SportType

    @abstractmethod
    def reference_landmarks(self) -> dict[str, tuple[float, float]]:
        """Standardized 2D field/diamond coordinates for camera homography calibration."""

    @abstractmethod
    def coordinate_unit(self) -> str:
        """Unit of measurement ('feet' for Baseball, 'yards' for American Football)."""

    @abstractmethod
    def process_frame(self, packet: FramePacket) -> list[SportEvent]:
        """Analyze current tracks & kinematics to emit SportEvents and enrich packet.sport_analytics."""

    @abstractmethod
    def generate_scouting_report(self) -> dict[str, Any]:
        """Summarize accumulated play-by-play metrics into a structured scouting report."""
