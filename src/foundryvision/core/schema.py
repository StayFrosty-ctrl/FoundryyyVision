"""Unified data schemas for frames, detections, tracks, calibration, identities, and events."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SportType(str, Enum):
    BASEBALL = "baseball"
    FOOTBALL = "football"


class ObjectCategory(str, Enum):
    PLAYER = "player"
    BALL = "ball"
    REFEREE = "referee"
    BAT = "bat"
    GLOVE = "glove"
    HOME_PLATE = "home_plate"
    HELMET = "helmet"
    YARD_MARKER = "yard_marker"


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned 2D bounding box in pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5)

    @property
    def bottom_center(self) -> tuple[float, float]:
        """Ground contact point (footpoint) used for planar field projection."""
        return ((self.x1 + self.x2) * 0.5, self.y2)

    def iou(self, other: BoundingBox) -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter <= 0.0:
            return 0.0
        union = self.width * self.height + other.width * other.height - inter
        return inter / union if union > 0.0 else 0.0


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float
    confidence: float = 1.0


@dataclass(frozen=True)
class FieldCoordinate:
    """Standardized 2D real-world field/diamond coordinate (e.g. yards or feet)."""

    x: float
    y: float
    unit: str = "yards"


@dataclass
class Detection:
    """Single-frame object detection output."""

    bbox: BoundingBox
    category: ObjectCategory
    confidence: float
    keypoints: dict[str, Point2D] = field(default_factory=dict)
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlayerIdentity:
    """Resolved player identity, team affiliation, jersey number, and tactical role."""

    athlete_id: str | None = None
    team_id: str | None = None
    jersey_number: int | None = None
    role: str | None = None
    confidence: float = 0.0


@dataclass
class Tracklet:
    """Multi-frame tracked entity with kinematic state and resolved identity."""

    track_id: int
    category: ObjectCategory
    bbox: BoundingBox
    confidence: float
    age: int = 1
    hits: int = 1
    time_since_update: int = 0
    velocity_px: tuple[float, float] = (0.0, 0.0)
    field_position: FieldCoordinate | None = None
    speed_mph: float = 0.0
    identity: PlayerIdentity = field(default_factory=PlayerIdentity)
    keypoints: dict[str, Point2D] = field(default_factory=dict)
    history: list[tuple[int, BoundingBox]] = field(default_factory=list)
    field_history: list[tuple[int, FieldCoordinate]] = field(default_factory=list)


@dataclass
class SportEvent:
    """Discrete temporal event or action spotted in a video stream."""

    event_type: str
    sport: SportType
    frame_index: int
    timestamp_sec: float
    confidence: float
    involved_track_ids: list[int] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class FramePacket:
    """Container for a single video frame and all enriched telemetry."""

    frame_index: int
    timestamp_sec: float
    width: int
    height: int
    fps: float = 30.0
    camera_id: str = "cam_0"
    raw_frame: Any = None
    field_keypoints: dict[str, Point2D] = field(default_factory=dict)
    detections: list[Detection] = field(default_factory=list)
    tracks: list[Tracklet] = field(default_factory=list)
    events: list[SportEvent] = field(default_factory=list)
    sport_analytics: dict[str, Any] = field(default_factory=dict)
