"""Perception interfaces for object detection, pose estimation, and field landmark detection."""

from __future__ import annotations

from typing import Protocol

from foundryvision.core.schema import Detection, FramePacket, Point2D


class PerceptionModel(Protocol):
    """Contract for detectors (YOLO/RF-DETR/Pose) operating on incoming FramePackets."""

    def infer(self, packet: FramePacket) -> tuple[list[Detection], dict[str, Point2D]]:
        """Return object detections and field calibration landmarks for the frame."""
        ...


class PassthroughPerception:
    """Uses pre-populated detections on FramePacket (useful for replay/testing/external inference)."""

    def infer(self, packet: FramePacket) -> tuple[list[Detection], dict[str, Point2D]]:
        return packet.detections, packet.field_keypoints
