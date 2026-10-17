"""Real-time vision pipeline orchestrating ingest, perception, calibration, tracking, identity, and sport plugins."""

from __future__ import annotations

from foundryvision.core.calibration import PlanarFieldCalibrator
from foundryvision.core.events import EventLog
from foundryvision.core.identity import IdentityResolver
from foundryvision.core.ingest import FrameRingBuffer
from foundryvision.core.perception import PassthroughPerception, PerceptionModel
from foundryvision.core.schema import FramePacket
from foundryvision.core.tracking import MultiObjectTracker
from foundryvision.sports.base import SportPlugin


class VisionPipeline:
    """End-to-end sport-agnostic pipeline paired with a pluggable SportPlugin."""

    def __init__(
        self,
        sport_plugin: SportPlugin,
        perception: PerceptionModel | None = None,
        tracker: MultiObjectTracker | None = None,
        identity_resolver: IdentityResolver | None = None,
        ring_buffer_size: int = 300,
    ) -> None:
        self.sport_plugin = sport_plugin
        self.perception = perception or PassthroughPerception()
        self.tracker = tracker or MultiObjectTracker()
        self.identity_resolver = identity_resolver or IdentityResolver()
        self.calibrator = PlanarFieldCalibrator(
            reference_landmarks=sport_plugin.reference_landmarks(),
            unit=sport_plugin.coordinate_unit(),
        )
        self.ring_buffer = FrameRingBuffer(maxlen=ring_buffer_size)
        self.event_log = EventLog()

    def step(self, packet: FramePacket) -> FramePacket:
        """Process a single FramePacket through all core stages and the active sport plugin."""
        detections, keypoints = self.perception.infer(packet)
        packet.detections = detections
        packet.field_keypoints = keypoints

        self.calibrator.update_from_keypoints(packet.field_keypoints)

        active_tracks = self.tracker.update(
            detections=packet.detections,
            frame_index=packet.frame_index,
            fps=packet.fps,
            calibrator=self.calibrator,
        )
        self.identity_resolver.update(active_tracks)
        packet.tracks = active_tracks

        events = self.sport_plugin.process_frame(packet)
        packet.events = events
        self.event_log.record(events)
        self.ring_buffer.push(packet)
        return packet
