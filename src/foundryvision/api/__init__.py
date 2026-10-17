"""FastAPI telemetry and scouting service interfaces."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from foundryvision.core.pipeline import VisionPipeline
from foundryvision.core.schema import FramePacket


def serialize_frame_telemetry(packet: FramePacket) -> dict[str, Any]:
    """Serialize a processed FramePacket into a JSON-ready dictionary for REST/WebSocket streaming."""
    return {
        "frame_index": packet.frame_index,
        "timestamp_sec": packet.timestamp_sec,
        "camera_id": packet.camera_id,
        "tracks": [
            {
                "track_id": t.track_id,
                "category": t.category.value,
                "bbox": asdict(t.bbox),
                "velocity_px": t.velocity_px,
                "speed_mph": t.speed_mph,
                "field_position": asdict(t.field_position) if t.field_position else None,
                "identity": asdict(t.identity),
            }
            for t in packet.tracks
        ],
        "events": [
            {
                "event_type": e.event_type,
                "sport": e.sport.value,
                "frame_index": e.frame_index,
                "timestamp_sec": e.timestamp_sec,
                "confidence": e.confidence,
                "involved_track_ids": e.involved_track_ids,
                "details": e.details,
            }
            for e in packet.events
        ],
        "sport_analytics": packet.sport_analytics,
    }


def build_pipeline_summary(pipeline: VisionPipeline) -> dict[str, Any]:
    """Return aggregated event log and scouting report from an active VisionPipeline."""
    return {
        "sport": pipeline.sport_plugin.sport.value,
        "buffered_frames": len(pipeline.ring_buffer),
        "calibrated": pipeline.calibrator.is_calibrated,
        "total_events": len(pipeline.event_log.all()),
        "events": [
            {
                "event_type": e.event_type,
                "frame_index": e.frame_index,
                "timestamp_sec": e.timestamp_sec,
                "details": e.details,
            }
            for e in pipeline.event_log.all()
        ],
        "scouting_report": pipeline.sport_plugin.generate_scouting_report(),
    }
