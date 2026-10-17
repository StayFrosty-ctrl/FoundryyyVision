"""Video stream ingestion and ring-buffer abstractions."""

from __future__ import annotations

from collections import deque
from typing import Iterator, Protocol

from foundryvision.core.schema import FramePacket


class FrameSource(Protocol):
    """Protocol implemented by RTSP/WebRTC stream readers, file decoders, or simulators."""

    def __iter__(self) -> Iterator[FramePacket]: ...


class FrameRingBuffer:
    """Fixed-capacity ring buffer for instant replay clipping and temporal window lookback."""

    def __init__(self, maxlen: int = 300) -> None:
        if maxlen <= 0:
            raise ValueError("maxlen must be positive")
        self._buffer: deque[FramePacket] = deque(maxlen=maxlen)

    def push(self, packet: FramePacket) -> None:
        self._buffer.append(packet)

    def recent(self, count: int) -> list[FramePacket]:
        if count <= 0:
            return []
        items = list(self._buffer)
        return items[-count:]

    def slice_by_frame(self, start_frame: int, end_frame: int) -> list[FramePacket]:
        return [p for p in self._buffer if start_frame <= p.frame_index <= end_frame]

    def __len__(self) -> int:
        return len(self._buffer)
