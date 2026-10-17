"""Temporal event indexing and searchable clip registry for scouting."""

from __future__ import annotations

from foundryvision.core.schema import SportEvent, SportType


class EventLog:
    """In-memory time-indexed store for spotted actions, plays, and scouting triggers."""

    def __init__(self) -> None:
        self._events: list[SportEvent] = []

    def record(self, events: list[SportEvent]) -> None:
        self._events.extend(events)

    def all(self) -> list[SportEvent]:
        return list(self._events)

    def query(
        self,
        sport: SportType | None = None,
        event_type: str | None = None,
        min_confidence: float = 0.0,
    ) -> list[SportEvent]:
        results: list[SportEvent] = []
        for ev in self._events:
            if sport is not None and ev.sport != sport:
                continue
            if event_type is not None and ev.event_type != event_type:
                continue
            if ev.confidence < min_confidence:
                continue
            results.append(ev)
        return results
