"""Player and individual identity resolution via temporal voting over jersey numbers and embeddings."""

from __future__ import annotations

from collections import Counter, defaultdict

from foundryvision.core.schema import ObjectCategory, Tracklet


class IdentityResolver:
    """Temporal identity stabilizer combining team assignment, jersey OCR, and roster lookup."""

    def __init__(self, roster: dict[tuple[str, int], str] | None = None) -> None:
        self.roster = roster or {}
        self._team_votes: dict[int, Counter[str]] = defaultdict(Counter)
        self._jersey_votes: dict[int, Counter[int]] = defaultdict(Counter)

    def update(self, tracks: list[Tracklet]) -> None:
        for trk in tracks:
            if trk.category != ObjectCategory.PLAYER:
                continue

            if trk.identity.team_id:
                self._team_votes[trk.track_id][trk.identity.team_id] += 1
            if trk.identity.jersey_number is not None:
                self._jersey_votes[trk.track_id][trk.identity.jersey_number] += 1

            team_counts = self._team_votes[trk.track_id]
            if team_counts:
                trk.identity.team_id = team_counts.most_common(1)[0][0]

            jersey_counts = self._jersey_votes[trk.track_id]
            if jersey_counts:
                best_num, votes = jersey_counts.most_common(1)[0]
                total = sum(jersey_counts.values())
                trk.identity.jersey_number = best_num
                trk.identity.confidence = round(votes / max(1, total), 3)

            if trk.identity.team_id and trk.identity.jersey_number is not None:
                key = (trk.identity.team_id, trk.identity.jersey_number)
                trk.identity.athlete_id = self.roster.get(
                    key, f"{trk.identity.team_id}_#{trk.identity.jersey_number}"
                )
