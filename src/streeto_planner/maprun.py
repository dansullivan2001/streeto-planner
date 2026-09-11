"""Real event results from MapRun's public results API.

MapRun (the GPS-tracked score-orienteering app) publishes a small, documented,
key-free API for an event's results -- see maprunners.weebly.com/access-to-
results.html. It has no GPS trace data (that's served by a private endpoint
the results-viewer page uses internally, out of scope here): just each
runner's score, distance, time and the order they punched controls in, which
is enough to compare a real run against what the solver finds for the same
distance budget.

MapRun's own docs ask for "reasonable use": no more than one call every 10
seconds, and only for a handful of current events, not bulk/every event for a
club. Callers should cache rather than refetch casually.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

RESULTS_URL = "https://p.fne.com.au:8886/resultsGetPublicForEventv2"
USER_AGENT = "streeto-planner/0.1"


@dataclass(frozen=True)
class Result:
    """One runner's actual performance at a MapRun event."""

    name: str
    club: str
    classifier: str
    distance_km: float
    gross_score: int
    net_score: int
    total_time_secs: int
    visits: list[str]

    @property
    def total_time_hhmmss(self) -> str:
        minutes, seconds = divmod(self.total_time_secs, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"


def parse_results(data: dict[str, Any]) -> list[Result]:
    """Turn a MapRun results payload into a list of Result, without the network."""
    if data.get("errorFlag"):
        raise LookupError(data.get("statusMessage") or "MapRun reported an error")

    rows = data.get("results")
    if not isinstance(rows, list):
        raise ValueError("MapRun payload has no results list")

    results = []
    for row in rows:
        name = f"{row.get('Firstname', '')} {row.get('Surname', '')}".strip()
        # "(Extra)" entries are re-punches MapRun records outside the main
        # sequence -- not part of the route actually walked in order.
        visits = [c for c in row.get("punchControlIds", []) if "(Extra)" not in c]
        results.append(
            Result(
                name=name or "(unnamed)",
                club=str(row.get("ClubName", "")),
                classifier=str(row.get("Classifier", "")),
                distance_km=float(row.get("Distance", 0.0)),
                gross_score=int(row.get("GrossScore", 0)),
                net_score=int(row.get("NetScore", 0)),
                total_time_secs=int(row.get("TotalTimeSecs", 0)),
                visits=visits,
            )
        )
    return results


def fetch_results(event_name: str, timeout: float = 30.0) -> list[Result]:
    """Download one event's results from MapRun by its full event name."""
    url = f"{RESULTS_URL}?{urlencode({'eventName': event_name})}"
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:  # noqa: S310
        data = json.load(response)
    return parse_results(data)
