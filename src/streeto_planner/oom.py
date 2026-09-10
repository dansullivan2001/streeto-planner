"""Read Street-O courses from OpenOrienteeringMap (OOM) v4.

OOM map IDs are PHP ``uniqid()`` values: eight hex digits of Unix seconds
followed by five of microseconds. Both halves are useful validators when the
ID has been read off a photograph rather than typed.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

LOAD_URL = "https://oomap.dna-software.co.uk/php/load.php"
USER_AGENT = "streeto-planner/0.1"
MAP_ID_PATTERN = re.compile(r"\A[0-9a-f]{13}\Z")
MICROSECOND_LIMIT = 1_000_000


@dataclass(frozen=True)
class Control:
    """One point on the course, scored or not."""

    number: str
    latitude: float
    longitude: float
    points: int

    @property
    def implied_points(self) -> int:
        """The score Street-O numbering implies: first digit times ten.

        Holds across every map checked so far, which makes it a strong
        cross-check on a number read by OCR.
        """
        return int(self.number[0]) * 10


@dataclass(frozen=True)
class Course:
    """A whole Street-O event: one start/finish plus the scored controls."""

    map_id: str
    title: str
    event_date: str
    club: str
    scale: int
    rotation: float
    centre_latitude: float
    centre_longitude: float
    start_finish: Control
    controls: list[Control]

    @property
    def total_points(self) -> int:
        """Every point on the map, if you could visit everything."""
        return sum(control.points for control in self.controls)

    @property
    def nodes(self) -> list[Control]:
        """The start/finish followed by the scored controls, in solve order."""
        return [self.start_finish, *self.controls]

    def scoring_anomalies(self) -> list[Control]:
        """Controls whose score disagrees with the first-digit convention."""
        return [c for c in self.controls if c.points != c.implied_points]


def created_at(map_id: str) -> dt.datetime:
    """Recover the moment a map ID was minted, in UTC."""
    if not MAP_ID_PATTERN.match(map_id):
        raise ValueError(f"{map_id!r} is not a 13-digit lowercase hex map ID")
    return dt.datetime.fromtimestamp(int(map_id[:8], 16), tz=dt.UTC)


def is_plausible(map_id: str, earliest: dt.date, latest: dt.date) -> bool:
    """Check an ID against the uniqid structure before spending a request.

    The timestamp half is a strong filter. The microsecond half is weak: only
    about 5% of five-digit hex values exceed one million, so it rejects little.
    """
    if not MAP_ID_PATTERN.match(map_id):
        return False
    if int(map_id[8:], 16) >= MICROSECOND_LIMIT:
        return False
    return earliest <= created_at(map_id).date() <= latest


def parse_course(map_id: str, data: dict[str, Any]) -> Course:
    """Turn an OOM payload into a Course, without touching the network."""
    controls = data.get("controls")
    if not isinstance(controls, list):
        raise ValueError("OOM payload has no control list")

    starts = [c for c in controls if c.get("type") == "c_startfinish"]
    if len(starts) != 1:
        raise ValueError(f"expected one shared start/finish, found {len(starts)}")

    scored = sorted(
        (
            Control(
                number=str(c["number"]),
                latitude=float(c["wgs84lat"]),
                longitude=float(c["wgs84lon"]),
                points=int(c["score"]),
            )
            for c in controls
            if c.get("type") == "c_regular"
        ),
        key=lambda control: int(control.number),
    )
    if not scored:
        raise ValueError("course has no scored controls")

    return Course(
        map_id=map_id,
        title=str(data.get("title", "")),
        event_date=str(data.get("eventdate", "")),
        club=str(data.get("club", "")),
        scale=int(str(data.get("scale", "s10000")).removeprefix("s")),
        rotation=float(data.get("rotation", 0.0)),
        centre_latitude=float(data["centre_lat"]),
        centre_longitude=float(data["centre_lon"]),
        start_finish=Control(
            number="start_finish",
            latitude=float(starts[0]["wgs84lat"]),
            longitude=float(starts[0]["wgs84lon"]),
            points=0,
        ),
        controls=scored,
    )


def fetch_course(map_id: str, timeout: float = 30.0) -> Course:
    """Download one course from OOM by its printed map ID."""
    if not MAP_ID_PATTERN.match(map_id):
        raise ValueError(f"{map_id!r} is not a 13-digit lowercase hex map ID")
    request = Request(
        LOAD_URL,
        data=urlencode({"shortcode": map_id}).encode(),
        headers={"User-Agent": USER_AGENT},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:  # noqa: S310
        result = json.load(response)
    if not result.get("success"):
        raise LookupError(result.get("message", "map not found"))
    return parse_course(map_id, result["data"])
