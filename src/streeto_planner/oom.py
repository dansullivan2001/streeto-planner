"""Read Street-O courses from OpenOrienteeringMap (OOM) v4.

OOM map IDs are PHP ``uniqid()`` values: eight hex digits of Unix seconds
followed by five of microseconds. Both halves are useful validators when the
ID has been read off a photograph rather than typed.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

LOAD_URL = "https://oomap.dna-software.co.uk/php/load.php"
RENDER_URL = "https://oomap.dna-software.co.uk/render/jpg/"
USER_AGENT = "streeto-planner/0.1"
MAP_ID_PATTERN = re.compile(r"\A[0-9a-f]{13}\Z")
MICROSECOND_LIMIT = 1_000_000
PAPERSIZE_PATTERN = re.compile(r"p(\d+)-(\d+)")


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
    paper_width_m: float
    paper_height_m: float
    start_finish: Control
    controls: list[Control]
    no_access_points: list[tuple[float, float]]

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


def _parse_papersize_mm(data: dict[str, Any]) -> tuple[float, float] | None:
    """The sheet's (long side, short side) in real millimetres, or None.

    ``papersize`` looks like ``p2970-2100``: both sides in tenths of a
    millimetre (2970 tenths-mm = 297.0mm, i.e. A4).
    """
    match = PAPERSIZE_PATTERN.match(str(data.get("papersize", "")))
    if not match:
        return None
    return int(match.group(1)) / 10, int(match.group(2)) / 10


def _oriented(
    long_value: float, short_value: float, data: dict[str, Any]
) -> tuple[float, float]:
    """(width, height) from (long side, short side), honouring orientation."""
    return (
        (short_value, long_value)
        if data.get("paperorientation") == "portrait"
        else (long_value, short_value)
    )


def _paper_dimensions_m(data: dict[str, Any], scale: int) -> tuple[float, float]:
    """Ground width and height the printed sheet covers, in metres.

    Multiplying the sheet's physical size by scale converts a distance on
    paper to the matching distance on the ground.

    Returns ``(0.0, 0.0)`` if the payload has no papersize we can parse -
    callers should treat that as "unknown", not a real zero-size map.
    """
    parsed = _parse_papersize_mm(data)
    if parsed is None:
        return 0.0, 0.0
    long_mm, short_mm = parsed
    width_mm, height_mm = _oriented(long_mm, short_mm, data)
    return width_mm * scale / 1000, height_mm * scale / 1000


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

    # c_cross marks a single point the course setter has explicitly ruled
    # out -- a locked gate, a fence with no gap -- rather than a scored
    # control, even though it carries the same number/score fields as one
    # (unused boilerplate: the rendered map shows no label there).
    no_access_points = [
        (float(c["wgs84lat"]), float(c["wgs84lon"]))
        for c in controls
        if c.get("type") == "c_cross"
    ]

    scale = int(str(data.get("scale", "s10000")).removeprefix("s"))
    paper_width_m, paper_height_m = _paper_dimensions_m(data, scale)

    return Course(
        map_id=map_id,
        title=str(data.get("title", "")),
        event_date=str(data.get("eventdate", "")),
        club=str(data.get("club", "")),
        scale=scale,
        rotation=float(data.get("rotation", 0.0)),
        centre_latitude=float(data["centre_lat"]),
        centre_longitude=float(data["centre_lon"]),
        paper_width_m=paper_width_m,
        paper_height_m=paper_height_m,
        start_finish=Control(
            number="start_finish",
            latitude=float(starts[0]["wgs84lat"]),
            longitude=float(starts[0]["wgs84lon"]),
            points=0,
        ),
        controls=scored,
        no_access_points=no_access_points,
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


def _web_mercator(latitude: float, longitude: float) -> tuple[int, int]:
    """WGS84 to rounded EPSG:3857 (northing, easting), as OOM's renderer wants."""
    lat_radians = math.radians(latitude)
    easting = 6_378_137 * math.radians(longitude)
    northing = 6_378_137 * math.log(math.tan(math.pi / 4 + lat_radians / 2))
    return round(northing), round(easting)


def parse_render_url(map_id: str, data: dict[str, Any]) -> str:
    """Build the URL of OOM's own rendered map image, without touching the network.

    This is the actual printed map: purple control circles, walls, terrain
    symbols and all, at the exact rotation and paper size it was printed at -
    reconstructed from the same fields ``parse_course`` reads, informed by how
    pbatch/StreetoSolver builds this same request (see README credits).
    """
    controls = data.get("controls")
    if not isinstance(controls, list):
        raise ValueError("OOM payload has no control list")

    def positions(control_type: str, include_angle: bool = False) -> str:
        values = []
        for c in controls:
            if not isinstance(c, dict) or c.get("type") != control_type:
                continue
            northing, easting = _web_mercator(
                float(c["wgs84lat"]), float(c["wgs84lon"])
            )
            if include_angle:
                values.extend([str(c.get("angle", 0)), str(northing), str(easting)])
            else:
                values.extend([str(northing), str(easting)])
        return ",".join(values)

    regular = []
    for c in controls:
        if not isinstance(c, dict) or c.get("type") != "c_regular":
            continue
        northing, easting = _web_mercator(float(c["wgs84lat"]), float(c["wgs84lon"]))
        regular.extend(
            [str(c["number"]), str(c.get("angle", 0)), str(northing), str(easting)]
        )

    parsed_size = _parse_papersize_mm(data)
    if parsed_size is None:
        raise ValueError("course has no parseable paper size")
    long_m, short_m = parsed_size[0] / 1000, parsed_size[1] / 1000
    paper_width_m, paper_height_m = _oriented(long_m, short_m, data)

    centre_northing, centre_easting = _web_mercator(
        float(data["centre_lat"]), float(data["centre_lon"])
    )
    scale = str(data.get("scale", "s10000")).removeprefix("s")

    params = [
        f"style={data.get('style', '')}",
        f"paper={paper_width_m:.3f},{paper_height_m:.3f}",
        f"scale={scale}",
        f"centre={centre_northing},{centre_easting}",
        f"title={quote(str(data.get('title', '')), safe='')}",
        f"eventdate={data.get('eventdate', '')}",
        f"club={data.get('club', '')}",
        f"id={map_id}",
        f"start={positions('c_startfinish')}",
        f"finish={positions('c_finish')}",
        f"crosses={positions('c_cross')}",
        f"cps={positions('c_crossingpoint', include_angle=True)}",
        f"controls={','.join(regular)}",
        f"rotation={float(data.get('rotation', 0.0)):.4f}",
        "grid=yes",
        "rail=yes",
        "walls=yes",
        "trees=yes",
        "hedges=yes",
        "drives=no",
        "fences=yes",
        "sidewalks=no",
        "schools=no",
        "power=yes",
        "privroads=yes",
        "buildings=yes",
        "benches=yes",
        "halo=no",
        f"linear={'yes' if str(data.get('linear')) == '1' else 'no'}",
        "dpi=150",
        "purple=A626FF",
    ]
    return RENDER_URL + "?" + "|".join(params)


def render_image_url(map_id: str, timeout: float = 30.0) -> str:
    """Download nothing, but build the URL of OOM's own rendered map image."""
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
    return parse_render_url(map_id, result["data"])
