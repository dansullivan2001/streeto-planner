"""Distances between controls, exactly when possible and roughly otherwise.

A real plan needs pedestrian routing. Without an API key we fall back on
great-circle distance inflated by a detour index, which is good enough to
compare courses but not to race on.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from streeto_planner.oom import Control, Course

EARTH_RADIUS_M = 6_371_008.8
ORS_MATRIX_URL = "https://api.heigit.org/openrouteservice/v2/matrix/foot-walking"
ORS_DIRECTIONS_URL = (
    "https://api.heigit.org/openrouteservice/v2/directions/foot-walking/geojson"
)
MAPBOX_MATRIX_URL = "https://api.mapbox.com/directions-matrix/v1/mapbox/walking"
#: Mapbox caps a single request's coordinate list at this many points, well
#: under the size of a typical Street-O course.
MAPBOX_MAX_COORDINATES = 25
USER_AGENT = "streeto-planner/0.1"

#: A control whose exact spot has no nearby mapped footpath (inside a garden,
#: a driveway, a park interior) gets silently snapped to whatever routable
#: point the router can find, however far away. Past this many metres, treat
#: every leg touching that control as unreliable rather than trust it blindly.
SNAP_WARNING_THRESHOLD_M = 50.0


@dataclass(frozen=True)
class SnapWarning:
    """A control whose real position sits well off the routed network."""

    number: str
    snapped_distance: float

#: Detour indices measured by dividing real routed distance by the straight-line
#: distance through the same controls, on two solved London street courses.
#: Dense inner city came out at 1.40, suburban north London at 1.47.
MEASURED_DETOUR = {"inner_city": 1.40, "suburban": 1.47}
DEFAULT_DETOUR = 1.45

#: Approximate matrices flatter the solver: it picks legs that are cheap as the
#: crow flies but blocked in reality. Measured against real routed answers on
#: two courses, the scores it reports came out this much too high.
APPROXIMATION_OPTIMISM = 0.07


def haversine(a: Control, b: Control) -> float:
    """Great-circle distance in metres between two controls."""
    lat1, lon1 = math.radians(a.latitude), math.radians(a.longitude)
    lat2, lon2 = math.radians(b.latitude), math.radians(b.longitude)
    inner = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(inner))


def approximate_matrix(
    nodes: Sequence[Control], detour: float = DEFAULT_DETOUR
) -> list[list[float]]:
    """Straight-line distances inflated by a detour index."""
    if detour < 1:
        raise ValueError("a detour index below 1 would be shorter than a straight line")
    size = len(nodes)
    matrix = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i + 1, size):
            matrix[i][j] = matrix[j][i] = haversine(nodes[i], nodes[j]) * detour
    return matrix


def routed_matrix(
    nodes: Sequence[Control], api_key: str | None = None, timeout: float = 60.0
) -> tuple[list[list[float]], list[SnapWarning]]:
    """Real pedestrian distances from OpenRouteService.

    Alongside the matrix, returns a warning for every control OpenRouteService
    had to snap more than :data:`SNAP_WARNING_THRESHOLD_M` from its real
    position to find a routable point — a sign the distances touching it may
    not reflect where the control actually is.
    """
    api_key = api_key or os.environ.get("OPENROUTESERVICE_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTESERVICE_API_KEY is not set")
    payload = {
        "locations": [[n.longitude, n.latitude] for n in nodes],
        "metrics": ["distance"],
        "units": "m",
    }
    request = Request(
        ORS_MATRIX_URL,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": api_key,
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        },
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:  # noqa: S310
        result = json.load(response)

    distances = result.get("distances")
    if not isinstance(distances, list) or len(distances) != len(nodes):
        raise RuntimeError("routing service returned an invalid distance matrix")
    if any(d is None for row in distances for d in row):
        raise RuntimeError("at least one pair of controls has no pedestrian route")

    warnings = [
        SnapWarning(node.number, source["snapped_distance"])
        for node, source in zip(nodes, result.get("sources") or [], strict=False)
        if isinstance(source, dict)
        and source.get("snapped_distance", 0) > SNAP_WARNING_THRESHOLD_M
    ]
    return [[float(d) for d in row] for row in distances], warnings


METRES_PER_DEGREE_LAT = EARTH_RADIUS_M * math.pi / 180

#: OOM's ``rotation`` field has no documented sign convention. Tested against
#: a real course (Cobham Street O 2026): negating it fit every one of that
#: course's own controls inside the printed sheet with zero slack, while the
#: field's sign as given put one control 423m outside. That is checked fresh
#: for every course below rather than trusted as a universal rule, and this
#: is how much slack (in metres) that check allows for float/projection error
#: before falling back to a cruder area instead of a wrong boundary.
BOUNDARY_FIT_TOLERANCE_M = 25.0

#: Fallback margin around a course's own controls when its exact printed
#: boundary can't be pinned down (no parseable papersize, or a course whose
#: controls don't fit within tolerance on either rotation sign).
COURSE_AREA_MARGIN_M = 250.0


def _overshoot(
    lat: float,
    lon: float,
    centre_lat: float,
    centre_lon: float,
    bearing: float,
    half_height: float,
    half_width: float,
) -> float:
    """How far (lat, lon) sits outside a centred rectangle at this bearing.

    Positive means outside by that many metres; zero or negative means
    inside. The rectangle's "up" (height) axis points along ``bearing``,
    clockwise from north; its "right" (width) axis is 90 degrees clockwise
    from that.
    """
    north = (lat - centre_lat) * METRES_PER_DEGREE_LAT
    east = (
        (lon - centre_lon) * METRES_PER_DEGREE_LAT * math.cos(math.radians(centre_lat))
    )
    height_axis = (math.cos(bearing), math.sin(bearing))
    width_axis = (math.cos(bearing + math.pi / 2), math.sin(bearing + math.pi / 2))
    local_height = north * height_axis[0] + east * height_axis[1]
    local_width = north * width_axis[0] + east * width_axis[1]
    return max(abs(local_height) - half_height, abs(local_width) - half_width)


def _best_bearing(course: Course) -> tuple[float, float]:
    """Whichever rotation sign fits the course's own controls better.

    Returns that bearing and how far (metres) its worst-fitting control still
    sits outside the printed rectangle at that bearing.
    """
    half_height, half_width = course.paper_height_m / 2, course.paper_width_m / 2

    def worst_overshoot(bearing: float) -> float:
        return max(
            _overshoot(
                n.latitude, n.longitude,
                course.centre_latitude, course.centre_longitude,
                bearing, half_height, half_width,
            )
            for n in course.nodes
        )

    candidates = [course.rotation, -course.rotation]
    return min(((b, worst_overshoot(b)) for b in candidates), key=lambda c: c[1])


def course_boundary(
    course: Course, margin: float = 0.0
) -> list[tuple[float, float]] | None:
    """The printed map's four corners as (lat, lon), padded by ``margin`` metres.

    Tests both rotation signs against the course's own controls (see
    :data:`BOUNDARY_FIT_TOLERANCE_M`) and uses whichever fits. Returns None if
    the course has no parseable paper size, or if neither sign fits within
    tolerance — callers should fall back to a cruder check rather than trust a
    boundary this course doesn't confirm.
    """
    if course.paper_width_m <= 0 or course.paper_height_m <= 0:
        return None

    bearing, overshoot = _best_bearing(course)
    if overshoot > BOUNDARY_FIT_TOLERANCE_M:
        return None

    half_height = course.paper_height_m / 2 + margin
    half_width = course.paper_width_m / 2 + margin
    height_axis = (math.cos(bearing), math.sin(bearing))
    width_axis = (math.cos(bearing + math.pi / 2), math.sin(bearing + math.pi / 2))
    m_per_deg_lon = METRES_PER_DEGREE_LAT * math.cos(
        math.radians(course.centre_latitude)
    )

    corners = []
    for width_sign, height_sign in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
        north = (
            height_sign * half_height * height_axis[0]
            + width_sign * half_width * width_axis[0]
        )
        east = (
            height_sign * half_height * height_axis[1]
            + width_sign * half_width * width_axis[1]
        )
        corners.append((
            course.centre_latitude + north / METRES_PER_DEGREE_LAT,
            course.centre_longitude + east / m_per_deg_lon,
        ))
    return corners


def route_strays_outside_course(geometry: dict[str, Any], course: Course) -> bool:
    """Whether any point of a routed path's geometry falls outside the course area.

    Uses the printed map's actual boundary when :func:`course_boundary` can
    confirm one, padded generously for road width and the router's own
    snapping; otherwise falls back to a looser box around the course's own
    controls.
    """
    points = [
        (lat, lon)
        for feature in geometry.get("features", [])
        for lon, lat in feature.get("geometry", {}).get("coordinates", [])
    ]

    if course.paper_width_m > 0 and course.paper_height_m > 0:
        bearing, overshoot = _best_bearing(course)
        if overshoot <= BOUNDARY_FIT_TOLERANCE_M:
            half_height = course.paper_height_m / 2 + 50.0
            half_width = course.paper_width_m / 2 + 50.0
            return any(
                _overshoot(
                    lat, lon,
                    course.centre_latitude, course.centre_longitude,
                    bearing, half_height, half_width,
                ) > 0
                for lat, lon in points
            )

    min_lat, max_lat, min_lon, max_lon = _course_bounds(course.nodes)
    return any(
        not (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon)
        for lat, lon in points
    )


def _course_bounds(
    nodes: Sequence[Control], margin: float = COURSE_AREA_MARGIN_M
) -> tuple[float, float, float, float]:
    """A lat/lon box around every node, padded by ``margin`` metres.

    Fallback stand-in for "the printed map area" when :func:`course_boundary`
    can't confirm the real one.
    """
    lats = [n.latitude for n in nodes]
    lons = [n.longitude for n in nodes]
    lat_pad = margin / METRES_PER_DEGREE_LAT
    mean_lat = sum(lats) / len(lats)
    lon_pad = margin / (METRES_PER_DEGREE_LAT * math.cos(math.radians(mean_lat)))
    return (
        min(lats) - lat_pad,
        max(lats) + lat_pad,
        min(lons) - lon_pad,
        max(lons) + lon_pad,
    )


def route_geometry(
    waypoints: Sequence[Control], api_key: str | None = None, timeout: float = 60.0
) -> dict[str, Any]:
    """Real pedestrian route geometry through an ordered list of waypoints.

    Returns GeoJSON straight from OpenRouteService's Directions API: one
    request through every waypoint in order gives back a single connected
    path, which is what a Street-O runner would actually walk.
    """
    api_key = api_key or os.environ.get("OPENROUTESERVICE_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTESERVICE_API_KEY is not set")
    if len(waypoints) < 2:
        raise ValueError("need at least two waypoints to route between")

    payload = {
        "coordinates": [[w.longitude, w.latitude] for w in waypoints],
        "instructions": False,
    }
    request = Request(
        ORS_DIRECTIONS_URL,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": api_key,
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        },
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:  # noqa: S310
        geojson = json.load(response)
    if geojson.get("type") != "FeatureCollection" or not geojson.get("features"):
        raise RuntimeError("routing service returned invalid GeoJSON")
    return geojson


def mapbox_matrix(
    nodes: Sequence[Control], api_key: str | None = None, timeout: float = 60.0
) -> tuple[list[list[float]], list[SnapWarning]]:
    """Real pedestrian distances from the Mapbox Matrix API.

    Mapbox caps a single request at 25 coordinates total, so a course with
    more controls than that is split into blocks and the full matrix is
    stitched together from one request per pair of blocks. Alongside the
    matrix, returns a warning for every control Mapbox had to snap more than
    :data:`SNAP_WARNING_THRESHOLD_M` from its real position to find a routable
    point.
    """
    api_key = api_key or os.environ.get("MAPBOX_ACCESS_TOKEN")
    if not api_key:
        raise ValueError("MAPBOX_ACCESS_TOKEN is not set")

    size = len(nodes)
    matrix = [[0.0] * size for _ in range(size)]
    snap_distances: dict[int, float] = {}
    block_size = MAPBOX_MAX_COORDINATES // 2
    blocks = [range(i, min(i + block_size, size)) for i in range(0, size, block_size)]

    for source_block in blocks:
        for dest_block in blocks:
            _fill_mapbox_block(
                nodes, source_block, dest_block,
                matrix, snap_distances, api_key, timeout,
            )

    warnings = [
        SnapWarning(nodes[node].number, snapped)
        for node, snapped in sorted(snap_distances.items())
        if snapped > SNAP_WARNING_THRESHOLD_M
    ]
    return matrix, warnings


def _fill_mapbox_block(
    nodes: Sequence[Control],
    source_block: range,
    dest_block: range,
    matrix: list[list[float]],
    snap_distances: dict[int, float],
    api_key: str,
    timeout: float,
) -> None:
    """Fetch one source/destination block pair and write it into ``matrix``."""
    combined = list(dict.fromkeys([*source_block, *dest_block]))
    position = {node: i for i, node in enumerate(combined)}
    coordinates = ";".join(
        f"{nodes[i].longitude},{nodes[i].latitude}" for i in combined
    )
    query = urlencode(
        {
            "access_token": api_key,
            "annotations": "distance",
            "sources": ";".join(str(position[i]) for i in source_block),
            "destinations": ";".join(str(position[i]) for i in dest_block),
        }
    )
    request = Request(
        f"{MAPBOX_MATRIX_URL}/{coordinates}?{query}",
        headers={"User-Agent": USER_AGENT},
    )
    with urlopen(request, timeout=timeout) as response:  # noqa: S310
        result = json.load(response)
    if result.get("code") != "Ok":
        raise RuntimeError(
            f"Mapbox matrix request failed: {result.get('message', result.get('code'))}"
        )

    block = result.get("distances")
    if not isinstance(block, list):
        raise RuntimeError("Mapbox returned no distance matrix")
    for si, source_node in enumerate(source_block):
        if source_node not in snap_distances:
            meta = (result.get("sources") or [None] * len(combined))[si]
            if isinstance(meta, dict) and "distance" in meta:
                snap_distances[source_node] = meta["distance"]
        for di, dest_node in enumerate(dest_block):
            value = block[si][di]
            if value is None:
                raise RuntimeError(
                    "at least one pair of controls has no pedestrian route"
                )
            matrix[source_node][dest_node] = float(value)


def detour_index(nodes: Sequence[Control], routed_distance: float) -> float:
    """How much longer a real route is than the straight line through it.

    Feed it the controls in visit order and the distance actually routed, and
    it gives back the index to use for approximate solves on similar terrain.
    """
    straight = sum(haversine(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1))
    if straight <= 0:
        raise ValueError("cannot calibrate against a zero-length route")
    return routed_distance / straight
