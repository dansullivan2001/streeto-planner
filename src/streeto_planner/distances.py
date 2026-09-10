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
from urllib.request import Request, urlopen

from streeto_planner.oom import Control

EARTH_RADIUS_M = 6_371_008.8
ORS_MATRIX_URL = "https://api.heigit.org/openrouteservice/v2/matrix/foot-walking"
USER_AGENT = "streeto-planner/0.1"

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
) -> list[list[float]]:
    """Real pedestrian distances from OpenRouteService."""
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
    return [[float(d) for d in row] for row in distances]


def detour_index(nodes: Sequence[Control], routed_distance: float) -> float:
    """How much longer a real route is than the straight line through it.

    Feed it the controls in visit order and the distance actually routed, and
    it gives back the index to use for approximate solves on similar terrain.
    """
    straight = sum(haversine(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1))
    if straight <= 0:
        raise ValueError("cannot calibrate against a zero-length route")
    return routed_distance / straight
