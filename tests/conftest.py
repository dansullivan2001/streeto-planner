"""Shared fixtures: a small synthetic course laid out on a known grid."""

from __future__ import annotations

import pytest

from streeto_planner.oom import Control, Course

# Roughly metres per degree at London's latitude, used to place test controls
# at distances we can reason about by hand.
METRES_PER_DEGREE_LAT = 111_195.0


@pytest.fixture
def payload() -> dict:
    """A minimal OOM load.php payload with three scored controls."""
    return {
        "title": "TEST STREET O",
        "eventdate": "2026-03-17",
        "club": "test",
        "scale": "s10000",
        "rotation": "-0.3929",
        "centre_lat": "51.3315",
        "centre_lon": "-0.4000",
        "controls": [
            {"type": "c_startfinish", "number": "S",
             "wgs84lat": "51.3315", "wgs84lon": "-0.4000", "score": "0"},
            {"type": "c_regular", "number": "31",
             "wgs84lat": "51.3405", "wgs84lon": "-0.4000", "score": "30"},
            {"type": "c_regular", "number": "10",
             "wgs84lat": "51.3325", "wgs84lon": "-0.4000", "score": "10"},
            {"type": "c_regular", "number": "52",
             "wgs84lat": "51.3345", "wgs84lon": "-0.4000", "score": "50"},
            {"type": "c_cross", "number": "X",
             "wgs84lat": "51.3300", "wgs84lon": "-0.4000", "score": "0"},
        ],
    }


@pytest.fixture
def course(payload: dict) -> Course:
    from streeto_planner.oom import parse_course

    return parse_course("698d11dd9c78c", payload)


def control(number: str, north_metres: float, points: int) -> Control:
    """Place a control a given distance due north of the test start."""
    return Control(
        number=number,
        latitude=51.3315 + north_metres / METRES_PER_DEGREE_LAT,
        longitude=-0.4000,
        points=points,
    )
