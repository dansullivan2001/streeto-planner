"""Distance approximation and detour calibration."""

from __future__ import annotations

import pytest
from conftest import METRES_PER_DEGREE_LAT, control

from streeto_planner.distances import (
    approximate_matrix,
    detour_index,
    haversine,
)


def test_haversine_matches_a_degree_of_latitude() -> None:
    a = control("S", 0, 0)
    b = control("A", METRES_PER_DEGREE_LAT * 1.0, 0)
    assert haversine(a, b) == pytest.approx(METRES_PER_DEGREE_LAT, rel=1e-3)


def test_haversine_is_zero_for_one_place() -> None:
    a = control("S", 0, 0)
    assert haversine(a, a) == pytest.approx(0.0, abs=1e-9)


def test_matrix_is_symmetric_with_an_empty_diagonal() -> None:
    nodes = [control("S", 0, 0), control("A", 500, 30), control("B", 1200, 50)]
    matrix = approximate_matrix(nodes, 1.0)
    for i in range(len(nodes)):
        assert matrix[i][i] == 0.0
        for j in range(len(nodes)):
            assert matrix[i][j] == pytest.approx(matrix[j][i])


def test_detour_index_scales_every_leg() -> None:
    nodes = [control("S", 0, 0), control("A", 500, 30)]
    plain = approximate_matrix(nodes, 1.0)
    inflated = approximate_matrix(nodes, 1.45)
    assert inflated[0][1] == pytest.approx(plain[0][1] * 1.45)


def test_a_detour_index_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="shorter than a straight line"):
        approximate_matrix([control("S", 0, 0)], 0.9)


def test_detour_index_recovers_the_ratio() -> None:
    visits = [control("S", 0, 0), control("A", 1000, 30), control("S", 0, 0)]
    straight = 2 * METRES_PER_DEGREE_LAT * (1000 / METRES_PER_DEGREE_LAT)
    assert detour_index(visits, straight * 1.4) == pytest.approx(1.4, rel=1e-3)


def test_detour_index_needs_a_real_route() -> None:
    with pytest.raises(ValueError, match="zero-length"):
        detour_index([control("S", 0, 0), control("S", 0, 0)], 100.0)
