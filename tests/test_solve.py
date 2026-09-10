"""The route solver: budgets are respected and more budget never hurts."""

from __future__ import annotations

import pytest
from conftest import control

from streeto_planner.distances import approximate_matrix
from streeto_planner.solve import solve, upper_bound


@pytest.fixture
def line_course() -> list:
    """Start plus four controls strung out due north at 500 m intervals."""
    return [
        control("start_finish", 0, 0),
        control("11", 500, 10),
        control("22", 1000, 20),
        control("33", 1500, 30),
        control("44", 2000, 40),
    ]


def test_route_starts_and_finishes_at_the_start(line_course) -> None:
    matrix = approximate_matrix(line_course, 1.0)
    route = solve(line_course, matrix, 3000)
    assert route.visits[0].number == "start_finish"
    assert route.visits[-1].number == "start_finish"


def test_route_respects_the_budget(line_course) -> None:
    matrix = approximate_matrix(line_course, 1.0)
    for budget in (1200, 2500, 4000, 6000):
        route = solve(line_course, matrix, budget)
        assert route.distance <= budget + 1e-6


def test_a_bigger_budget_never_scores_less(line_course) -> None:
    matrix = approximate_matrix(line_course, 1.0)
    scores = [solve(line_course, matrix, km).points for km in range(1000, 6000, 500)]
    assert scores == sorted(scores)


def test_out_and_back_reaches_the_far_control(line_course) -> None:
    # The controls are collinear, so 2000 m out and back collects all of them.
    # Allow a little headroom: the fixture places controls to the nearest metre,
    # so the true haversine legs do not sum to exactly 4000 m.
    matrix = approximate_matrix(line_course, 1.0)
    route = solve(line_course, matrix, 4100)
    assert route.points == 100


def test_a_tight_budget_collects_nothing(line_course) -> None:
    matrix = approximate_matrix(line_course, 1.0)
    route = solve(line_course, matrix, 100)
    assert route.points == 0
    assert route.controls == []


def test_upper_bound_drops_unreachable_controls(line_course) -> None:
    matrix = approximate_matrix(line_course, 1.0)
    values = [node.points for node in line_course]
    # 2200 m only allows an out-and-back to the 1000 m control.
    assert upper_bound(matrix, values, 2200) == 30
    assert upper_bound(matrix, values, 100_000) == 100


def test_bound_is_never_beaten_by_the_route(line_course) -> None:
    matrix = approximate_matrix(line_course, 1.0)
    for budget in (1500, 3000, 4500):
        route = solve(line_course, matrix, budget)
        assert route.points <= route.bound
        assert 0.0 <= route.gap <= 1.0


def test_a_negative_budget_is_rejected(line_course) -> None:
    matrix = approximate_matrix(line_course, 1.0)
    with pytest.raises(ValueError, match="must be positive"):
        solve(line_course, matrix, -1)


def test_matrix_and_nodes_must_agree(line_course) -> None:
    matrix = approximate_matrix(line_course[:3], 1.0)
    with pytest.raises(ValueError, match="disagree in size"):
        solve(line_course, matrix, 1000)


def test_solving_is_deterministic_for_a_seed(line_course) -> None:
    matrix = approximate_matrix(line_course, 1.0)
    first = solve(line_course, matrix, 3000, seed=7)
    second = solve(line_course, matrix, 3000, seed=7)
    assert [c.number for c in first.controls] == [c.number for c in second.controls]
