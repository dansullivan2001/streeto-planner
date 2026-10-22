"""The route solver: budgets are respected and more budget never hurts."""

from __future__ import annotations

import math
import random

import pytest
from conftest import METRES_PER_DEGREE_LAT, control

from streeto_planner.distances import approximate_matrix
from streeto_planner.oom import Control
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


@pytest.fixture
def scattered_course() -> list:
    """Start plus thirty controls scattered over a 3 km square, fixed by seed."""
    rng = random.Random(0)
    # A degree of longitude is shorter than one of latitude this far north.
    metres_per_degree_lon = METRES_PER_DEGREE_LAT * math.cos(math.radians(51.3315))
    nodes = [Control("start_finish", 51.3315, -0.4000, 0)]
    for i in range(30):
        points = rng.choice((10, 20, 30, 40, 50))
        nodes.append(Control(
            number=f"{points // 10}{i:02d}",
            latitude=51.3315 + rng.uniform(-1500, 1500) / METRES_PER_DEGREE_LAT,
            longitude=-0.4000 + rng.uniform(-1500, 1500) / metres_per_degree_lon,
            points=points,
        ))
    return nodes


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


def test_improvement_beats_greedy_alone(scattered_course) -> None:
    # At 11.5 km on this course the greedy construction stalls at 850 points:
    # getting more means giving up controls it has already chosen.
    matrix = approximate_matrix(scattered_course, 1.0)
    greedy = solve(scattered_course, matrix, 11_500, rounds=0)
    improved = solve(scattered_course, matrix, 11_500)
    assert improved.points > greedy.points


def test_improvement_never_scores_less_or_breaks_the_budget(scattered_course) -> None:
    matrix = approximate_matrix(scattered_course, 1.0)
    for budget in (4000, 7000, 10_000):
        greedy = solve(scattered_course, matrix, budget, rounds=0)
        improved = solve(scattered_course, matrix, budget)
        assert improved.points >= greedy.points
        assert improved.distance <= budget + 1e-6
        assert len(improved.controls) == len({c.number for c in improved.controls})


def test_an_unlimited_budget_visits_everything_without_a_longer_route(
    scattered_course,
) -> None:
    matrix = approximate_matrix(scattered_course, 1.0)
    greedy = solve(scattered_course, matrix, math.inf, rounds=0)
    improved = solve(scattered_course, matrix, math.inf)
    assert len(improved.controls) == len(scattered_course) - 1
    assert improved.distance <= greedy.distance + 1e-6
