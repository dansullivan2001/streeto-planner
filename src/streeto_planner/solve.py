"""Choose which controls to visit, and in what order.

This is the orienteering problem: a closed walk from the start that collects as
many points as it can inside a distance budget. It is NP-hard, so what follows
is a randomised greedy insertion with 2-opt cleanup, restarted a few times. At
Street-O sizes (fewer than a hundred controls) it runs in well under a second
and lands close to the best known answers.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

from streeto_planner.oom import Control


@dataclass(frozen=True)
class Route:
    """A closed route, starting and finishing at the start/finish."""

    visits: list[Control]
    distance: float
    points: int
    bound: int

    @property
    def controls(self) -> list[Control]:
        """The scored controls, in the order they are visited."""
        return [c for c in self.visits if c.number != "start_finish"]

    @property
    def gap(self) -> float:
        """Fraction by which the true optimum could still beat this route."""
        return 0.0 if self.bound == 0 else 1 - self.points / self.bound


def upper_bound(matrix: Sequence[Sequence[float]], values: Sequence[int],
                budget: float) -> int:
    """Every point that is even reachable, ignoring the cost of combining them.

    A control can only be visited if going straight out and straight back fits
    inside the budget, so anything further away is unreachable no matter how the
    rest of the route is arranged. Summing the survivors bounds the optimum from
    above. It is loose, but it is honest and it costs nothing.
    """
    return sum(
        value
        for node, value in enumerate(values)
        if node == 0 or matrix[0][node] + matrix[node][0] <= budget
    )


def _tour_length(tour: Sequence[int], matrix: Sequence[Sequence[float]]) -> float:
    """Total length of a closed tour."""
    return sum(matrix[tour[i]][tour[i + 1]] for i in range(len(tour) - 1))


def _two_opt(tour: list[int], matrix: Sequence[Sequence[float]]) -> list[int]:
    """Shorten a tour by reversing any segment that crosses itself."""
    improved = True
    while improved:
        improved = False
        for i in range(1, len(tour) - 2):
            for j in range(i + 1, len(tour) - 1):
                a, b, c, d = tour[i - 1], tour[i], tour[j], tour[j + 1]
                delta = (matrix[a][c] + matrix[b][d]) - (matrix[a][b] + matrix[c][d])
                if delta < -1e-9:
                    tour[i : j + 1] = reversed(tour[i : j + 1])
                    improved = True
    return tour


def _cheapest_insertion(
    tour: Sequence[int], node: int, matrix: Sequence[Sequence[float]]
) -> tuple[float, int]:
    """Where to splice a node into a tour, and the detour it costs."""
    best_cost, best_at = math.inf, 1
    for i in range(1, len(tour)):
        a, b = tour[i - 1], tour[i]
        cost = matrix[a][node] + matrix[node][b] - matrix[a][b]
        if cost < best_cost:
            best_cost, best_at = cost, i
    return best_cost, best_at


def solve(
    nodes: Sequence[Control],
    matrix: Sequence[Sequence[float]],
    budget: float,
    iterations: int = 60,
    seed: int = 0,
) -> Route:
    """Collect as many points as possible within the distance budget.

    ``nodes[0]`` must be the start/finish; the route returns to it.
    """
    if budget <= 0:
        raise ValueError("the distance budget must be positive")
    if len(matrix) != len(nodes):
        raise ValueError("matrix and node list disagree in size")

    values = [node.points for node in nodes]
    rng = random.Random(seed)
    best_tour, best_points, best_distance = [0, 0], 0, 0.0

    for _ in range(iterations):
        tour, unused, length = [0, 0], set(range(1, len(nodes))), 0.0

        # Grow the route by whichever control buys the most points per metre,
        # picking randomly among the best few so restarts explore differently.
        while True:
            options = []
            for node in unused:
                cost, at = _cheapest_insertion(tour, node, matrix)
                if cost > 0 and length + cost <= budget:
                    options.append((values[node] / cost, node, cost, at))
            if not options:
                break
            options.sort(reverse=True)
            _, node, cost, at = options[rng.randrange(min(3, len(options)))]
            tour.insert(at, node)
            unused.discard(node)
            length += cost

        tour = _two_opt(tour, matrix)
        length = _tour_length(tour, matrix)

        # Straightening the route frees up slack; spend it on the best control
        # that still fits, preferring high scores over cheap ones.
        added = True
        while added:
            added = False
            for node in sorted(unused, key=lambda n: -values[n]):
                cost, at = _cheapest_insertion(tour, node, matrix)
                if length + cost <= budget:
                    tour.insert(at, node)
                    unused.discard(node)
                    length += cost
                    added = True
                    break

        points = sum(values[node] for node in set(tour))
        if points > best_points or (points == best_points and length < best_distance):
            best_tour, best_points, best_distance = list(tour), points, length

    return Route(
        visits=[nodes[node] for node in best_tour],
        distance=best_distance,
        points=best_points,
        bound=upper_bound(matrix, values, budget),
    )
