"""Command line entry point."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from streeto_planner import distances, mapid, oom
from streeto_planner.solve import solve


def _course_command(args: argparse.Namespace) -> int:
    course = oom.fetch_course(args.map_id)
    print(f"{course.title}  ({course.event_date}, {course.club})")
    minted = oom.created_at(course.map_id)
    print(f"  map ID    {course.map_id}, minted {minted:%Y-%m-%d %H:%M}Z")
    print(f"  scale     1:{course.scale}, rotation {course.rotation:.4f} rad")
    print(f"  controls  {len(course.controls)} scored, {course.total_points} points")
    anomalies = course.scoring_anomalies()
    if anomalies:
        print("  NOTE: scores departing from the first-digit convention:")
        for control in anomalies:
            print(f"    {control.number}: {control.points} (implies "
                  f"{control.implied_points})")
    return 0


def _resolve_command(args: argparse.Namespace) -> int:
    shortlist = mapid.shortlist(args.reading, latest=dt.date.today())
    print(f"{len(mapid.candidates(args.reading))} spellings -> "
          f"{len(shortlist)} plausible")
    if len(shortlist) > args.max_requests:
        print(f"Too many to try politely (cap {args.max_requests}). "
              "Re-read the printed ID rather than guessing.")
        for candidate in shortlist[:10]:
            print(f"  {candidate}  {oom.created_at(candidate):%Y-%m-%d}")
        return 1
    for candidate in shortlist:
        try:
            course = oom.fetch_course(candidate)
        except LookupError:
            continue
        print(f"\nFOUND {candidate}: {course.title} ({course.event_date})")
        return 0
    print("\nNo candidate matched. Re-read the printed ID.")
    return 1


def _solve_command(args: argparse.Namespace) -> int:
    course = oom.fetch_course(args.map_id)
    nodes = course.nodes
    if args.routed:
        matrix = distances.routed_matrix(nodes)
        caveat = "real pedestrian routing"
    else:
        matrix = distances.approximate_matrix(nodes, args.detour)
        caveat = (
            f"approximate: straight line x {args.detour}, "
            f"historically ~{distances.APPROXIMATION_OPTIMISM:.0%} optimistic"
        )

    print(f"{course.title}: {len(course.controls)} controls, "
          f"{course.total_points} points available")
    print(f"Distances: {caveat}\n")
    print(f"{'km':>4} {'points':>7} {'%':>5} {'stops':>6} {'gap<=':>6}  route")

    results = []
    for km in args.km:
        route = solve(nodes, matrix, km * 1000, args.iterations, args.seed)
        visits = [c.number for c in route.controls]
        results.append({"km": km, "points": route.points,
                        "distance": round(route.distance), "visits": visits})
        print(f"{km:>4} {route.points:>7} "
              f"{100 * route.points / course.total_points:>4.0f}% "
              f"{len(visits):>6} {route.gap:>5.0%}  {' '.join(visits)}")

    if args.json:
        args.json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {args.json}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="streeto", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    course = subparsers.add_parser("course", help="summarise a course by map ID")
    course.add_argument("map_id")
    course.set_defaults(func=_course_command)

    resolve = subparsers.add_parser(
        "resolve", help="recover a map ID from an uncertain reading"
    )
    resolve.add_argument("reading", help="e.g. '698d11dd[9g]c78c'")
    resolve.add_argument("--max-requests", type=int, default=25)
    resolve.set_defaults(func=_resolve_command)

    plan = subparsers.add_parser("solve", help="plan routes for a course")
    plan.add_argument("map_id")
    plan.add_argument("--km", type=int, nargs="+", default=list(range(5, 16)))
    plan.add_argument("--detour", type=float, default=distances.DEFAULT_DETOUR)
    plan.add_argument("--routed", action="store_true",
                      help="use OpenRouteService instead of the approximation")
    plan.add_argument("--iterations", type=int, default=60)
    plan.add_argument("--seed", type=int, default=0)
    plan.add_argument("--json", type=Path)
    plan.set_defaults(func=_solve_command)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (LookupError, ValueError, RuntimeError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
