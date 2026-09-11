"""A thin HTTP API in front of the CLI's own modules, plus a static frontend.

OOM's ``load.php`` sends no CORS headers, so a browser page cannot call it
directly. Everything here just proxies to :mod:`streeto_planner.oom`,
:mod:`streeto_planner.mapid`, :mod:`streeto_planner.distances` and
:mod:`streeto_planner.solve` on the server side, where CORS does not apply.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles

from streeto_planner import distances, mapid, oom
from streeto_planner.solve import solve

load_dotenv()

app = FastAPI(title="Streeto Planner")

STATIC_DIR = Path(__file__).parent / "static"


def _course_summary(course: oom.Course) -> dict[str, Any]:
    return {
        "map_id": course.map_id,
        "title": course.title,
        "event_date": course.event_date,
        "club": course.club,
        "scale": course.scale,
        "rotation": course.rotation,
        "minted": oom.created_at(course.map_id).isoformat(),
        "total_points": course.total_points,
        "start_finish": {
            "lat": course.start_finish.latitude,
            "lon": course.start_finish.longitude,
        },
        "controls": [
            {
                "number": c.number,
                "lat": c.latitude,
                "lon": c.longitude,
                "points": c.points,
            }
            for c in course.controls
        ],
        "anomalies": [
            {"number": c.number, "points": c.points, "implied": c.implied_points}
            for c in course.scoring_anomalies()
        ],
        "boundary": distances.course_boundary(course),
    }


@app.get("/api/course/{map_id}")
def get_course(map_id: str) -> dict[str, Any]:
    try:
        course = oom.fetch_course(map_id)
    except (ValueError, LookupError, OSError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return _course_summary(course)


@app.get("/api/solve/{map_id}")
def get_solve(
    map_id: str,
    km: list[float] = Query(default=[5, 8, 10, 13, 15]),  # noqa: B008
    routed: bool = False,
    provider: str = "ors",
    detour: float = distances.DEFAULT_DETOUR,
    iterations: int = 60,
    seed: int = 0,
) -> dict[str, Any]:
    warnings: list[distances.SnapWarning] = []
    try:
        course = oom.fetch_course(map_id)
        nodes = course.nodes
        if routed:
            if provider == "mapbox":
                matrix, warnings = distances.mapbox_matrix(nodes)
                caveat = "real pedestrian routing (Mapbox)"
            else:
                matrix, warnings = distances.routed_matrix(nodes)
                caveat = "real pedestrian routing (OpenRouteService)"
        else:
            matrix = distances.approximate_matrix(nodes, detour)
            caveat = (
                f"approximate: straight line x {detour}, historically "
                f"~{distances.APPROXIMATION_OPTIMISM:.0%} optimistic"
            )

        results = []
        for budget_km in km:
            route = solve(nodes, matrix, budget_km * 1000, iterations, seed)
            results.append(
                {
                    "km": budget_km,
                    "points": route.points,
                    "percent": (
                        100 * route.points / course.total_points
                        if course.total_points
                        else 0
                    ),
                    "distance": round(route.distance),
                    "gap": route.gap,
                    "visits": [c.number for c in route.controls],
                }
            )
    except (ValueError, LookupError, RuntimeError, OSError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return {
        "course": _course_summary(course),
        "caveat": caveat,
        "warnings": [
            {"number": w.number, "snapped_distance": w.snapped_distance}
            for w in warnings
        ],
        "results": results,
    }


@app.get("/api/geometry/{map_id}")
def get_geometry(
    map_id: str,
    numbers: list[str] = Query(default=[]),  # noqa: B008
) -> dict[str, Any]:
    """Real walking-route geometry through the given controls, in order."""
    if not numbers:
        raise HTTPException(
            status_code=400, detail="no controls given to route through"
        )
    try:
        course = oom.fetch_course(map_id)
        by_number = {c.number: c for c in course.controls}
        waypoints = [
            course.start_finish,
            *(by_number[n] for n in numbers),
            course.start_finish,
        ]
        geojson = distances.route_geometry(waypoints)
        strays = distances.route_strays_outside_course(geojson, course)
    except KeyError as error:
        raise HTTPException(
            status_code=400, detail=f"unknown control number {error}"
        ) from error
    except (ValueError, LookupError, RuntimeError, OSError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return {"geojson": geojson, "strays_outside_course": strays}


@app.post("/api/resolve")
def post_resolve(payload: dict[str, Any]) -> dict[str, Any]:
    reading = payload.get("reading", "")
    max_requests = int(payload.get("max_requests", 25))
    try:
        options = mapid.shortlist(reading, latest=dt.date.today())
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    if len(options) > max_requests:
        return {
            "status": "too_many",
            "count": len(options),
            "preview": [
                {"map_id": c, "minted": oom.created_at(c).date().isoformat()}
                for c in options[:10]
            ],
        }

    for candidate in options:
        try:
            course = oom.fetch_course(candidate)
        except LookupError:
            continue
        return {"status": "found", "course": _course_summary(course)}

    return {"status": "not_found", "tried": len(options)}


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


def run() -> None:
    """Entry point for ``streeto-web``.

    Binds to 127.0.0.1 by default. Pass ``--host 0.0.0.0`` to make it
    reachable from other devices (e.g. a phone) on the same network.
    """
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(prog="streeto-web")
    parser.add_argument(
        "--host", default="127.0.0.1", help="use 0.0.0.0 for LAN access"
    )
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    run()
