"""Turn a solved route into files people can share or load onto a watch."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from xml.sax.saxutils import escape

from streeto_planner.oom import Course
from streeto_planner.solve import Route


def straight_line(course: Course, route: Route) -> list[tuple[float, float]]:
    """The route as straight legs between controls, as (lat, lon) pairs."""
    stops = [course.start_finish, *route.controls, course.start_finish]
    return [(c.latitude, c.longitude) for c in stops]


def geometry_line(geometry: dict[str, Any]) -> list[tuple[float, float]]:
    """A routed GeoJSON path flattened to (lat, lon) pairs.

    GeoJSON stores [lon, lat] (optionally with elevation), so this swaps
    them round to match everything else here.
    """
    line: list[tuple[float, float]] = []
    for feature in geometry.get("features", []):
        coordinates = feature.get("geometry", {}).get("coordinates", [])
        line += [(point[1], point[0]) for point in coordinates]
    return line


def route_gpx(
    course: Course, route: Route, line: Sequence[tuple[float, float]]
) -> str:
    """A GPX file: one waypoint per stop, plus the path as a track.

    Waypoints are named in visit order ("03 - 31 (30pts)") so they sort
    the right way in apps that list them alphabetically.
    """
    name = course.title or course.map_id

    def wpt(lat: float, lon: float, label: str) -> str:
        return (
            f'  <wpt lat="{lat:.6f}" lon="{lon:.6f}">'
            f"<name>{escape(label)}</name></wpt>"
        )

    sf = course.start_finish
    waypoints = [wpt(sf.latitude, sf.longitude, "Start/Finish")]
    width = len(str(len(route.controls)))
    for i, c in enumerate(route.controls, start=1):
        label = f"{i:0{width}d} - {c.number} ({c.points}pts)"
        waypoints.append(wpt(c.latitude, c.longitude, label))

    track = "\n".join(
        f'      <trkpt lat="{lat:.6f}" lon="{lon:.6f}"/>' for lat, lon in line
    )
    description = (
        f"{route.points} points, {len(route.controls)} controls, "
        f"{round(route.distance)} m"
    )
    return "\n".join([
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<gpx version="1.1" creator="Streeto Planner" '
        'xmlns="http://www.topografix.com/GPX/1/1">',
        f"  <metadata><name>{escape(name)}</name>"
        f"<desc>{escape(description)}</desc></metadata>",
        *waypoints,
        f"  <trk><name>{escape(name)}</name><desc>{escape(description)}</desc>",
        "    <trkseg>",
        track,
        "    </trkseg>",
        "  </trk>",
        "</gpx>",
        "",
    ])


def file_stem(course: Course, route: Route) -> str:
    """A filesystem-safe name for exported files."""
    title = course.title or course.map_id
    safe = "".join(ch if ch.isalnum() else "-" for ch in title.lower())
    safe = "-".join(part for part in safe.split("-") if part)
    return f"{safe or 'route'}-{route.points}pts-{round(route.distance)}m"

