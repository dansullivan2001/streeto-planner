"""Exporting a solved route: GPX contents and the line it's built from."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from streeto_planner import export
from streeto_planner.distances import approximate_matrix
from streeto_planner.solve import solve

GPX = "{http://www.topografix.com/GPX/1/1}"


def _route(course):
    matrix = approximate_matrix(course.nodes, 1.0)
    return solve(course.nodes, matrix, 1e9)


def test_straight_line_starts_and_ends_at_the_start(course) -> None:
    route = _route(course)
    line = export.straight_line(course, route)
    start = (course.start_finish.latitude, course.start_finish.longitude)
    assert line[0] == start
    assert line[-1] == start
    assert len(line) == len(route.controls) + 2


def test_geometry_line_swaps_geojson_lon_lat() -> None:
    geometry = {
        "type": "FeatureCollection",
        "features": [
            {"geometry": {"coordinates": [[-0.4, 51.3, 12.0], [-0.41, 51.31]]}}
        ],
    }
    assert export.geometry_line(geometry) == [(51.3, -0.4), (51.31, -0.41)]


def test_gpx_has_a_waypoint_per_stop_in_visit_order(course) -> None:
    route = _route(course)
    line = export.straight_line(course, route)
    root = ET.fromstring(export.route_gpx(course, route, line))

    names = [w.findtext(f"{GPX}name") for w in root.findall(f"{GPX}wpt")]
    assert names[0] == "Start/Finish"
    assert names[1:] == [
        f"{i} - {c.number} ({c.points}pts)"
        for i, c in enumerate(route.controls, start=1)
    ]

    points = root.findall(f"{GPX}trk/{GPX}trkseg/{GPX}trkpt")
    assert len(points) == len(line)
    assert float(points[0].get("lat")) == round(line[0][0], 6)


def test_gpx_escapes_awkward_titles(course) -> None:
    from dataclasses import replace

    awkward = replace(course, title="Fish & Chips <Street-O>")
    route = _route(awkward)
    gpx = export.route_gpx(awkward, route, export.straight_line(awkward, route))
    root = ET.fromstring(gpx)
    assert root.findtext(f"{GPX}metadata/{GPX}name") == "Fish & Chips <Street-O>"


def test_file_stem_is_filesystem_safe(course) -> None:
    route = _route(course)
    stem = export.file_stem(course, route)
    assert stem.startswith("test-street-o-")
    assert all(ch.isalnum() or ch == "-" for ch in stem)
