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
    stem = export.file_stem(course, f"{route.points}pts", "vs", "Zoë O'Neil")
    assert stem == f"test-street-o-{route.points}pts-vs-zoë-o-neil"


def _tilted_course():
    """The real Cobham sheet's geometry: A4 landscape, 1:10000, rotated."""
    from streeto_planner.oom import Control, Course

    centre = (51.3315, -0.4)
    return Course(
        map_id="698d11dd9c78c", title="TILTED", event_date="", club="",
        scale=10000, rotation=-0.3929,
        centre_latitude=centre[0], centre_longitude=centre[1],
        paper_width_m=2970.0, paper_height_m=2100.0,
        start_finish=Control("start_finish", *centre, 0),
        controls=[
            Control("31", 51.3355, -0.395, 30),
            Control("52", 51.3290, -0.410, 50),
        ],
        no_access_points=[],
    )


def test_projection_maps_boundary_corners_to_image_corners() -> None:
    from streeto_planner.distances import course_boundary

    course = _tilted_course()
    boundary = course_boundary(course)
    assert boundary is not None
    project = export.image_projection(boundary, (1753, 1240))

    expected = [(0, 1240), (1753, 1240), (1753, 0), (0, 0)]
    for (lat, lon), (x, y) in zip(boundary, expected, strict=True):
        px, py = project(lat, lon)
        assert abs(px - x) < 0.01 and abs(py - y) < 0.01
    cx, cy = project(course.centre_latitude, course.centre_longitude)
    assert abs(cx - 1753 / 2) < 0.01 and abs(cy - 620) < 0.01


def test_route_on_map_keeps_size_and_exports() -> None:
    import io

    from PIL import Image

    from streeto_planner.distances import approximate_matrix, course_boundary

    course = _tilted_course()
    route = solve(course.nodes, approximate_matrix(course.nodes, 1.0), 1e9)
    blank = io.BytesIO()
    Image.new("RGB", (1753, 1240), "white").save(blank, format="JPEG")

    layer = export.route_layer(course, route, export.straight_line(course, route))
    image = export.route_on_map(blank.getvalue(), course_boundary(course), [layer])
    assert image.size == (1753, 1240)
    # Something red got drawn at the start, where the route begins.
    r, g, b = image.getpixel((1753 // 2, 620))
    assert r > 200 and g < 150 and b < 150

    assert export.image_png(image).startswith(b"\x89PNG")
    pdf = export.image_pdf(image, title="TILTED")
    assert pdf.startswith(b"%PDF")
    # At 150 dpi the page is the printed sheet's real size: A4 here.
    import re

    match = re.search(rb"/MediaBox \[ 0 0 ([\d.]+) ([\d.]+) \]", pdf)
    assert match is not None
    width_mm, height_mm = (float(v) / 72 * 25.4 for v in match.groups())
    assert abs(width_mm - 297) < 1 and abs(height_mm - 210) < 1


def test_badges_sit_outside_the_corner() -> None:
    # A right-angle turn at the origin: legs go east and north (screen up),
    # so the clear side is south-west.
    dx, dy = export._badge_direction((10, 0), (0, 0), (0, -10))
    assert dx < 0 and dy > 0


def test_two_routes_get_a_legend_in_the_emptiest_corner() -> None:
    import io

    from PIL import Image

    from streeto_planner.distances import course_boundary

    course = _tilted_course()
    sf, a, b = course.start_finish, *course.controls
    blank = io.BytesIO()
    Image.new("RGB", (1753, 1240), "white").save(blank, format="JPEG")
    layers = [
        export.MapLayer(
            [sf, a, sf], [(sf.latitude, sf.longitude), (a.latitude, a.longitude)],
            export.RUNNER_RGB, label="Runner",
        ),
        export.MapLayer(
            [sf, b, sf], [(sf.latitude, sf.longitude), (b.latitude, b.longitude)],
            export.SOLVER_RGB, label="Solver",
        ),
    ]
    image = export.route_on_map(blank.getvalue(), course_boundary(course), layers)

    # Both colours made it onto the map.
    colours = {colour for _, colour in image.getcolors(maxcolors=1_000_000)}
    assert any(c[2] > 150 and c[0] < 120 for c in colours)  # blue-ish
    assert any(c[0] > 180 and c[1] < 100 and c[2] < 100 for c in colours)  # red

    # Whichever corner the legend picks, no stop should sit under it.
    project = export.image_projection(course_boundary(course), image.size)
    points = [project(c.latitude, c.longitude) for layer in layers for c in layer.stops]
    corner = export._legend_corner(image.size, (300, 60), (60, 95), points)
    x0, y0 = corner
    assert not any(x0 <= x <= x0 + 300 and y0 <= y <= y0 + 60 for x, y in points)


def test_legend_corner_avoids_the_busy_one() -> None:
    size, box, inset = (1000, 800), (200, 50), (10, 10)
    busy_bottom_left = [(50, 770)] * 5
    assert export._legend_corner(size, box, inset, busy_bottom_left) == (790, 740)
