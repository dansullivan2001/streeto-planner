"""Turn a solved route into files people can share or load onto a watch."""

from __future__ import annotations

import io
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from xml.sax.saxutils import escape

from streeto_planner.distances import METRES_PER_DEGREE_LAT
from streeto_planner.oom import Control, Course
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


def file_stem(course: Course, *parts: str) -> str:
    """A filesystem-safe name for exported files: the course title, then parts.

    e.g. ``file_stem(course, "670pts", "7300m")`` gives
    ``cobham-street-o-2026-670pts-7300m``.
    """
    text = " ".join([course.title or course.map_id, *parts])
    safe = "".join(ch if ch.isalnum() else "-" for ch in text.lower())
    return "-".join(part for part in safe.split("-") if part) or "route"


#: OOM renders its map image at this resolution (checked against a real A4
#: course: 1753 x 1240 px), so saving a PDF at it prints at the true scale.
RENDER_DPI = 150

# Solid colours, each with a thin white outline: see-through lines blend
# into the black streets and orange areas and turn muddy where they cross.
SOLVER_RGB = (225, 0, 0)
RUNNER_RGB = (0, 90, 220)
ROUTE_WIDTH_MM = 1.0
ROUTE_OUTLINE_MM = 0.35
BADGE_RADIUS_MM = 2.7
#: How far a visit-order badge sits from its control's centre: just outside
#: OOM's printed control circle (about 6 mm across).
BADGE_OFFSET_MM = 6.0
#: Keeps a legend clear of the sheet's edges, title banner and credits.
LEGEND_INSET_MM = (10.0, 16.0)


@dataclass(frozen=True)
class MapLayer:
    """One route to draw on the printed map."""

    stops: Sequence[Control]  # in visit order, start and finish included
    line: Sequence[tuple[float, float]]  # the path itself, as (lat, lon)
    colour: tuple[int, int, int]
    label: str = ""  # shown in a legend when more than one route is drawn
    #: Draw an underneath route wider, so where two share a street the top
    #: one runs down the middle of it and both stay visible.
    width_mm: float = ROUTE_WIDTH_MM


def route_layer(
    course: Course,
    route: Route,
    line: Sequence[tuple[float, float]],
    colour: tuple[int, int, int] = SOLVER_RGB,
    label: str = "",
) -> MapLayer:
    """A solved route as a map layer."""
    stops = [course.start_finish, *route.controls, course.start_finish]
    return MapLayer(stops, line, colour, label)


def image_projection(
    boundary: Sequence[tuple[float, float]], size: tuple[int, int]
):
    """A function turning (lat, lon) into (x, y) pixels on the map image.

    ``boundary`` is :func:`distances.course_boundary`'s corners (bottom-left,
    bottom-right, top-right, top-left), which are the image's own corners.
    Works in local metres, the same flat approximation the boundary was
    built in, so it lines up exactly with it.
    """
    centre_lat = sum(lat for lat, _ in boundary) / 4
    centre_lon = sum(lon for _, lon in boundary) / 4
    m_per_deg_lat = METRES_PER_DEGREE_LAT
    m_per_deg_lon = m_per_deg_lat * math.cos(math.radians(centre_lat))

    def metres(lat: float, lon: float) -> tuple[float, float]:
        return (lon - centre_lon) * m_per_deg_lon, (lat - centre_lat) * m_per_deg_lat

    origin = metres(*boundary[3])
    right = metres(*boundary[2])
    down = metres(*boundary[0])
    ux, uy = right[0] - origin[0], right[1] - origin[1]
    vx, vy = down[0] - origin[0], down[1] - origin[1]
    det = ux * vy - uy * vx
    width, height = size

    def project(lat: float, lon: float) -> tuple[float, float]:
        px, py = metres(lat, lon)
        dx, dy = px - origin[0], py - origin[1]
        u = (dx * vy - dy * vx) / det
        v = (ux * dy - uy * dx) / det
        return u * width, v * height

    return project


def _badge_direction(
    prev: tuple[float, float], here: tuple[float, float], nxt: tuple[float, float]
) -> tuple[float, float]:
    """A unit vector pointing away from both legs at a stop.

    That's the outside of the corner the route turns, so a badge placed
    along it stays clear of the line.
    """
    total = [0.0, 0.0]
    for other in (prev, nxt):
        dx, dy = other[0] - here[0], other[1] - here[1]
        length = math.hypot(dx, dy)
        if length > 0:
            total[0] -= dx / length
            total[1] -= dy / length
    length = math.hypot(*total)
    if length < 1e-6:
        # Straight through: either side is clear, so pick the left one.
        dx, dy = nxt[0] - prev[0], nxt[1] - prev[1]
        length = math.hypot(dx, dy) or 1.0
        return dy / length, -dx / length
    return total[0] / length, total[1] / length


def _legend_corner(
    size: tuple[int, int],
    box: tuple[int, int],
    inset: tuple[float, float],
    points: Sequence[tuple[float, float]],
) -> tuple[float, float]:
    """The top-left of whichever corner box covers the fewest route points."""
    width, height = size
    box_w, box_h = box
    side, edge = inset
    candidates = [
        (side, height - edge - box_h),  # bottom-left
        (width - side - box_w, height - edge - box_h),  # bottom-right
        (width - side - box_w, edge),  # top-right
        (side, edge),  # top-left
    ]

    def covered(corner: tuple[float, float]) -> int:
        x0, y0 = corner
        return sum(
            x0 <= x <= x0 + box_w and y0 <= y <= y0 + box_h for x, y in points
        )

    return min(candidates, key=covered)


def route_on_map(
    image_bytes: bytes,
    boundary: Sequence[tuple[float, float]],
    layers: Sequence[MapLayer],
    fade: float = 0.0,
):
    """The real printed map with routes and their visit order drawn on top.

    ``fade`` (0 to 1) washes the map out towards white so the routes stand
    out more. With more than one labelled route, a legend says which colour
    is which. Returns a Pillow image the same size as the original, so it
    still prints at the map's own scale.
    """
    from PIL import Image, ImageDraw, ImageFont

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if fade > 0:
        image = Image.blend(image, Image.new("RGB", image.size, "white"), fade)
    project = image_projection(boundary, image.size)
    px_per_mm = RENDER_DPI / 25.4

    draw = ImageDraw.Draw(image)
    paths = [[project(lat, lon) for lat, lon in layer.line] for layer in layers]
    drawable = [
        (path, layer) for path, layer in zip(paths, layers, strict=True)
        if len(path) >= 2
    ]
    # Every outline goes down before any colour, so a later route's outline
    # never cuts into an earlier one: a shared street shows the lower route
    # as a band with the upper one down its middle.
    for path, layer in drawable:
        draw.line(
            path, fill="white", joint="curve",
            width=round((layer.width_mm + 2 * ROUTE_OUTLINE_MM) * px_per_mm),
        )
    for path, layer in drawable:
        draw.line(
            path, fill=layer.colour, joint="curve",
            width=max(2, round(layer.width_mm * px_per_mm)),
        )

    radius = BADGE_RADIUS_MM * px_per_mm
    offset = BADGE_OFFSET_MM * px_per_mm
    font = ImageFont.load_default(size=round(radius * 1.25))
    all_points = [p for path in paths for p in path]
    for layer in layers:
        points = [project(c.latitude, c.longitude) for c in layer.stops]
        all_points += points
        for i in range(1, len(points) - 1):
            dx, dy = _badge_direction(points[i - 1], points[i], points[i + 1])
            x, y = points[i][0] + dx * offset, points[i][1] + dy * offset
            draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius],
                fill="white", outline=layer.colour,
                width=max(1, round(radius / 5)),
            )
            draw.text((x, y), str(i), fill=layer.colour, font=font, anchor="mm")

    labelled = [layer for layer in layers if layer.label]
    if len(labelled) > 1:
        _draw_legend(draw, image.size, labelled, all_points, px_per_mm, font)
    return image


def _draw_legend(draw, size, layers, points, px_per_mm, font) -> None:
    """A small key in the corner with the least route under it."""
    pad = round(2 * px_per_mm)
    swatch = round(8 * px_per_mm)
    row_h = round(6 * px_per_mm)
    text_w = max(draw.textlength(layer.label, font=font) for layer in layers)
    box = (round(pad * 3 + swatch + text_w), pad * 2 + row_h * len(layers))
    inset = (LEGEND_INSET_MM[0] * px_per_mm, LEGEND_INSET_MM[1] * px_per_mm)
    x0, y0 = _legend_corner(size, box, inset, points)
    draw.rounded_rectangle(
        [x0, y0, x0 + box[0], y0 + box[1]], radius=pad,
        fill="white", outline=(60, 60, 60), width=1,
    )
    for i, layer in enumerate(layers):
        y = y0 + pad + row_h * i + row_h / 2
        draw.line(
            [(x0 + pad, y), (x0 + pad + swatch, y)], fill=layer.colour,
            width=max(2, round(layer.width_mm * px_per_mm)),
        )
        draw.text(
            (x0 + pad * 2 + swatch, y), layer.label, fill=(30, 30, 30),
            font=font, anchor="lm",
        )


def image_png(image) -> bytes:
    """A Pillow image as PNG bytes."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def image_pdf(image, title: str = "") -> bytes:
    """A Pillow image as a one-page PDF at the map's true print size."""
    buffer = io.BytesIO()
    image.save(
        buffer, format="PDF", resolution=RENDER_DPI,
        title=title, creator="Streeto Planner",
    )
    return buffer.getvalue()
