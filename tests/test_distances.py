"""Distance approximation and detour calibration."""

from __future__ import annotations

import math

import pytest
from conftest import METRES_PER_DEGREE_LAT, control

from streeto_planner.distances import (
    NO_ACCESS_BUFFER_M,
    NO_ACCESS_PENALTY_M,
    SnapWarning,
    _best_bearing,
    _overshoot,
    _patch_unreliable_legs,
    _point_to_segment_distance_m,
    _square_polygon_m,
    approximate_matrix,
    course_boundary,
    detour_index,
    geometry_snap_warnings,
    haversine,
    route_strays_outside_course,
)
from streeto_planner.oom import parse_course


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


def _synthetic_course(bearing: float):
    """A course whose controls sit inside a 100m x 50m sheet rotated to
    ``bearing``, with its stored ``rotation`` field negated -- mirroring the
    sign convention OOM actually uses, empirically confirmed against a real
    course (see the comment on BOUNDARY_FIT_TOLERANCE_M in distances.py).
    """
    centre_lat, centre_lon = 0.0, 0.0

    def place(local_height: float, local_width: float) -> tuple[str, str]:
        north = local_height * math.cos(bearing) - local_width * math.sin(bearing)
        east = local_height * math.sin(bearing) + local_width * math.cos(bearing)
        return (
            str(centre_lat + north / METRES_PER_DEGREE_LAT),
            str(centre_lon + east / METRES_PER_DEGREE_LAT),
        )

    start_lat, start_lon = place(0, 0)
    a_lat, a_lon = place(20, 40)
    b_lat, b_lon = place(-20, -40)
    c_lat, c_lon = place(10, -30)
    payload = {
        "title": "SYNTHETIC",
        "eventdate": "2026-01-01",
        "club": "test",
        "scale": "s1000",
        "rotation": str(-bearing),
        "centre_lat": str(centre_lat),
        "centre_lon": str(centre_lon),
        "papersize": "p1000-500",
        "paperorientation": "landscape",
        "controls": [
            {"type": "c_startfinish", "number": "", "wgs84lat": start_lat,
             "wgs84lon": start_lon, "score": "0"},
            {"type": "c_regular", "number": "10", "wgs84lat": a_lat,
             "wgs84lon": a_lon, "score": "10"},
            {"type": "c_regular", "number": "20", "wgs84lat": b_lat,
             "wgs84lon": b_lon, "score": "20"},
            {"type": "c_regular", "number": "30", "wgs84lat": c_lat,
             "wgs84lon": c_lon, "score": "30"},
        ],
    }
    return parse_course("698d11dd9c78c", payload)


def test_best_bearing_recovers_the_true_rotation_sign() -> None:
    bearing = math.radians(40)
    course = _synthetic_course(bearing)
    found_bearing, overshoot = _best_bearing(course)
    assert found_bearing == pytest.approx(bearing, abs=1e-9)
    assert overshoot < 0  # comfortably inside, not just touching the edge


def test_best_bearing_rejects_the_wrong_sign() -> None:
    bearing = math.radians(40)
    course = _synthetic_course(bearing)
    half_height, half_width = course.paper_height_m / 2, course.paper_width_m / 2
    wrong_sign_overshoot = max(
        _overshoot(
            n.latitude, n.longitude,
            course.centre_latitude, course.centre_longitude,
            course.rotation,  # the raw field is -bearing: the wrong sign here
            half_height, half_width,
        )
        for n in course.nodes
    )
    assert wrong_sign_overshoot > 0


def test_course_boundary_matches_the_known_rectangle() -> None:
    bearing = math.radians(40)
    course = _synthetic_course(bearing)
    corners = course_boundary(course)
    assert corners is not None
    assert len(corners) == 4
    half_height, half_width = course.paper_height_m / 2, course.paper_width_m / 2
    for lat, lon in corners:
        # every corner should sit exactly on the rectangle's edge
        overshoot = _overshoot(
            lat, lon, course.centre_latitude, course.centre_longitude,
            bearing, half_height, half_width,
        )
        assert overshoot == pytest.approx(0.0, abs=1e-6)


def test_course_boundary_is_unknown_without_a_papersize(course) -> None:
    assert course_boundary(course) is None


def test_route_strays_outside_course_flags_a_distant_point() -> None:
    bearing = math.radians(40)
    course = _synthetic_course(bearing)
    far_away = {
        "features": [
            {
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[0.0, 0.0], [5.0, 5.0]],
                }
            }
        ]
    }
    assert route_strays_outside_course(far_away, course) is True


def test_route_strays_outside_course_accepts_points_within_the_sheet() -> None:
    bearing = math.radians(40)
    course = _synthetic_course(bearing)
    on_sheet = {
        "features": [
            {"geometry": {"type": "LineString", "coordinates": [
                [course.centre_longitude, course.centre_latitude],
                [course.start_finish.longitude, course.start_finish.latitude],
            ]}}
        ]
    }
    assert route_strays_outside_course(on_sheet, course) is False


def test_geometry_snap_warnings_flags_a_control_landed_far_away() -> None:
    start = control("S", 0, 0)
    target = control("46", 200, 40)
    landed_far = control("_", 320, 0)  # 120m further north than requested
    geometry = {
        "features": [
            {
                "properties": {"way_points": [0, 1]},
                "geometry": {
                    "coordinates": [
                        [start.longitude, start.latitude],
                        [landed_far.longitude, landed_far.latitude],
                    ]
                },
            }
        ]
    }
    warnings = geometry_snap_warnings(geometry, [start, target])
    assert len(warnings) == 1
    assert warnings[0].number == "46"
    assert warnings[0].snapped_distance == pytest.approx(120, rel=0.02)


def test_geometry_snap_warnings_ignores_a_close_snap() -> None:
    start = control("S", 0, 0)
    target = control("46", 200, 40)
    landed_close = control("_", 210, 0)  # only 10m off, under the threshold
    geometry = {
        "features": [
            {
                "properties": {"way_points": [0, 1]},
                "geometry": {
                    "coordinates": [
                        [start.longitude, start.latitude],
                        [landed_close.longitude, landed_close.latitude],
                    ]
                },
            }
        ]
    }
    assert geometry_snap_warnings(geometry, [start, target]) == []


def test_patch_unreliable_legs_replaces_distances_touching_a_flagged_control() -> None:
    nodes = [control("S", 0, 30), control("A", 500, 30), control("B", 1200, 50)]
    # A bogus "shortcut" -- as if A snapped onto B's nearby routable point.
    matrix = [
        [0.0, 500.0, 1200.0],
        [500.0, 0.0, 5.0],
        [1200.0, 5.0, 0.0],
    ]
    _patch_unreliable_legs(nodes, matrix, [SnapWarning("A", 150.0)], detour=1.0)

    # Legs touching the flagged control (A, index 1) are replaced...
    assert matrix[1][2] == pytest.approx(haversine(nodes[1], nodes[2]))
    assert matrix[2][1] == pytest.approx(haversine(nodes[1], nodes[2]))
    assert matrix[0][1] == pytest.approx(haversine(nodes[0], nodes[1]))
    # ...but a leg between two unflagged controls is left untouched.
    assert matrix[0][2] == 1200.0


def test_patch_unreliable_legs_does_nothing_without_warnings() -> None:
    nodes = [control("S", 0, 30), control("A", 500, 30)]
    matrix = [[0.0, 40.0], [40.0, 0.0]]
    _patch_unreliable_legs(nodes, matrix, [])
    assert matrix == [[0.0, 40.0], [40.0, 0.0]]


def test_point_to_segment_distance_is_near_zero_on_the_segment() -> None:
    a = control("A", 0, 0)
    b = control("B", 1000, 0)
    midpoint = ((a.latitude + b.latitude) / 2, a.longitude)
    assert _point_to_segment_distance_m(midpoint, a, b) < 1.0


def test_point_to_segment_distance_measures_perpendicular_offset() -> None:
    a = control("A", 0, 0)
    b = control("B", 1000, 0)
    offset_lon = a.longitude + 200 / (
        METRES_PER_DEGREE_LAT * math.cos(math.radians(a.latitude))
    )
    point = ((a.latitude + b.latitude) / 2, offset_lon)
    assert _point_to_segment_distance_m(point, a, b) == pytest.approx(200, rel=0.02)


def test_point_to_segment_distance_clamps_beyond_the_endpoint() -> None:
    a = control("A", 0, 0)
    b = control("B", 1000, 0)
    before_a = (a.latitude - 500 / METRES_PER_DEGREE_LAT, a.longitude)
    assert _point_to_segment_distance_m(before_a, a, b) == pytest.approx(500, rel=0.02)


def test_approximate_matrix_penalises_a_blocked_leg() -> None:
    nodes = [control("A", 0, 0), control("B", 1000, 0)]
    midpoint = ((nodes[0].latitude + nodes[1].latitude) / 2, nodes[0].longitude)
    plain = approximate_matrix(nodes, detour=1.0)
    blocked = approximate_matrix(nodes, detour=1.0, no_access_points=[midpoint])
    assert blocked[0][1] == pytest.approx(plain[0][1] + NO_ACCESS_PENALTY_M)


def test_approximate_matrix_ignores_a_distant_no_access_point() -> None:
    nodes = [control("A", 0, 0), control("B", 1000, 0)]
    far_away = (nodes[0].latitude, nodes[0].longitude + 1.0)
    plain = approximate_matrix(nodes, detour=1.0)
    unaffected = approximate_matrix(nodes, detour=1.0, no_access_points=[far_away])
    assert unaffected[0][1] == pytest.approx(plain[0][1])
    assert NO_ACCESS_BUFFER_M > 0  # sanity: the buffer constant is meaningful


def test_square_polygon_m_is_a_closed_ring_of_the_right_size() -> None:
    lat, lon = 51.33, -0.40
    ring = _square_polygon_m((lat, lon), 10.0)
    assert len(ring) == 5
    assert ring[0] == ring[-1]
    lons = [p[0] for p in ring]
    lats = [p[1] for p in ring]
    width_m = (
        (max(lons) - min(lons)) * METRES_PER_DEGREE_LAT * math.cos(math.radians(lat))
    )
    height_m = (max(lats) - min(lats)) * METRES_PER_DEGREE_LAT
    assert width_m == pytest.approx(20.0, rel=0.02)
    assert height_m == pytest.approx(20.0, rel=0.02)
