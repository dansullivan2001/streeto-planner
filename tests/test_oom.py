"""Course parsing and the uniqid structure of OOM map IDs."""

from __future__ import annotations

import datetime as dt

import pytest

from streeto_planner.oom import (
    Control,
    created_at,
    is_plausible,
    parse_course,
    parse_render_url,
)


@pytest.mark.parametrize(
    ("map_id", "expected"),
    [
        ("6a9939a6dd86d", "2026-09-03"),
        ("65df3394229ce", "2024-02-28"),
        ("698d11dd9c78c", "2026-02-11"),
    ],
)
def test_created_at_decodes_real_map_ids(map_id: str, expected: str) -> None:
    assert created_at(map_id).strftime("%Y-%m-%d") == expected


def test_created_at_rejects_malformed_ids() -> None:
    for bad in ["", "abc", "6a9939a6dd86", "6a9939a6dd86dd", "6A9939A6DD86D", "zzz"]:
        with pytest.raises(ValueError):
            created_at(bad)


def test_is_plausible_accepts_a_real_id() -> None:
    assert is_plausible("698d11dd9c78c", dt.date(2026, 1, 1), dt.date(2026, 12, 31))


def test_is_plausible_rejects_dates_outside_the_window() -> None:
    assert not is_plausible("698d11dd9c78c", dt.date(2020, 1, 1), dt.date(2020, 12, 31))


def test_is_plausible_rejects_impossible_microseconds() -> None:
    # fffff is 1048575, above the million-microsecond ceiling uniqid can emit.
    assert not is_plausible("698d11ddfffff", dt.date(2012, 1, 1), dt.date(2030, 1, 1))


def test_parse_course_keeps_only_scored_controls(course) -> None:
    assert [c.number for c in course.controls] == ["10", "31", "52"]
    assert course.total_points == 90
    assert course.start_finish.points == 0


def test_nodes_put_the_start_first(course) -> None:
    assert course.nodes[0] is course.start_finish
    assert len(course.nodes) == 4


def test_parse_course_reads_geometry(course) -> None:
    assert course.scale == 10000
    assert course.rotation == pytest.approx(-0.3929)
    assert course.title == "TEST STREET O"


def test_first_digit_scoring_convention(course) -> None:
    assert course.scoring_anomalies() == []
    assert Control("47", 0.0, 0.0, 40).implied_points == 40


def test_scoring_anomalies_flags_a_mismatch() -> None:
    odd = Control("31", 51.0, -0.4, 50)
    assert odd.implied_points == 30
    assert odd.points != odd.implied_points


def test_parse_course_requires_exactly_one_start(payload: dict) -> None:
    payload["controls"] = [
        c for c in payload["controls"] if c["type"] != "c_startfinish"
    ]
    with pytest.raises(ValueError, match="start/finish"):
        parse_course("698d11dd9c78c", payload)


def test_parse_course_requires_scored_controls(payload: dict) -> None:
    payload["controls"] = [c for c in payload["controls"] if c["type"] != "c_regular"]
    with pytest.raises(ValueError, match="no scored controls"):
        parse_course("698d11dd9c78c", payload)


def test_paper_dimensions_default_to_unknown(course) -> None:
    # The shared fixture payload has no papersize, matching a real OOM
    # response we can't parse for this reason.
    assert course.paper_width_m == 0.0
    assert course.paper_height_m == 0.0


def test_paper_dimensions_from_papersize(payload: dict) -> None:
    payload["papersize"] = "p2970-2100"
    payload["paperorientation"] = "landscape"
    payload["scale"] = "s10000"
    course = parse_course("698d11dd9c78c", payload)
    assert course.paper_width_m == pytest.approx(2970.0)
    assert course.paper_height_m == pytest.approx(2100.0)


def test_paper_dimensions_swap_for_portrait(payload: dict) -> None:
    payload["papersize"] = "p2970-2100"
    payload["paperorientation"] = "portrait"
    payload["scale"] = "s10000"
    course = parse_course("698d11dd9c78c", payload)
    assert course.paper_width_m == pytest.approx(2100.0)
    assert course.paper_height_m == pytest.approx(2970.0)


def test_parse_render_url_needs_a_paper_size(payload: dict) -> None:
    with pytest.raises(ValueError, match="paper size"):
        parse_render_url("698d11dd9c78c", payload)


def test_parse_render_url_builds_the_expected_query(payload: dict) -> None:
    payload["papersize"] = "p2970-2100"
    payload["paperorientation"] = "landscape"
    payload["scale"] = "s10000"
    payload["style"] = "streeto-LIDAR-5"
    url = parse_render_url("698d11dd9c78c", payload)

    assert url.startswith("https://oomap.dna-software.co.uk/render/jpg/?")
    assert "style=streeto-LIDAR-5" in url
    # A4 landscape: 297mm x 210mm, in physical metres (not ground metres).
    assert "paper=0.297,0.210" in url
    assert "scale=10000" in url
    assert "id=698d11dd9c78c" in url
    assert "rotation=-0.3929" in url
    # The three scored controls' numbers appear in the controls list.
    assert "controls=" in url
    for number in ("10", "31", "52"):
        assert f"{number}," in url.split("controls=")[1].split("|")[0]


def test_parse_render_url_portrait_swaps_paper_dimensions(payload: dict) -> None:
    payload["papersize"] = "p2970-2100"
    payload["paperorientation"] = "portrait"
    payload["scale"] = "s10000"
    url = parse_render_url("698d11dd9c78c", payload)
    assert "paper=0.210,0.297" in url
