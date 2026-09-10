"""Course parsing and the uniqid structure of OOM map IDs."""

from __future__ import annotations

import datetime as dt

import pytest

from streeto_planner.oom import Control, created_at, is_plausible, parse_course


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
