"""Parsing MapRun's public results payload."""

from __future__ import annotations

import pytest

from streeto_planner.maprun import parse_results


@pytest.fixture
def payload() -> dict:
    return {
        "errorFlag": False,
        "statusMessage": "",
        "warningFlag": False,
        "warningMessage": "",
        "results": [
            {
                "Id": 585793,
                "Surname": "Sullivan",
                "Firstname": "Dan",
                "Gender": "M",
                "YearOfBirth": 1990,
                "TrackStartDateTimeUTC": "2026-03-23T23:15:11.000Z",
                "StartPunchTimeLocal": "05:02:30",
                "FinishPunchTimeLocal": "06:01:16",
                "TotalTimehhmmss": "58:46",
                "TotalTimeSecs": 3526,
                "Classifier": "OK",
                "ClubName": "MVOC",
                "GrossScore": 1150,
                "NetScore": 1150,
                "MapRunVersion": "MapRunF 7.10.7",
                "Distance": 14.19,
                "Pacemmss": "4:08",
                "PaceMins": 4.14,
                "punchControlIds": ["21", "51", "41", "30", "30 (Extra)"],
                "punchTimeAfterStartSecs": [59, 169, 247, 1114, 1183],
            }
        ],
    }


def test_parse_results_reads_the_core_fields(payload: dict) -> None:
    results = parse_results(payload)
    assert len(results) == 1
    result = results[0]
    assert result.name == "Dan Sullivan"
    assert result.club == "MVOC"
    assert result.classifier == "OK"
    assert result.distance_km == pytest.approx(14.19)
    assert result.gross_score == 1150
    assert result.net_score == 1150
    assert result.total_time_secs == 3526


def test_parse_results_drops_extra_punch_markers(payload: dict) -> None:
    result = parse_results(payload)[0]
    assert result.visits == ["21", "51", "41", "30"]


def test_parse_results_reports_the_error_flag(payload: dict) -> None:
    payload["errorFlag"] = True
    payload["statusMessage"] = "event not found"
    with pytest.raises(LookupError, match="event not found"):
        parse_results(payload)


def test_parse_results_requires_a_results_list(payload: dict) -> None:
    del payload["results"]
    with pytest.raises(ValueError, match="no results list"):
        parse_results(payload)


def test_total_time_hhmmss_formats_short_and_long_runs() -> None:
    from streeto_planner.maprun import Result

    short = Result("A", "", "OK", 1.0, 10, 10, 90, [])
    assert short.total_time_hhmmss == "1:30"

    long = Result("B", "", "OK", 1.0, 10, 10, 3771, [])
    assert long.total_time_hhmmss == "1:02:51"
