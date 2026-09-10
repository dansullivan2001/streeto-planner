"""Recovering a map ID from an uncertain reading."""

from __future__ import annotations

import datetime as dt

import pytest

from streeto_planner.mapid import candidates, parse_reading, shortlist


def test_confident_hex_characters_stand_alone() -> None:
    assert parse_reading("698d11dd9c78c") == list("698d11dd9c78c")
    assert candidates("698d11dd9c78c") == ["698d11dd9c78c"]


def test_non_hex_characters_are_definite_misreads() -> None:
    # 'l' is not hex, so it can only ever have been a 1.
    assert candidates("698d11dd9c78l") == ["698d11dd9c781"]


def test_brackets_mark_an_uncertain_position() -> None:
    assert sorted(candidates("698d11dd[9g]c78c")) == ["698d11dd9c78c"]
    assert sorted(candidates("698d11dd9c78[cd]")) == [
        "698d11dd9c780",
        "698d11dd9c78a",
        "698d11dd9c78c",
        "698d11dd9c78d",
        "698d11dd9c78e",
    ]


def test_two_uncertain_positions_stay_a_small_set() -> None:
    found = candidates("6a9939a[6b]dd8[6b]d")
    assert len(found) == 16
    assert "6a9939a6dd86d" in found


def test_shortlist_prunes_by_creation_date() -> None:
    # Uncertain positions here only shift the ID by seconds, so every spelling
    # lands in the same month: the window must match, or nothing survives.
    reading = "6a9939a[6b]dd8[6b]d"
    assert "6a9939a6dd86d" in shortlist(
        reading, dt.date(2026, 9, 1), dt.date(2026, 9, 30)
    )
    assert shortlist(reading, dt.date(2024, 1, 1), dt.date(2024, 12, 31)) == []


def test_unmappable_characters_are_rejected() -> None:
    with pytest.raises(ValueError, match="unmappable"):
        candidates("698d11dd9c78!")


def test_unclosed_bracket_is_rejected() -> None:
    with pytest.raises(ValueError, match="unclosed"):
        candidates("698d11dd[9g")


def test_readings_of_the_wrong_length_never_survive() -> None:
    assert shortlist("698d11dd9c78", dt.date(2012, 1, 1), dt.date(2030, 1, 1)) == []
