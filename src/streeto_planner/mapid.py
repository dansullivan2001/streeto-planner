"""Recover an OOM map ID from a shaky reading of the map's corner.

Every OOM map prints its own ID next to the credits. Reading that one string
is far easier than detecting circles and georeferencing a photograph, so it is
the cheapest route from a phone camera to an exact course.
"""

from __future__ import annotations

import datetime as dt
import itertools

from streeto_planner.oom import is_plausible

HEX = "0123456789abcdef"

#: Glyphs an OCR pass plausibly confuses, mapped to the hex digits they may be.
CONFUSIONS = {
    "0": "0d", "o": "0", "O": "0", "D": "0d",
    "1": "17", "l": "1", "i": "1", "I": "1", "7": "17",
    "2": "2", "3": "38", "4": "4", "5": "56", "6": "6b58",
    "8": "863", "9": "9", "a": "a48", "b": "6b8",
    "c": "ce0", "d": "d0a", "e": "c6", "f": "ef",
    "g": "9", "q": "9", "s": "5", "z": "2",
}


def parse_reading(reading: str) -> list[str]:
    """Split a reading into the options available at each position.

    A valid hex character stands for itself. Anything else is a definite
    misread and expands to whatever it might have been. A bracketed run, as in
    ``a[6b]d``, marks a position the reader was unsure about and lists the
    guesses. Expanding only flagged positions is what keeps the candidate set
    small enough to be useful.
    """
    options: list[str] = []
    index = 0
    while index < len(reading):
        char = reading[index]
        if char == "[":
            end = reading.find("]", index)
            if end == -1:
                raise ValueError("unclosed '[' in reading")
            merged = "".join(
                CONFUSIONS.get(guess, guess) for guess in reading[index + 1 : end]
            )
            options.append("".join(dict.fromkeys(c for c in merged if c in HEX)))
            index = end + 1
            continue
        lowered = char.lower()
        options.append(lowered if lowered in HEX else CONFUSIONS.get(char, ""))
        index += 1
    if any(not option for option in options):
        raise ValueError(f"unmappable character in reading {reading!r}")
    return options


def candidates(reading: str) -> list[str]:
    """Every hex string a reading could really have been."""
    return ["".join(combo) for combo in itertools.product(*parse_reading(reading))]


def shortlist(
    reading: str,
    earliest: dt.date = dt.date(2012, 1, 1),
    latest: dt.date | None = None,
) -> list[str]:
    """Candidates that survive the uniqid structure check, likeliest first.

    Ordering keeps the reader's own first guess at each position ahead of the
    alternatives, which in practice puts the right ID near the top.
    """
    latest = latest or dt.date.today()
    return [c for c in candidates(reading) if is_plausible(c, earliest, latest)]
