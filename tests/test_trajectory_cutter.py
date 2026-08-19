"""Tests for the standalone trajectory-cutter GUI helpers."""

import pytest

from lammpalyze.chop import parse_trajectory_cut_range


def test_parse_trajectory_cut_range_accepts_inclusive_equal_endpoints():
    """A one-timestep interval is a valid inclusive export range."""

    assert parse_trajectory_cut_range(" 12300 ", "12300") == (12300, 12300)


def test_parse_trajectory_cut_range_accepts_distinct_endpoints():
    """Parse the requested start and end timestep values as integers."""

    assert parse_trajectory_cut_range("12300", "15400") == (12300, 15400)


@pytest.mark.parametrize(
    ("start_value", "end_value", "message"),
    [
        ("", "15400", "Enter both"),
        ("12300", "", "Enter both"),
        ("12.5", "15400", "whole numbers"),
        ("15400", "12300", "less than or equal"),
    ],
)
def test_parse_trajectory_cut_range_rejects_invalid_entries(
    start_value: str,
    end_value: str,
    message: str,
):
    """Give focused validation errors for malformed or reversed ranges."""

    with pytest.raises(ValueError, match=message):
        parse_trajectory_cut_range(start_value, end_value)
