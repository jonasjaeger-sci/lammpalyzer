"""Tests for atom-ID list generation from trajectory metadata."""

import numpy as np
import pytest

from lammpalyze.atomic_indices import (
    atomic_ids_from_frame,
    format_atomic_id_list,
    parse_atomic_index_selection,
    parse_repeat_count,
)
from lammpalyze.parsers import TrajectoryAtom, TrajectoryFrame


def test_parse_atomic_index_selection_expands_inclusive_star_ranges():
    """Expand, deduplicate, and sort individual values and star ranges."""

    assert parse_atomic_index_selection("7, 1,3,4*7") == [1, 3, 4, 5, 6, 7]


def test_parse_atomic_index_selection_allows_zero_for_molecule_ids():
    """Permit the LAMMPS zero molecule tag only when explicitly requested."""

    assert parse_atomic_index_selection("0,2*4", allow_zero=True) == [0, 2, 3, 4]
    with pytest.raises(ValueError, match="positive"):
        parse_atomic_index_selection("0,2*4")


@pytest.mark.parametrize("value", ["4*2", "1**3", "1*a", ""])
def test_parse_atomic_index_selection_rejects_invalid_values(value: str):
    """Reject reversed, malformed, non-integer, or empty selector input."""

    with pytest.raises(ValueError):
        parse_atomic_index_selection(value)


def test_atomic_ids_from_frame_selects_types_sorts_and_repeats():
    """Repeat each sorted matching atom ID before moving to the next ID."""

    frame = _frame()

    atom_ids = atomic_ids_from_frame(frame, "atom_type", [3, 4], repeat=3)

    assert atom_ids == [1, 1, 1, 3, 3, 3, 5, 5, 5]
    assert format_atomic_id_list(atom_ids) == "[1, 1, 1, 3, 3, 3, 5, 5, 5]"


def test_atomic_ids_from_frame_uses_only_trajectory_mol_values():
    """Select molecule membership from atom-table mol values."""

    assert atomic_ids_from_frame(_frame(), "molecule_id", [7]) == [1, 5]


def test_atomic_ids_from_frame_rejects_missing_mol_column():
    """Do not infer molecule membership when the trajectory omits mol."""

    frame = TrajectoryFrame(
        timestep=0,
        bounds=np.zeros((3, 2)),
        atoms=[TrajectoryAtom(1, 3, 0.0, 0.0, 0.0)],
    )

    with pytest.raises(ValueError, match="does not provide a mol column"):
        atomic_ids_from_frame(frame, "molecule_id", [1])


@pytest.mark.parametrize(("value", "expected"), [("1", 1), (" 3 ", 3)])
def test_parse_repeat_count_accepts_positive_integers(value: str, expected: int):
    """Validate the optional per-atom repetition count."""

    assert parse_repeat_count(value) == expected


@pytest.mark.parametrize("value", ["", "0", "-1", "2.5"])
def test_parse_repeat_count_rejects_non_positive_integers(value: str):
    """Reject repetition counts that cannot produce a meaningful list."""

    with pytest.raises(ValueError, match="positive integer"):
        parse_repeat_count(value)


def _frame() -> TrajectoryFrame:
    """Return an intentionally unsorted first-frame atom fixture."""

    return TrajectoryFrame(
        timestep=100,
        bounds=np.zeros((3, 2)),
        atoms=[
            TrajectoryAtom(5, 4, 0.0, 0.0, 0.0, values={"mol": 7.0}),
            TrajectoryAtom(3, 3, 0.0, 0.0, 0.0, values={"mol": 8.0}),
            TrajectoryAtom(1, 3, 0.0, 0.0, 0.0, values={"mol": 7.0}),
            TrajectoryAtom(2, 2, 0.0, 0.0, 0.0, values={"mol": 8.0}),
        ],
    )
