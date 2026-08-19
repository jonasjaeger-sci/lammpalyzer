"""Generate atom-ID lists from the first frame of a LAMMPS trajectory."""

from __future__ import annotations

import re
from collections.abc import Iterable

from lammpalyze.parsers import TrajectoryFrame

_SELECTION_SEPARATOR = re.compile(r"[\s,;]+")


def parse_atomic_index_selection(value: str, *, allow_zero: bool = False) -> list[int]:
    """Parse individual values and inclusive ``start*end`` ranges."""

    tokens = [token for token in _SELECTION_SEPARATOR.split(value.strip()) if token]
    if not tokens:
        raise ValueError("Enter at least one atom type or molecule ID.")

    minimum = 0 if allow_zero else 1
    selected: set[int] = set()
    for token in tokens:
        parts = token.split("*")
        if len(parts) > 2 or any(not part for part in parts):
            raise ValueError(f"Invalid selection {token!r}; use values such as 1,3,4*7.")
        try:
            bounds = [int(part) for part in parts]
        except ValueError as exc:
            raise ValueError(f"Selection values must be integers; received {token!r}.") from exc
        if any(bound < minimum for bound in bounds):
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"Selection values must be {qualifier} integers.")
        start = bounds[0]
        end = bounds[-1]
        if start > end:
            raise ValueError(f"Range start must not exceed range end; received {token!r}.")
        selected.update(range(start, end + 1))
    return sorted(selected)


def parse_repeat_count(value: str) -> int:
    """Return a positive per-ID repetition count."""

    try:
        repeat = int(value.strip())
    except ValueError as exc:
        raise ValueError("Repeat count must be a positive integer.") from exc
    if repeat < 1:
        raise ValueError("Repeat count must be a positive integer.")
    return repeat


def atomic_ids_from_frame(
    frame: TrajectoryFrame,
    selection_mode: str,
    selected_values: Iterable[int],
    *,
    repeat: int = 1,
) -> list[int]:
    """Select sorted atom IDs by atom type or trajectory-provided molecule ID."""

    if selection_mode not in {"atom_type", "molecule_id"}:
        raise ValueError("Selection mode must be 'atom_type' or 'molecule_id'.")
    if repeat < 1:
        raise ValueError("Repeat count must be a positive integer.")

    selected = set(selected_values)
    if not selected:
        raise ValueError("Select at least one atom type or molecule ID.")
    if selection_mode == "molecule_id" and any("mol" not in atom.values for atom in frame.atoms):
        raise ValueError("The first trajectory frame does not provide a mol column.")

    atom_ids = []
    for atom in frame.atoms:
        value = atom.atom_type
        if selection_mode == "molecule_id":
            molecule_value = atom.values["mol"]
            if not float(molecule_value).is_integer():
                raise ValueError(
                    f"Atom {atom.atom_id} has a non-integer mol value: {molecule_value}."
                )
            value = int(molecule_value)
        if value in selected:
            atom_ids.append(atom.atom_id)

    atom_ids.sort()
    if not atom_ids:
        label = "atom types" if selection_mode == "atom_type" else "molecule IDs"
        values = ", ".join(str(item) for item in sorted(selected))
        raise ValueError(f"No atoms in the first trajectory frame match {label}: {values}.")
    return [atom_id for atom_id in atom_ids for _ in range(repeat)]


def format_atomic_id_list(atom_ids: Iterable[int]) -> str:
    """Format atom IDs for direct use in GUI atom-ID entry fields."""

    return f"[{', '.join(str(atom_id) for atom_id in atom_ids)}]"
