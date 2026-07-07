"""Pair-distance and three-atom-angle analysis for LAMMPS trajectories."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import numpy as np

from lammpalyze.analysis import LoadedSimulation
from lammpalyze.parsers import iter_lammpstrj_frames

GeometryKind = Literal["distance", "angle"]
_ATOM_ID_SEPARATOR = re.compile(r"[\s,;]+")


@dataclass(frozen=True)
class GeometrySeries:
    """One atom tuple measured through one simulation trajectory."""

    simulation_index: int
    atom_ids: tuple[int, ...]
    timesteps: np.ndarray
    values: np.ndarray


def parse_atom_ids(value: str) -> list[int]:
    """Parse one atom ID or a bracketed/separated list of atom IDs."""

    stripped = re.sub(r"[()\[\]{}]", " ", value).strip()
    if not stripped:
        raise ValueError("Enter at least one atom ID in every required field.")
    try:
        atom_ids = [int(token) for token in _ATOM_ID_SEPARATOR.split(stripped) if token]
    except ValueError as exc:
        raise ValueError(f"Atom IDs must be integers; received {value!r}.") from exc
    if any(atom_id <= 0 for atom_id in atom_ids):
        raise ValueError("Atom IDs must be positive integers.")
    return atom_ids


def atom_id_groups(*columns: list[int]) -> list[tuple[int, ...]]:
    """Zip equally sized atom-ID columns into distance pairs or angle triples."""

    if len(columns) not in {2, 3}:
        raise ValueError("Geometry measurements require two or three atom-ID fields.")
    lengths = {len(column) for column in columns}
    if len(lengths) != 1:
        raise ValueError("Atom-ID lists must have the same number of elements.")
    groups = list(zip(*columns, strict=True))
    for group in groups:
        if len(set(group)) != len(group):
            raise ValueError(f"Each measurement must use distinct atoms; received {group}.")
    return groups


def compute_geometry(
    simulations: list[LoadedSimulation],
    kind: GeometryKind,
    groups: list[tuple[int, ...]],
    timestep_range: tuple[float, float] | None = None,
) -> list[GeometrySeries]:
    """Calculate selected distances or angles for every trajectory frame."""

    expected_size = 2 if kind == "distance" else 3 if kind == "angle" else 0
    if not expected_size:
        raise ValueError("Geometry kind must be 'distance' or 'angle'.")
    if not groups:
        raise ValueError("Select at least one atom pair or triple.")
    if any(len(group) != expected_size for group in groups):
        raise ValueError(f"{kind.title()} measurements require {expected_size} atom IDs.")

    results = []
    for simulation in simulations:
        if simulation.trajectory_path is None:
            continue
        timesteps: list[int] = []
        values_by_group = [[] for _ in groups]
        for frame in iter_lammpstrj_frames(simulation.trajectory_path, timestep_range):
            positions = {
                atom.atom_id: np.array([atom.x, atom.y, atom.z], dtype=float)
                for atom in frame.atoms
            }
            required_ids = {atom_id for group in groups for atom_id in group}
            missing = sorted(required_ids - positions.keys())
            if missing:
                missing_text = ", ".join(str(atom_id) for atom_id in missing)
                raise ValueError(
                    f"Simulation {simulation.index}, timestep {frame.timestep} lacks atom ID(s): "
                    f"{missing_text}."
                )
            box_lengths = frame.bounds[:, 1] - frame.bounds[:, 0]
            if np.any(box_lengths <= 0):
                raise ValueError(
                    f"Simulation {simulation.index}, timestep {frame.timestep} has invalid box bounds."
                )
            timesteps.append(frame.timestep)
            for values, group in zip(values_by_group, groups, strict=True):
                values.append(_geometry_value(kind, group, positions, box_lengths))

        if not timesteps:
            raise ValueError(
                f"No trajectory frames found in the selected timestep range for simulation "
                f"{simulation.index}."
            )
        for group, values in zip(groups, values_by_group, strict=True):
            results.append(
                GeometrySeries(
                    simulation_index=simulation.index,
                    atom_ids=group,
                    timesteps=np.asarray(timesteps, dtype=int),
                    values=np.asarray(values, dtype=float),
                )
            )
    if not results:
        raise ValueError("None of the selected simulations has a trajectory file.")
    return results


def _geometry_value(
    kind: GeometryKind,
    group: tuple[int, ...],
    positions: dict[int, np.ndarray],
    box_lengths: np.ndarray,
) -> float:
    """Return one minimum-image distance or angle in degrees."""

    if kind == "distance":
        displacement = _minimum_image(positions[group[0]] - positions[group[1]], box_lengths)
        return float(np.linalg.norm(displacement))

    first = _minimum_image(positions[group[0]] - positions[group[1]], box_lengths)
    second = _minimum_image(positions[group[2]] - positions[group[1]], box_lengths)
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator == 0:
        raise ValueError(f"Cannot calculate angle {group}: one arm has zero length.")
    cosine = np.clip(np.dot(first, second) / denominator, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _minimum_image(displacement: np.ndarray, box_lengths: np.ndarray) -> np.ndarray:
    """Wrap one Cartesian displacement into the nearest periodic image."""

    return displacement - box_lengths * np.round(displacement / box_lengths)
