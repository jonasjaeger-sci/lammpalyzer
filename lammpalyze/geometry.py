"""Trajectory-derived distance and angle analysis."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from itertools import combinations, product
from typing import Literal

import numpy as np

from lammpalyze.analysis import LoadedSimulation
from lammpalyze.parsers import TrajectoryAtom, TrajectoryFrame, iter_lammpstrj_frames
from lammpalyze.rdf import ATOM_MASSES

GeometryKind = Literal["distance", "angle"]
DistanceSelectionKind = Literal["atom", "com_atoms", "com_molecule", "plane"]
IntramolecularKind = Literal["atoms", "molecules"]
_ATOM_ID_SEPARATOR = re.compile(r"[\s,;]+")


@dataclass(frozen=True)
class GeometrySelection:
    """One atom, center of mass, or plane used as a distance endpoint."""

    kind: DistanceSelectionKind
    ids: tuple[int, ...]

    @property
    def label(self) -> str:
        """Return a compact human-readable endpoint label."""

        joined = ",".join(str(value) for value in self.ids)
        if self.kind == "atom":
            return f"atom {joined}"
        if self.kind == "com_atoms":
            return f"COM(atoms {joined})"
        if self.kind == "com_molecule":
            return f"COM(mol {joined})"
        return f"plane({joined})"


DistancePair = tuple[GeometrySelection, GeometrySelection]


@dataclass(frozen=True)
class GeometrySeries:
    """One geometry measurement followed through one simulation trajectory."""

    simulation_index: int
    atom_ids: tuple[int, ...]
    timesteps: np.ndarray
    values: np.ndarray
    label: str | None = None


def parse_atom_ids(value: str) -> list[int]:
    """Parse one atom ID or a bracketed/separated list of atom IDs."""

    stripped = re.sub(r"[()\[\]{}]", " ", value).strip()
    if not stripped:
        raise ValueError("Enter at least one atom ID in every required field.")
    try:
        atom_ids = [int(token) for token in _ATOM_ID_SEPARATOR.split(stripped) if token]
    except ValueError as exc:
        raise ValueError(f"Atom IDs must be integers; received {value!r}.") from exc
    _validate_positive_ids(atom_ids)
    return atom_ids


def parse_atom_id_groups(value: str) -> list[tuple[int, ...]]:
    """Parse one atom-ID group or a nested list containing several groups."""

    stripped = value.strip()
    if not stripped:
        raise ValueError("Enter at least one atom ID in every required field.")
    try:
        parsed = ast.literal_eval(stripped)
    except (SyntaxError, ValueError):
        if stripped.count("[") > 1 or stripped.count("(") > 1:
            raise ValueError(f"Invalid nested atom-ID list: {value!r}.") from None
        return [tuple(parse_atom_ids(value))]

    if _is_positive_integer(parsed):
        return [(int(parsed),)]
    if not isinstance(parsed, (list, tuple)):
        raise ValueError("Atom-ID groups must be an integer, list, or nested list.")
    if not parsed:
        raise ValueError("Enter at least one atom ID in every required field.")
    if all(_is_positive_integer(item) for item in parsed):
        group = tuple(int(item) for item in parsed)
        _validate_positive_ids(group)
        return [group]
    if not all(isinstance(item, (list, tuple)) for item in parsed):
        raise ValueError("Do not mix atom IDs and nested atom-ID lists in one field.")

    groups = []
    for item in parsed:
        if not item or not all(_is_positive_integer(atom_id) for atom_id in item):
            raise ValueError("Every nested atom-ID group must contain positive integers.")
        group = tuple(int(atom_id) for atom_id in item)
        _validate_positive_ids(group)
        groups.append(group)
    return groups


def parse_distance_selections(value: str, kind: DistanceSelectionKind) -> list[GeometrySelection]:
    """Parse one GUI distance field into typed endpoint selections."""

    if kind == "atom":
        return [GeometrySelection(kind, (atom_id,)) for atom_id in parse_atom_ids(value)]
    if kind == "com_molecule":
        return [GeometrySelection(kind, (mol_id,)) for mol_id in parse_atom_ids(value)]
    if kind not in {"com_atoms", "plane"}:
        raise ValueError(f"Unsupported distance selection kind: {kind!r}.")

    groups = parse_atom_id_groups(value)
    if kind == "plane" and any(len(group) != 3 for group in groups):
        raise ValueError("Every plane must be defined by exactly three atom IDs.")
    return [GeometrySelection(kind, group) for group in groups]


def distance_pairs(
    first: list[GeometrySelection],
    second: list[GeometrySelection],
) -> list[DistancePair]:
    """Pair equal endpoint lists by position, otherwise form a Cartesian product."""

    if not first or not second:
        raise ValueError("Distance measurements require two non-empty selections.")
    raw_pairs = list(zip(first, second, strict=True)) if len(first) == len(second) else list(
        product(first, second)
    )
    pairs = []
    for first_selection, second_selection in raw_pairs:
        if first_selection.kind == second_selection.kind == "plane":
            raise ValueError("Plane-to-plane distance is not a unique scalar measurement.")
        if first_selection == second_selection and first_selection.kind == "atom":
            raise ValueError(
                f"A distance measurement must use distinct atoms; received atom "
                f"{first_selection.ids[0]} twice."
            )
        pairs.append((first_selection, second_selection))
    return pairs


def parse_intramolecular_groups(value: str, kind: IntramolecularKind) -> list[tuple[int, ...]]:
    """Parse atom groups or molecule IDs for intramolecular distances."""

    if kind == "molecules":
        return [(molecule_id,) for molecule_id in parse_atom_ids(value)]
    if kind != "atoms":
        raise ValueError(f"Unsupported intramolecular selection kind: {kind!r}.")
    groups = parse_atom_id_groups(value)
    if any(len(group) < 2 for group in groups):
        raise ValueError("Every intramolecular atom group must contain at least two atom IDs.")
    return groups


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
    """Calculate selected direct atom distances or angles for every frame."""

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
            positions = _frame_positions(frame)
            _require_atom_ids(simulation.index, frame.timestep, positions, groups)
            box_lengths = _validated_box_lengths(simulation.index, frame)
            timesteps.append(frame.timestep)
            for values, group in zip(values_by_group, groups, strict=True):
                values.append(
                    _geometry_value(kind, group, positions, box_lengths, simulation.boundary)
                )

        _require_timesteps(simulation.index, timesteps)
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


def compute_distances(
    simulations: list[LoadedSimulation],
    pairs: list[DistancePair],
    timestep_range: tuple[float, float] | None = None,
) -> list[GeometrySeries]:
    """Calculate atom, COM, and point-to-plane distances for every frame."""

    if not pairs:
        raise ValueError("Select at least one distance measurement.")
    results = []
    for simulation in simulations:
        if simulation.trajectory_path is None:
            continue
        timesteps: list[int] = []
        values_by_pair = [[] for _ in pairs]
        for frame in iter_lammpstrj_frames(simulation.trajectory_path, timestep_range):
            box_lengths = _validated_box_lengths(simulation.index, frame)
            atoms_by_id = {atom.atom_id: atom for atom in frame.atoms}
            timesteps.append(frame.timestep)
            for values, pair in zip(values_by_pair, pairs, strict=True):
                values.append(_distance_value(pair, simulation, frame, atoms_by_id, box_lengths))

        _require_timesteps(simulation.index, timesteps)
        for pair, values in zip(pairs, values_by_pair, strict=True):
            atom_ids = tuple(atom_id for selection in pair for atom_id in selection.ids)
            results.append(
                GeometrySeries(
                    simulation_index=simulation.index,
                    atom_ids=atom_ids,
                    timesteps=np.asarray(timesteps, dtype=int),
                    values=np.asarray(values, dtype=float),
                    label=f"{pair[0].label} - {pair[1].label}",
                )
            )
    if not results:
        raise ValueError("None of the selected simulations has a trajectory file.")
    return results


def compute_intramolecular_distances(
    simulations: list[LoadedSimulation],
    groups: list[tuple[int, ...]],
    kind: IntramolecularKind,
    timestep_range: tuple[float, float] | None = None,
) -> list[GeometrySeries]:
    """Calculate unique atom-pair distances within atom groups or molecules."""

    if not groups:
        raise ValueError("Select at least one intramolecular group.")
    if kind not in {"atoms", "molecules"}:
        raise ValueError(f"Unsupported intramolecular selection kind: {kind!r}.")

    results = []
    for simulation in simulations:
        if simulation.trajectory_path is None:
            continue
        timesteps: list[int] = []
        pairs: list[tuple[int, int]] | None = None
        labels: list[str] = []
        values_by_pair: list[list[float]] = []
        for frame in iter_lammpstrj_frames(simulation.trajectory_path, timestep_range):
            if pairs is None:
                pairs, labels = _intramolecular_pairs(simulation, frame, groups, kind)
                values_by_pair = [[] for _ in pairs]
            positions = _frame_positions(frame)
            _require_atom_ids(simulation.index, frame.timestep, positions, pairs)
            box_lengths = _validated_box_lengths(simulation.index, frame)
            timesteps.append(frame.timestep)
            for values, pair in zip(values_by_pair, pairs, strict=True):
                values.append(
                    _geometry_value("distance", pair, positions, box_lengths, simulation.boundary)
                )

        _require_timesteps(simulation.index, timesteps)
        if not pairs:
            raise ValueError("Intramolecular selections must contain at least one atom pair.")
        for pair, label, values in zip(pairs, labels, values_by_pair, strict=True):
            results.append(
                GeometrySeries(
                    simulation_index=simulation.index,
                    atom_ids=pair,
                    timesteps=np.asarray(timesteps, dtype=int),
                    values=np.asarray(values, dtype=float),
                    label=label,
                )
            )
    if not results:
        raise ValueError("None of the selected simulations has a trajectory file.")
    return results


def _intramolecular_pairs(
    simulation: LoadedSimulation,
    frame: TrajectoryFrame,
    groups: list[tuple[int, ...]],
    kind: IntramolecularKind,
) -> tuple[list[tuple[int, int]], list[str]]:
    """Expand intramolecular inputs into unique atom pairs using the first frame."""

    atom_groups = groups
    prefixes = [""] * len(groups)
    if kind == "molecules":
        atoms_by_molecule: dict[int, list[int]] = {}
        for atom in frame.atoms:
            molecule_value = atom.values.get("mol")
            if molecule_value is None:
                raise ValueError(
                    "Intramolecular molecule-ID mode requires a 'mol' trajectory column."
                )
            atoms_by_molecule.setdefault(int(molecule_value), []).append(atom.atom_id)
        molecule_ids = [group[0] for group in groups]
        missing = [molecule_id for molecule_id in molecule_ids if molecule_id not in atoms_by_molecule]
        if missing:
            missing_text = ", ".join(str(value) for value in missing)
            raise ValueError(
                f"Simulation {simulation.index}, timestep {frame.timestep} lacks molecule ID(s): "
                f"{missing_text}."
            )
        atom_groups = [tuple(atoms_by_molecule[molecule_id]) for molecule_id in molecule_ids]
        prefixes = [f"mol {molecule_id}: " for molecule_id in molecule_ids]

    pairs = []
    labels = []
    seen = set()
    for group, prefix in zip(atom_groups, prefixes, strict=True):
        for pair in combinations(group, 2):
            canonical = tuple(sorted(pair))
            if canonical in seen:
                continue
            seen.add(canonical)
            pairs.append(canonical)
            labels.append(f"{prefix}atom {canonical[0]} - atom {canonical[1]}")
    return pairs, labels


def _distance_value(
    pair: DistancePair,
    simulation: LoadedSimulation,
    frame: TrajectoryFrame,
    atoms_by_id: dict[int, TrajectoryAtom],
    box_lengths: np.ndarray,
) -> float:
    """Calculate one endpoint or point-to-plane distance."""

    first, second = pair
    plane = first if first.kind == "plane" else second if second.kind == "plane" else None
    if plane is not None:
        point = second if plane is first else first
        point_position = _selection_position(point, simulation, frame, atoms_by_id, box_lengths)
        plane_positions = _plane_positions(plane, simulation, frame, atoms_by_id, box_lengths)
        first_arm = plane_positions[1] - plane_positions[0]
        second_arm = plane_positions[2] - plane_positions[0]
        normal = np.cross(first_arm, second_arm)
        normal_length = np.linalg.norm(normal)
        if normal_length == 0:
            raise ValueError(f"Cannot define {plane.label}: its atoms are collinear.")
        displacement = _minimum_image(
            point_position - plane_positions[0], box_lengths, simulation.boundary
        )
        return float(abs(np.dot(displacement, normal / normal_length)))

    first_position = _selection_position(first, simulation, frame, atoms_by_id, box_lengths)
    second_position = _selection_position(second, simulation, frame, atoms_by_id, box_lengths)
    displacement = _minimum_image(
        first_position - second_position, box_lengths, simulation.boundary
    )
    return float(np.linalg.norm(displacement))


def _selection_position(
    selection: GeometrySelection,
    simulation: LoadedSimulation,
    frame: TrajectoryFrame,
    atoms_by_id: dict[int, TrajectoryAtom],
    box_lengths: np.ndarray,
) -> np.ndarray:
    """Resolve an atom or COM selection to one Cartesian position."""

    if selection.kind == "plane":
        raise ValueError("A plane cannot be used as a point endpoint.")
    if selection.kind == "atom":
        atom = _required_atoms(selection.ids, simulation, frame, atoms_by_id)[0]
        return _atom_position(atom)
    if selection.kind == "com_atoms":
        atoms = _required_atoms(selection.ids, simulation, frame, atoms_by_id)
    else:
        molecule_id = selection.ids[0]
        if any(atom.values.get("mol") is None for atom in frame.atoms):
            raise ValueError("COM molecule-ID mode requires a 'mol' trajectory column.")
        atoms = [atom for atom in frame.atoms if int(atom.values["mol"]) == molecule_id]
        if not atoms:
            raise ValueError(
                f"Simulation {simulation.index}, timestep {frame.timestep} lacks molecule ID "
                f"{molecule_id}."
            )
    return _periodic_center_of_mass(
        atoms,
        frame,
        box_lengths,
        simulation.boundary,
        simulation.type_to_element or {},
    )


def _plane_positions(
    selection: GeometrySelection,
    simulation: LoadedSimulation,
    frame: TrajectoryFrame,
    atoms_by_id: dict[int, TrajectoryAtom],
    box_lengths: np.ndarray,
) -> np.ndarray:
    """Return three plane points unwrapped around the first atom."""

    atoms = _required_atoms(selection.ids, simulation, frame, atoms_by_id)
    positions = np.array([_atom_position(atom) for atom in atoms], dtype=float)
    positions[1:] = positions[0] + _minimum_image(
        positions[1:] - positions[0], box_lengths, simulation.boundary
    )
    return positions


def _periodic_center_of_mass(
    atoms: list[TrajectoryAtom],
    frame: TrajectoryFrame,
    box_lengths: np.ndarray,
    boundary: tuple[str, str, str],
    type_to_element: dict[int, str],
) -> np.ndarray:
    """Calculate a mass-weighted COM after periodic unwrapping."""

    if len(atoms) == 1:
        return _atom_position(atoms[0])
    reference = _atom_position(atoms[0])
    positions = np.array([_atom_position(atom) for atom in atoms], dtype=float)
    unwrapped = reference + _minimum_image(positions - reference, box_lengths, boundary)
    masses = np.asarray([_atom_mass(atom, type_to_element) for atom in atoms], dtype=float)
    center = np.average(unwrapped, axis=0, weights=masses)
    periodic = np.asarray([mode == "p" for mode in boundary], dtype=bool)
    center[periodic] = frame.bounds[periodic, 0] + (
        center[periodic] - frame.bounds[periodic, 0]
    ) % box_lengths[periodic]
    return center


def _atom_mass(atom: TrajectoryAtom, type_to_element: dict[int, str]) -> float:
    """Return a trajectory-provided or element-derived atomic mass."""

    trajectory_mass = atom.values.get("mass")
    if trajectory_mass is not None and trajectory_mass > 0:
        return trajectory_mass
    element = atom.element or type_to_element.get(atom.atom_type)
    mass = ATOM_MASSES.get(element or "")
    if mass is None:
        raise ValueError(
            f"No atomic mass is known for atom {atom.atom_id}, type {atom.atom_type} "
            f"({element!r}). Use element symbols in element_list or include a positive "
            "'mass' trajectory column."
        )
    return mass


def _geometry_value(
    kind: GeometryKind,
    group: tuple[int, ...],
    positions: dict[int, np.ndarray],
    box_lengths: np.ndarray,
    boundary: tuple[str, str, str] = ("p", "p", "p"),
) -> float:
    """Return one minimum-image distance or angle in degrees."""

    if kind == "distance":
        displacement = _minimum_image(
            positions[group[0]] - positions[group[1]], box_lengths, boundary
        )
        return float(np.linalg.norm(displacement))

    first = _minimum_image(positions[group[0]] - positions[group[1]], box_lengths, boundary)
    second = _minimum_image(positions[group[2]] - positions[group[1]], box_lengths, boundary)
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator == 0:
        raise ValueError(f"Cannot calculate angle {group}: one arm has zero length.")
    cosine = np.clip(np.dot(first, second) / denominator, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _minimum_image(
    displacement: np.ndarray,
    box_lengths: np.ndarray,
    boundary: tuple[str, str, str] = ("p", "p", "p"),
) -> np.ndarray:
    """Wrap Cartesian displacements into the nearest configured periodic image."""

    result = np.array(displacement, dtype=float, copy=True)
    periodic = np.asarray([mode == "p" for mode in boundary], dtype=bool)
    result[..., periodic] -= box_lengths[periodic] * np.round(
        result[..., periodic] / box_lengths[periodic]
    )
    return result


def _required_atoms(
    atom_ids: tuple[int, ...],
    simulation: LoadedSimulation,
    frame: TrajectoryFrame,
    atoms_by_id: dict[int, TrajectoryAtom],
) -> list[TrajectoryAtom]:
    """Return selected atoms or report exactly which IDs are absent."""

    missing = [atom_id for atom_id in atom_ids if atom_id not in atoms_by_id]
    if missing:
        missing_text = ", ".join(str(atom_id) for atom_id in missing)
        raise ValueError(
            f"Simulation {simulation.index}, timestep {frame.timestep} lacks atom ID(s): "
            f"{missing_text}."
        )
    return [atoms_by_id[atom_id] for atom_id in atom_ids]


def _frame_positions(frame: TrajectoryFrame) -> dict[int, np.ndarray]:
    """Map atom IDs to Cartesian position vectors."""

    return {atom.atom_id: _atom_position(atom) for atom in frame.atoms}


def _atom_position(atom: TrajectoryAtom) -> np.ndarray:
    """Return one atom's Cartesian position vector."""

    return np.array([atom.x, atom.y, atom.z], dtype=float)


def _require_atom_ids(
    simulation_index: int,
    timestep: int,
    positions: dict[int, np.ndarray],
    groups: list[tuple[int, ...]],
) -> None:
    """Report missing atom IDs used by direct atom measurements."""

    required_ids = {atom_id for group in groups for atom_id in group}
    missing = sorted(required_ids - positions.keys())
    if missing:
        missing_text = ", ".join(str(atom_id) for atom_id in missing)
        raise ValueError(
            f"Simulation {simulation_index}, timestep {timestep} lacks atom ID(s): "
            f"{missing_text}."
        )


def _validated_box_lengths(simulation_index: int, frame: TrajectoryFrame) -> np.ndarray:
    """Return positive box lengths or report malformed bounds."""

    box_lengths = frame.bounds[:, 1] - frame.bounds[:, 0]
    if np.any(box_lengths <= 0):
        raise ValueError(
            f"Simulation {simulation_index}, timestep {frame.timestep} has invalid box bounds."
        )
    return box_lengths


def _require_timesteps(simulation_index: int, timesteps: list[int]) -> None:
    """Ensure the requested trajectory range yielded at least one frame."""

    if not timesteps:
        raise ValueError(
            f"No trajectory frames found in the selected timestep range for simulation "
            f"{simulation_index}."
        )


def _is_positive_integer(value: object) -> bool:
    """Return whether a parsed literal is a positive non-boolean integer."""

    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _validate_positive_ids(atom_ids) -> None:
    """Validate positivity and uniqueness of one ID list."""

    if any(not _is_positive_integer(atom_id) for atom_id in atom_ids):
        raise ValueError("Atom IDs must be positive integers.")
    if len(set(atom_ids)) != len(atom_ids):
        raise ValueError("Atom-ID lists must not contain duplicates.")
