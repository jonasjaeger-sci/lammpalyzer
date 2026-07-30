"""Radial distribution function helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from lammpalyze.analysis import LoadedSimulation
from lammpalyze.parsers import TrajectoryFrame, iter_lammpstrj_frames


ATOM_MASSES = {
    "H": 1.008,
    "He": 4.0026,
    "Li": 6.94,
    "Be": 9.0122,
    "B": 10.81,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "Ne": 20.180,
    "Na": 22.990,
    "Mg": 24.305,
    "Al": 26.982,
    "Si": 28.085,
    "P": 30.974,
    "S": 32.06,
    "Cl": 35.45,
    "Ar": 39.948,
    "K": 39.098,
    "Ca": 40.078,
    "Sc": 44.956,
    "Ti": 47.867,
    "V": 50.942,
    "Cr": 51.996,
    "Mn": 54.938,
    "Fe": 55.845,
    "Co": 58.933,
    "Ni": 58.693,
    "Cu": 63.546,
    "Zn": 65.38,
    "Ga": 69.723,
    "Ge": 72.630,
    "As": 74.922,
    "Se": 78.971,
    "Br": 79.904,
    "Kr": 83.798,
    "Rb": 85.468,
    "Sr": 87.62,
    "Y": 88.906,
    "Zr": 91.224,
    "Nb": 92.906,
    "Mo": 95.95,
    "Ru": 101.07,
    "Rh": 102.91,
    "Pd": 106.42,
    "Ag": 107.87,
    "Cd": 112.41,
    "In": 114.82,
    "Sn": 118.71,
    "Sb": 121.76,
    "Te": 127.60,
    "I": 126.90,
    "Xe": 131.29,
    "Cs": 132.91,
    "Ba": 137.33,
    "La": 138.91,
    "Ce": 140.12,
    "Pr": 140.91,
    "Nd": 144.24,
    "Sm": 150.36,
    "Eu": 151.96,
    "Gd": 157.25,
    "Tb": 158.93,
    "Dy": 162.50,
    "Ho": 164.93,
    "Er": 167.26,
    "Tm": 168.93,
    "Yb": 173.05,
    "Lu": 174.97,
    "Hf": 178.49,
    "Ta": 180.95,
    "W": 183.84,
    "Re": 186.21,
    "Os": 190.23,
    "Ir": 192.22,
    "Pt": 195.08,
    "Au": 196.97,
    "Hg": 200.59,
    "Tl": 204.38,
    "Pb": 207.2,
    "Bi": 208.98,
    "Th": 232.04,
    "Pa": 231.04,
    "U": 238.03,
}


@dataclass(frozen=True)
class RDFResult:
    """A time-averaged RDF curve for one simulation."""

    simulation_index: int
    r: np.ndarray
    g_r: np.ndarray
    timesteps: list[int]


def compute_rdf(
    simulations: list[LoadedSimulation],
    element_a: str,
    element_b: str,
    timestep_range: tuple[int, int],
    bin_width: float,
    *,
    timesteps_by_simulation: Mapping[int, Iterable[int]] | None = None,
    sampling_frequency: int = 1,
    atom_types_a: Iterable[int] | None = None,
    atom_types_b: Iterable[int] | None = None,
    molecule_ids_a: Iterable[int] | None = None,
    molecule_ids_b: Iterable[int] | None = None,
) -> list[RDFResult]:
    """Compute time-averaged RDF curves for selected simulations.

    By default, ``element_a`` and ``element_b`` select atoms through the
    simulation's atom-type-to-element mapping. Passing both ``atom_types_*``
    arguments selects the explicit atom types instead. Passing both
    ``molecule_ids_*`` arguments selects molecular centers of mass.
    """

    if bin_width <= 0:
        raise ValueError("Bin width must be greater than zero.")
    sampling_frequency = _validated_sampling_frequency(sampling_frequency)
    selection_mode, selection_a, selection_b = _rdf_selection(
        atom_types_a,
        atom_types_b,
        molecule_ids_a,
        molecule_ids_b,
    )

    results = []
    for simulation in simulations:
        if simulation.trajectory_path is None:
            continue
        if simulation.type_to_element is None:
            raise ValueError(f"Simulation {simulation.index} has no atom-type element mapping.")

        selected_timesteps = _selected_timesteps(timesteps_by_simulation, simulation.index)
        r_max = _selected_r_max(
            simulation,
            timestep_range,
            selected_timesteps,
            sampling_frequency,
        )
        if r_max is None:
            continue

        if r_max <= 0:
            raise ValueError(f"Simulation {simulation.index} has invalid trajectory box dimensions.")

        bins = np.arange(0.0, r_max + bin_width, bin_width)
        if len(bins) < 2:
            bins = np.array([0.0, r_max])
        bin_centers = (bins[:-1] + bins[1:]) / 2.0

        frame_curves = []
        timesteps = []
        for frame in _iter_selected_frames(
            simulation,
            timestep_range,
            selected_timesteps,
            sampling_frequency,
        ):
            curve = _frame_rdf(
                frame,
                simulation.type_to_element,
                element_a,
                element_b,
                bins,
                selection_mode=selection_mode,
                selection_a=selection_a,
                selection_b=selection_b,
            )
            if curve is None:
                continue
            frame_curves.append(curve)
            timesteps.append(frame.timestep)

        if not frame_curves:
            particle_kind = "molecule" if selection_mode == "molecule" else "atom"
            raise ValueError(
                f"Simulation {simulation.index} has no {element_a}-{element_b} "
                f"{particle_kind} pairs "
                "in the selected timestep range."
            )

        results.append(
            RDFResult(
                simulation_index=simulation.index,
                r=bin_centers,
                g_r=np.mean(np.vstack(frame_curves), axis=0),
                timesteps=timesteps,
            )
        )

    if not results:
        start, end = sorted(timestep_range)
        raise ValueError(f"No trajectory frames found between timesteps {start} and {end}.")
    return results


def _validated_sampling_frequency(value: int) -> int:
    """Return a positive integer RDF sampling interval."""

    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("Sampling frequency must be a positive integer.")
    if value <= 0:
        raise ValueError("Sampling frequency must be a positive integer.")
    return int(value)


def parse_rdf_ids(value: str) -> list[int]:
    """Parse positive IDs with comma/space separators and inclusive ``*`` ranges."""

    tokens = value.replace(",", " ").split()
    identifiers = set()
    for token in tokens:
        parts = token.split("*")
        if len(parts) > 2 or any(not part.isdigit() for part in parts):
            raise ValueError(
                f"Invalid ID or range {token!r}; use values such as 1*11,15,17."
            )
        start = int(parts[0])
        end = int(parts[-1])
        if start <= 0 or end <= 0:
            raise ValueError("RDF atom types and molecule IDs must be positive integers.")
        if end < start:
            raise ValueError(f"ID range ends before it starts: {token!r}.")
        identifiers.update(range(start, end + 1))
    if not identifiers:
        raise ValueError("Enter at least one RDF atom type or molecule ID.")
    return sorted(identifiers)


def _rdf_selection(
    atom_types_a: Iterable[int] | None,
    atom_types_b: Iterable[int] | None,
    molecule_ids_a: Iterable[int] | None,
    molecule_ids_b: Iterable[int] | None,
) -> tuple[str, set[int] | None, set[int] | None]:
    """Validate and normalize the optional explicit RDF particle selection."""

    atom_selection = atom_types_a is not None or atom_types_b is not None
    molecule_selection = molecule_ids_a is not None or molecule_ids_b is not None
    if atom_selection and molecule_selection:
        raise ValueError("Select either atom types or molecule IDs for an RDF, not both.")
    if atom_selection:
        if atom_types_a is None or atom_types_b is None:
            raise ValueError("Provide atom types for both RDF selections.")
        return "atom_type", _positive_id_set(atom_types_a, "atom types"), _positive_id_set(
            atom_types_b, "atom types"
        )
    if molecule_selection:
        if molecule_ids_a is None or molecule_ids_b is None:
            raise ValueError("Provide molecule IDs for both RDF selections.")
        return (
            "molecule",
            _positive_id_set(molecule_ids_a, "molecule IDs"),
            _positive_id_set(molecule_ids_b, "molecule IDs"),
        )
    return "element", None, None


def _positive_id_set(values: Iterable[int], label: str) -> set[int]:
    """Return a validated, non-empty set of positive integer IDs."""

    identifiers = {int(value) for value in values}
    if not identifiers:
        raise ValueError(f"Provide at least one set of {label}.")
    if min(identifiers) <= 0:
        raise ValueError(f"RDF {label} must be positive integers.")
    return identifiers


def _selected_timesteps(
    timesteps_by_simulation: Mapping[int, Iterable[int]] | None,
    simulation_index: int,
) -> list[int] | None:
    """Return exact timesteps requested for one simulation, if any."""

    if timesteps_by_simulation is None:
        return None
    return sorted({int(timestep) for timestep in timesteps_by_simulation.get(simulation_index, [])})


def _selected_r_max(
    simulation: LoadedSimulation,
    timestep_range: tuple[int, int],
    selected_timesteps: list[int] | None,
    sampling_frequency: int,
) -> float | None:
    """Find the largest valid RDF radius without retaining range frames."""

    r_max: float | None = None
    for frame in _iter_selected_frames(
        simulation,
        timestep_range,
        selected_timesteps,
        sampling_frequency,
    ):
        frame_r_max = float(np.min(_box_lengths(frame))) / 2.0
        r_max = frame_r_max if r_max is None else min(r_max, frame_r_max)
    return r_max


def _iter_selected_frames(
    simulation: LoadedSimulation,
    timestep_range: tuple[int, int],
    selected_timesteps: list[int] | None,
    sampling_frequency: int,
) -> Iterable[TrajectoryFrame]:
    """Yield exact timesteps or regularly sampled frames in a numeric range."""

    if selected_timesteps is not None:
        for timestep in selected_timesteps:
            yield simulation.read_trajectory_frame(timestep)
        return
    if simulation.trajectory_path is None:
        return
    start, _end = sorted(timestep_range)
    for frame in iter_lammpstrj_frames(simulation.trajectory_path, timestep_range):
        if (frame.timestep - start) % sampling_frequency == 0:
            yield frame


def _frame_rdf(
    frame: TrajectoryFrame,
    type_to_element: dict[int, str],
    element_a: str,
    element_b: str,
    bins: np.ndarray,
    *,
    selection_mode: str = "element",
    selection_a: set[int] | None = None,
    selection_b: set[int] | None = None,
) -> np.ndarray | None:
    """Compute the RDF contribution for one trajectory frame."""

    box_lengths = _box_lengths(frame)
    volume = float(np.prod(box_lengths))
    particle_ids_a, positions_a = _selected_particle_positions(
        frame,
        type_to_element,
        element_a,
        selection_mode,
        selection_a,
    )
    particle_ids_b, positions_b = _selected_particle_positions(
        frame,
        type_to_element,
        element_b,
        selection_mode,
        selection_b,
    )
    n_a = len(positions_a)
    n_b = len(positions_b)

    if n_a == 0 or n_b == 0:
        return None

    distances = _minimum_image_distances(positions_a, positions_b, box_lengths)
    eligible_pairs = particle_ids_a[:, np.newaxis] != particle_ids_b[np.newaxis, :]
    pair_count = int(np.count_nonzero(eligible_pairs))
    if pair_count == 0:
        return None
    distances = distances[eligible_pairs]
    bulk_density = pair_count / (n_a * volume)

    counts, _ = np.histogram(distances, bins=bins)
    bin_widths = np.diff(bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    shells = 4.0 * np.pi * bin_centers**2 * bin_widths
    local_density = counts / (n_a * shells)
    return local_density / bulk_density


def _selected_particle_positions(
    frame: TrajectoryFrame,
    type_to_element: dict[int, str],
    element: str,
    selection_mode: str,
    selection: set[int] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return stable particle IDs and positions for one RDF selection."""

    if selection_mode == "molecule":
        return _molecule_centers_of_mass(frame, type_to_element, selection or set())
    atoms = (
        [
            atom
            for atom in frame.atoms
            if atom.atom_type in (selection or set())
        ]
        if selection_mode == "atom_type"
        else [
            atom
            for atom in frame.atoms
            if type_to_element.get(atom.atom_type) == element
        ]
    )
    return (
        np.array([atom.atom_id for atom in atoms], dtype=int),
        np.array([[atom.x, atom.y, atom.z] for atom in atoms], dtype=float).reshape((-1, 3)),
    )


def _molecule_centers_of_mass(
    frame: TrajectoryFrame,
    type_to_element: dict[int, str],
    molecule_ids: set[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return selected molecule IDs and periodic centers of mass."""

    atoms_by_molecule = {}
    for atom in frame.atoms:
        molecule_value = atom.values.get("mol")
        if molecule_value is None:
            raise ValueError(
                "Molecule RDF requires a 'mol' column in every selected trajectory frame."
            )
        molecule_id = int(molecule_value)
        if molecule_id in molecule_ids:
            atoms_by_molecule.setdefault(molecule_id, []).append(atom)

    selected_ids = sorted(atoms_by_molecule)
    centers = [
        _periodic_center_of_mass(atoms_by_molecule[molecule_id], frame, type_to_element)
        for molecule_id in selected_ids
    ]
    return np.array(selected_ids, dtype=int), np.array(centers, dtype=float).reshape((-1, 3))


def _periodic_center_of_mass(
    atoms,
    frame: TrajectoryFrame,
    type_to_element: dict[int, str],
) -> np.ndarray:
    """Calculate a molecule COM after unwrapping atoms around a reference atom."""

    box_lengths = _box_lengths(frame)
    reference = np.array([atoms[0].x, atoms[0].y, atoms[0].z], dtype=float)
    positions = np.array([[atom.x, atom.y, atom.z] for atom in atoms], dtype=float)
    displacement = positions - reference
    displacement -= box_lengths * np.round(displacement / box_lengths)
    unwrapped = reference + displacement
    masses = np.array(
        [_atom_mass(atom, type_to_element) for atom in atoms],
        dtype=float,
    )
    center = np.average(unwrapped, axis=0, weights=masses)
    return frame.bounds[:, 0] + (center - frame.bounds[:, 0]) % box_lengths


def _atom_mass(atom, type_to_element: dict[int, str]) -> float:
    """Return a trajectory-provided or element-derived atomic mass."""

    trajectory_mass = atom.values.get("mass")
    if trajectory_mass is not None and trajectory_mass > 0:
        return trajectory_mass
    element = type_to_element.get(atom.atom_type)
    mass = ATOM_MASSES.get(element or "")
    if mass is None:
        raise ValueError(
            f"No atomic mass is known for atom type {atom.atom_type} ({element!r}). "
            "Use element symbols in element_list or include a positive 'mass' trajectory column."
        )
    return mass


def _positions_for_element(
    frame: TrajectoryFrame,
    type_to_element: dict[int, str],
    element: str,
) -> np.ndarray:
    """Return Cartesian positions for atoms matching ``element``."""

    return np.array(
        [
            [atom.x, atom.y, atom.z]
            for atom in frame.atoms
            if type_to_element.get(atom.atom_type) == element
        ],
        dtype=float,
    )


def _minimum_image_distances(
    positions_a: np.ndarray,
    positions_b: np.ndarray,
    box_lengths: np.ndarray,
) -> np.ndarray:
    """Return pair distances under periodic minimum-image wrapping."""

    displacement = positions_a[:, np.newaxis, :] - positions_b[np.newaxis, :, :]
    displacement -= box_lengths * np.round(displacement / box_lengths)
    return np.linalg.norm(displacement, axis=2)


def _box_lengths(frame: TrajectoryFrame) -> np.ndarray:
    """Return x, y, and z box lengths for a trajectory frame."""

    return frame.bounds[:, 1] - frame.bounds[:, 0]
