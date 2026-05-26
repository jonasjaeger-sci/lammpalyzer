"""Radial distribution function helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lammpalyze.analysis import LoadedSimulation
from lammpalyze.parsers import TrajectoryFrame, iter_lammpstrj_frames


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
) -> list[RDFResult]:
    """Compute time-averaged RDF curves for selected simulations."""

    if bin_width <= 0:
        raise ValueError("Bin width must be greater than zero.")

    results = []
    for simulation in simulations:
        if simulation.trajectory_path is None:
            continue
        if simulation.type_to_element is None:
            raise ValueError(f"Simulation {simulation.index} has no atom-type element mapping.")

        frames = list(iter_lammpstrj_frames(simulation.trajectory_path, timestep_range))
        if not frames:
            continue

        r_max = min(float(np.min(_box_lengths(frame))) / 2.0 for frame in frames)
        if r_max <= 0:
            raise ValueError(f"Simulation {simulation.index} has invalid trajectory box dimensions.")

        bins = np.arange(0.0, r_max + bin_width, bin_width)
        if len(bins) < 2:
            bins = np.array([0.0, r_max])
        bin_centers = (bins[:-1] + bins[1:]) / 2.0

        frame_curves = []
        timesteps = []
        for frame in frames:
            curve = _frame_rdf(frame, simulation.type_to_element, element_a, element_b, bins)
            if curve is None:
                continue
            frame_curves.append(curve)
            timesteps.append(frame.timestep)

        if not frame_curves:
            raise ValueError(
                f"Simulation {simulation.index} has no {element_a}-{element_b} atom pairs "
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


def _frame_rdf(
    frame: TrajectoryFrame,
    type_to_element: dict[int, str],
    element_a: str,
    element_b: str,
    bins: np.ndarray,
) -> np.ndarray | None:
    """Compute the RDF contribution for one trajectory frame."""

    box_lengths = _box_lengths(frame)
    volume = float(np.prod(box_lengths))
    positions_a = _positions_for_element(frame, type_to_element, element_a)
    positions_b = _positions_for_element(frame, type_to_element, element_b)
    n_a = len(positions_a)
    n_b = len(positions_b)

    if n_a == 0 or n_b == 0:
        return None
    same_element = element_a == element_b
    if same_element and n_a < 2:
        return None

    distances = _minimum_image_distances(positions_a, positions_b, box_lengths)
    if same_element:
        distances = distances[np.triu_indices(n_a, k=1)]
        bulk_density = (n_a - 1) * 0.5 / volume
    else:
        distances = distances.ravel()
        bulk_density = n_b / volume

    counts, _ = np.histogram(distances, bins=bins)
    bin_widths = np.diff(bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    shells = 4.0 * np.pi * bin_centers**2 * bin_widths
    local_density = counts / (n_a * shells)
    return local_density / bulk_density


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
