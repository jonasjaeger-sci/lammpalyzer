"""Static structure factor and incoherent scattering helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lammpalyze.analysis import LoadedSimulation
from lammpalyze.parsers import TrajectoryFrame, iter_lammpstrj_frames


@dataclass(frozen=True)
class StaticStructureFactorResult:
    """Static structure factor averaged over frames and q-vector shells."""

    simulation_index: int
    q: np.ndarray
    s_q: np.ndarray
    s_q_error: np.ndarray
    timesteps: list[int]
    peak_q: float
    peak_shell_index: int
    q_shell_vectors: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class IncoherentScatteringResult:
    """Self intermediate scattering function at the first S(q) peak."""

    simulation_index: int
    time: np.ndarray
    f_s: np.ndarray
    f_s_error: np.ndarray
    origin_timesteps: list[int]
    q: float


@dataclass(frozen=True)
class StructuralRelaxationResult:
    """Combined structural-relaxation result for one simulation."""

    simulation_index: int
    static_structure_factor: StaticStructureFactorResult
    incoherent_scattering: IncoherentScatteringResult


def compute_structural_relaxation(
    simulations: list[LoadedSimulation],
    start_timestep: int,
    frame_count: int = 100,
    time_origin_count: int = 10,
    max_q_index: int = 8,
    block_count: int = 5,
    element: str | None = None,
) -> list[StructuralRelaxationResult]:
    """Compute S(q) and F_s(q,t) for selected trajectory simulations."""

    if frame_count < 1:
        raise ValueError("Frame count must be at least 1.")
    if time_origin_count < 1:
        raise ValueError("Time-origin count must be at least 1.")
    if max_q_index < 1:
        raise ValueError("Max q index must be at least 1.")
    if block_count < 1:
        raise ValueError("Block count must be at least 1.")

    results: list[StructuralRelaxationResult] = []
    for simulation in simulations:
        if simulation.trajectory_path is None:
            continue
        frames = [
            frame
            for frame in iter_lammpstrj_frames(simulation.trajectory_path)
            if frame.timestep >= start_timestep
        ]
        selected_frames = _uniform_sample(frames, frame_count)
        if not selected_frames:
            continue
        if len(selected_frames) < 2:
            raise ValueError(
                f"Simulation {simulation.index} needs at least two selected frames "
                "to calculate the incoherent scattering function."
            )

        positions_by_frame = [
            _positions_for_element(frame, simulation.type_to_element, element)
            for frame in selected_frames
        ]
        atom_count = len(positions_by_frame[0])
        if atom_count == 0:
            label = element if element is not None else "selected"
            raise ValueError(f"Simulation {simulation.index} has no {label} atoms in the trajectory.")
        if any(len(positions) != atom_count for positions in positions_by_frame):
            raise ValueError(
                f"Simulation {simulation.index} changes the selected atom count across frames."
            )

        reference_lengths = np.mean(
            np.vstack([_box_lengths(frame) for frame in selected_frames]),
            axis=0,
        )
        q_shells = _q_shells(reference_lengths, max_q_index)
        static_result = _static_structure_factor(
            simulation.index,
            selected_frames,
            positions_by_frame,
            q_shells,
            block_count,
        )
        incoherent_result = _incoherent_scattering_function(
            simulation.index,
            selected_frames,
            positions_by_frame,
            q_shells[static_result.peak_shell_index][1],
            static_result.peak_q,
            time_origin_count,
            block_count,
        )
        results.append(
            StructuralRelaxationResult(
                simulation_index=simulation.index,
                static_structure_factor=static_result,
                incoherent_scattering=incoherent_result,
            )
        )

    if not results:
        raise ValueError(f"No trajectory frames found at or after timestep {start_timestep}.")
    return results


def _static_structure_factor(
    simulation_index: int,
    frames: list[TrajectoryFrame],
    positions_by_frame: list[np.ndarray],
    q_shells: list[tuple[float, np.ndarray]],
    block_count: int,
) -> StaticStructureFactorResult:
    """Compute frame-wise S(q) samples and block-averaged uncertainty."""

    samples = np.array(
        [
            [
                _frame_static_structure_factor(positions, q_vectors)
                for _, q_vectors in q_shells
            ]
            for positions in positions_by_frame
        ],
        dtype=float,
    )
    s_q = np.mean(samples, axis=0)
    s_q_error = _block_standard_error(samples, block_count)
    q_values = np.array([q for q, _ in q_shells], dtype=float)
    peak_index = _first_peak_index(q_values, s_q)
    return StaticStructureFactorResult(
        simulation_index=simulation_index,
        q=q_values,
        s_q=s_q,
        s_q_error=s_q_error,
        timesteps=[frame.timestep for frame in frames],
        peak_q=float(q_values[peak_index]),
        peak_shell_index=peak_index,
        q_shell_vectors=tuple(q_vectors for _, q_vectors in q_shells),
    )


def _incoherent_scattering_function(
    simulation_index: int,
    frames: list[TrajectoryFrame],
    positions_by_frame: list[np.ndarray],
    q_vectors: np.ndarray,
    q_magnitude: float,
    time_origin_count: int,
    block_count: int,
) -> IncoherentScatteringResult:
    """Compute F_s(q,t) from uniformly spaced time origins."""

    origin_indexes = _uniform_origin_indexes(len(frames), time_origin_count)
    origin_samples = np.full((len(origin_indexes), len(frames)), np.nan, dtype=float)
    time_samples = np.full((len(origin_indexes), len(frames)), np.nan, dtype=float)
    timesteps = np.array([frame.timestep for frame in frames], dtype=float)

    for row, origin_index in enumerate(origin_indexes):
        origin_positions = positions_by_frame[origin_index]
        for target_index in range(origin_index, len(frames)):
            lag_index = target_index - origin_index
            displacement = positions_by_frame[target_index] - origin_positions
            phases = displacement @ q_vectors.T
            origin_samples[row, lag_index] = float(np.cos(phases).mean())
            time_samples[row, lag_index] = timesteps[target_index] - timesteps[origin_index]

    valid = np.any(np.isfinite(origin_samples), axis=0)
    values = origin_samples[:, valid]
    times = np.nanmean(time_samples[:, valid], axis=0)
    return IncoherentScatteringResult(
        simulation_index=simulation_index,
        time=times,
        f_s=np.nanmean(values, axis=0),
        f_s_error=_block_standard_error(values, block_count),
        origin_timesteps=[frames[index].timestep for index in origin_indexes],
        q=q_magnitude,
    )


def _frame_static_structure_factor(positions: np.ndarray, q_vectors: np.ndarray) -> float:
    """Return S(q) averaged over one shell of q vectors for one frame."""

    phases = positions @ q_vectors.T
    density_modes = np.exp(1j * phases).sum(axis=0)
    return float(np.mean(np.abs(density_modes) ** 2) / len(positions))


def _positions_for_element(
    frame: TrajectoryFrame,
    type_to_element: dict[int, str] | None,
    element: str | None,
) -> np.ndarray:
    """Return positions for all atoms or one selected element."""

    if element is not None and type_to_element is None:
        raise ValueError("Selected an element, but no atom-type element mapping is available.")
    return np.array(
        [
            [atom.x, atom.y, atom.z]
            for atom in frame.atoms
            if element is None or type_to_element.get(atom.atom_type) == element
        ],
        dtype=float,
    )


def _box_lengths(frame: TrajectoryFrame) -> np.ndarray:
    """Return x, y, and z box lengths for a trajectory frame."""

    return frame.bounds[:, 1] - frame.bounds[:, 0]


def _q_shells(box_lengths: np.ndarray, max_q_index: int) -> list[tuple[float, np.ndarray]]:
    """Return q-vector shells grouped by equal rounded magnitude."""

    if np.any(box_lengths <= 0):
        raise ValueError("Trajectory box dimensions must be positive.")
    shells: dict[float, list[np.ndarray]] = {}
    for nx in range(-max_q_index, max_q_index + 1):
        for ny in range(-max_q_index, max_q_index + 1):
            for nz in range(-max_q_index, max_q_index + 1):
                if nx == 0 and ny == 0 and nz == 0:
                    continue
                vector = 2.0 * np.pi * np.array([nx, ny, nz], dtype=float) / box_lengths
                magnitude = float(np.linalg.norm(vector))
                shells.setdefault(round(magnitude, 10), []).append(vector)
    return [
        (magnitude, np.vstack(vectors))
        for magnitude, vectors in sorted(shells.items(), key=lambda item: item[0])
    ]


def _first_peak_index(q_values: np.ndarray, s_q: np.ndarray) -> int:
    """Return the first local maximum, falling back to the global maximum."""

    for index in range(1, len(s_q) - 1):
        if s_q[index] >= s_q[index - 1] and s_q[index] >= s_q[index + 1]:
            return index
    return int(np.argmax(s_q))


def _uniform_sample(items: list[TrajectoryFrame], count: int) -> list[TrajectoryFrame]:
    """Select up to ``count`` uniformly spaced items without duplicates."""

    if not items:
        return []
    count = min(count, len(items))
    indexes = np.linspace(0, len(items) - 1, count)
    unique_indexes = sorted({int(round(index)) for index in indexes})
    return [items[index] for index in unique_indexes]


def _uniform_origin_indexes(frame_count: int, origin_count: int) -> list[int]:
    """Return uniformly spaced origin indexes with at least one future frame."""

    if frame_count < 2:
        return [0]
    origin_count = min(origin_count, frame_count - 1)
    indexes = np.linspace(0, frame_count - 2, origin_count)
    return sorted({int(round(index)) for index in indexes})


def _block_standard_error(values: np.ndarray, block_count: int) -> np.ndarray:
    """Estimate the standard error from means of contiguous sample blocks."""

    if values.ndim == 1:
        values = values[:, np.newaxis]
    sample_count = values.shape[0]
    if sample_count < 2:
        return np.zeros(values.shape[1], dtype=float)
    block_count = min(block_count, sample_count)
    block_means = []
    for block in np.array_split(values, block_count, axis=0):
        means = np.full(block.shape[1], np.nan, dtype=float)
        for column in range(block.shape[1]):
            column_values = block[:, column]
            column_values = column_values[np.isfinite(column_values)]
            if len(column_values):
                means[column] = float(np.mean(column_values))
        block_means.append(means)
    block_means_array = np.vstack(block_means)
    finite_counts = np.sum(np.isfinite(block_means_array), axis=0)
    error = np.zeros(block_means_array.shape[1], dtype=float)
    for column, finite_count in enumerate(finite_counts):
        if finite_count < 2:
            continue
        column_values = block_means_array[:, column]
        column_values = column_values[np.isfinite(column_values)]
        error[column] = float(np.std(column_values, ddof=1) / np.sqrt(finite_count))
    return error
