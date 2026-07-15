"""Tests for structural-relaxation calculations."""

from pathlib import Path

import numpy as np
import pytest

from lammpalyze.analysis import LoadedSimulation
from lammpalyze.structure import compute_structural_relaxation


def test_compute_structural_relaxation_returns_static_and_incoherent_results(tmp_path: Path):
    """Calculate S(q), identify a peak shell, and compute F_s(q,t)."""

    trajectory = tmp_path / "traj.lammpstrj"
    trajectory.write_text(_trajectory_text(), encoding="utf-8")
    simulation = LoadedSimulation(
        index=1,
        trajectory_path=trajectory,
        type_to_element={1: "Li", 2: "O"},
    )

    results = compute_structural_relaxation(
        [simulation],
        start_timestep=10,
        frame_count=3,
        time_origin_count=2,
        max_q_index=1,
        block_count=2,
        element="Li",
    )

    assert len(results) == 1
    static = results[0].static_structure_factor
    incoherent = results[0].incoherent_scattering
    assert static.timesteps == [10, 20, 30]
    assert len(static.q) == len(static.s_q) == len(static.s_q_error)
    assert static.peak_q > 0
    assert np.all(np.isfinite(static.s_q))
    assert incoherent.origin_timesteps == [10, 20]
    assert incoherent.time[0] == 0
    assert incoherent.f_s[0] == pytest.approx(1.0)
    assert np.all(np.isfinite(incoherent.f_s))
    assert np.all(np.isfinite(incoherent.f_s_error))


def test_compute_structural_relaxation_reports_empty_production_range(tmp_path: Path):
    """Report when no frames exist after the requested production start."""

    trajectory = tmp_path / "traj.lammpstrj"
    trajectory.write_text(_trajectory_text(), encoding="utf-8")
    simulation = LoadedSimulation(index=1, trajectory_path=trajectory)

    with pytest.raises(ValueError, match="No trajectory frames found"):
        compute_structural_relaxation([simulation], start_timestep=1000)


def test_compute_structural_relaxation_rejects_missing_element_mapping(tmp_path: Path):
    """Selecting an element requires the loaded atom-type mapping."""

    trajectory = tmp_path / "traj.lammpstrj"
    trajectory.write_text(_trajectory_text(), encoding="utf-8")
    simulation = LoadedSimulation(index=1, trajectory_path=trajectory)

    with pytest.raises(ValueError, match="element mapping"):
        compute_structural_relaxation([simulation], start_timestep=0, element="Li")


def _trajectory_text() -> str:
    """Return a minimal trajectory with a small Li displacement."""

    frames = []
    for timestep, x_position in [(0, 1.0), (10, 1.1), (20, 1.3), (30, 1.6)]:
        frames.append(
            f"""ITEM: TIMESTEP
{timestep}
ITEM: NUMBER OF ATOMS
3
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS id type q xu yu zu
1 1 0 {x_position} 1 1
2 1 0 3 1 1
3 2 0 5 5 5
"""
        )
    return "".join(frames)
