from pathlib import Path

import numpy as np
import pytest

from lammpalyze.analysis import LoadedSimulation
from lammpalyze.parsers import list_lammpstrj_timesteps
from lammpalyze.rdf import compute_rdf


def test_compute_rdf_averages_selected_timestep_range(tmp_path: Path):
    trajectory = tmp_path / "traj.lammpstrj"
    trajectory.write_text(
        """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
3
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS id type q xu yu zu
1 1 0 0 0 0
2 2 0 1 0 0
3 2 0 2 0 0
ITEM: TIMESTEP
10
ITEM: NUMBER OF ATOMS
3
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS id type q xu yu zu
1 1 0 0 0 0
2 2 0 1.5 0 0
3 2 0 2.5 0 0
""",
        encoding="utf-8",
    )
    simulation = LoadedSimulation(
        index=1,
        trajectory_path=trajectory,
        type_to_element={1: "Li", 2: "O"},
    )

    results = compute_rdf([simulation], "Li", "O", (0, 10), 1.0)

    assert list_lammpstrj_timesteps(trajectory) == [0, 10]
    assert len(results) == 1
    assert results[0].timesteps == [0, 10]
    assert np.all(np.isfinite(results[0].g_r))
    assert np.any(results[0].g_r > 0)


def test_compute_rdf_uses_per_simulation_radial_grid(tmp_path: Path):
    trajectory_1 = tmp_path / "traj_1.lammpstrj"
    trajectory_2 = tmp_path / "traj_2.lammpstrj"
    trajectory_1.write_text(_trajectory_text(box_length=10), encoding="utf-8")
    trajectory_2.write_text(_trajectory_text(box_length=8), encoding="utf-8")
    simulations = [
        LoadedSimulation(index=1, trajectory_path=trajectory_1, type_to_element={1: "Li", 2: "O"}),
        LoadedSimulation(index=2, trajectory_path=trajectory_2, type_to_element={1: "Li", 2: "O"}),
    ]

    results = compute_rdf(simulations, "Li", "O", (0, 0), 1.0)

    assert len(results) == 2
    assert results[0].r[-1] != results[1].r[-1]


def test_compute_rdf_rejects_non_positive_bin_width(tmp_path: Path):
    trajectory = tmp_path / "traj.lammpstrj"
    trajectory.write_text(_trajectory_text(box_length=10), encoding="utf-8")
    simulation = LoadedSimulation(
        index=1,
        trajectory_path=trajectory,
        type_to_element={1: "Li", 2: "O"},
    )

    with pytest.raises(ValueError, match="Bin width"):
        compute_rdf([simulation], "Li", "O", (0, 0), 0.0)


def test_compute_rdf_reports_empty_timestep_range(tmp_path: Path):
    trajectory = tmp_path / "traj.lammpstrj"
    trajectory.write_text(_trajectory_text(box_length=10), encoding="utf-8")
    simulation = LoadedSimulation(
        index=1,
        trajectory_path=trajectory,
        type_to_element={1: "Li", 2: "O"},
    )

    with pytest.raises(ValueError, match="No trajectory frames found"):
        compute_rdf([simulation], "Li", "O", (5, 10), 1.0)


def test_compute_rdf_handles_same_element_pairs_without_self_distances(tmp_path: Path):
    trajectory = tmp_path / "traj.lammpstrj"
    trajectory.write_text(
        """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
3
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS id type q xu yu zu
1 1 0 0 0 0
2 1 0 1 0 0
3 2 0 5 0 0
""",
        encoding="utf-8",
    )
    simulation = LoadedSimulation(
        index=1,
        trajectory_path=trajectory,
        type_to_element={1: "Li", 2: "O"},
    )

    results = compute_rdf([simulation], "Li", "Li", (0, 0), 1.0)

    assert len(results) == 1
    assert np.all(np.isfinite(results[0].g_r))
    assert np.any(results[0].g_r > 0)


def _trajectory_text(box_length: int) -> str:
    return f"""ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
3
ITEM: BOX BOUNDS pp pp pp
0 {box_length}
0 {box_length}
0 {box_length}
ITEM: ATOMS id type q xu yu zu
1 1 0 0 0 0
2 2 0 1 0 0
3 2 0 2 0 0
"""
