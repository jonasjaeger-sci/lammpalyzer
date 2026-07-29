"""Tests for analysis aggregation helpers."""

from pathlib import Path

import pandas as pd

import lammpalyze.analysis as analysis_module
from lammpalyze.analysis import LoadedSimulation, aggregate_thermo, load_project
from lammpalyze.config import parse_input_file


def test_aggregate_thermo_mean_and_std():
    """Aggregate thermo values into timestep means and standard deviations."""

    simulations = [
        LoadedSimulation(index=1, thermo_df=pd.DataFrame({"Step": [0, 1], "Temp": [300.0, 310.0]})),
        LoadedSimulation(index=2, thermo_df=pd.DataFrame({"Step": [0, 1], "Temp": [320.0, 330.0]})),
    ]

    result = aggregate_thermo(simulations, "Temp")

    assert result["mean"].tolist() == [310.0, 320.0]
    assert result["std"].round(6).tolist() == [14.142136, 14.142136]


def test_load_project_reads_configured_pairwise_and_msd_data(tmp_path: Path):
    """Attach both new computed-data formats to their configured simulation."""

    (tmp_path / "pairs.dump").write_text(
        """ITEM: TIMESTEP
0
ITEM: NUMBER OF ENTRIES
1
ITEM: ENTRIES index id1 id2 distance
1 4 2 1.5
""",
        encoding="utf-8",
    )
    (tmp_path / "msd.dat").write_text(
        "# TimeStep c_msd_C[4]\n0 0\n100 0.5\n",
        encoding="utf-8",
    )
    input_file = tmp_path / "lmplyz.inp"
    input_file.write_text(
        'element_list = ["C"]\nDump1 = pairs.dump\nMSD1 = msd.dat\n',
        encoding="utf-8",
    )

    project = load_project(parse_input_file(input_file))

    simulation = project.simulations[0]
    assert simulation.pairwise_df["Pair"].tolist() == ["2-4"]
    assert simulation.msd_df["c_msd_C[4]"].tolist() == [0.0, 0.5]


def test_loaded_simulation_caches_trajectory_timesteps(tmp_path: Path, monkeypatch):
    """Share one trajectory scan across GUI tabs."""

    scans = []

    def fake_index_frames(path):
        scans.append(path)
        return {0: 0, 10: 100, 20: 200}

    monkeypatch.setattr(analysis_module, "index_lammpstrj_frames", fake_index_frames)
    trajectory = tmp_path / "trajectory.lammpstrj"
    simulation = LoadedSimulation(index=1, trajectory_path=trajectory)

    assert simulation.trajectory_timesteps() == [0, 10, 20]
    assert simulation.trajectory_timesteps() == [0, 10, 20]
    assert scans == [trajectory]
