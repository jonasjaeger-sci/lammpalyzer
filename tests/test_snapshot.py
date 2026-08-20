"""Tests for atom and molecule Snapshot table data."""

from pathlib import Path

import pytest

from lammpalyze.analysis import LoadedSimulation
from lammpalyze.snapshot import (
    SnapshotColumn,
    atom_snapshot_table,
    is_snapshot_notation_column,
    molecule_snapshot_table,
    snapshot_timestep_window,
)


def test_snapshot_notation_columns_identify_copyable_cells():
    """Allow molecule and temporal state cells to participate in copy actions."""

    assert is_snapshot_notation_column("molecule")
    assert is_snapshot_notation_column("state_500")
    assert not is_snapshot_notation_column("atom_id")
    assert not is_snapshot_notation_column("mol_id")


def test_snapshot_timestep_window_uses_available_observations():
    """Select up to five analyzed observations on either side of the target."""

    timesteps = list(range(0, 1100, 100))

    assert snapshot_timestep_window(timesteps, 500) == timesteps
    assert snapshot_timestep_window(timesteps, 100) == timesteps[:7]
    assert snapshot_timestep_window(timesteps, 1000) == timesteps[5:]


def test_atom_snapshot_table_maps_types_components_and_molecule_history(tmp_path: Path):
    """Show every atom with its selected component and surrounding molecule labels."""

    simulation = _snapshot_simulation(tmp_path)

    table = atom_snapshot_table(simulation, 500, "formula")

    assert len(table.columns) == 14
    assert table.columns[:3] == (
        SnapshotColumn("atom_id", "Atom ID"),
        SnapshotColumn("atom_type", "Atom type"),
        SnapshotColumn("mol_id", "Calculated mol_id"),
    )
    assert table.columns[8].heading == "500 (selected)"
    assert table.observation_timesteps == tuple(range(0, 1100, 100))
    assert table.rows[0][:3] == ("1", "2", "0")
    assert table.rows[0][3:8] == ("A", "A", "A", "A", "A")
    assert table.rows[0][8:] == ("AB", "AB", "AB", "AB", "AB", "AB")
    assert table.rows[2][:3] == ("3", "4", "1")


def test_atom_snapshot_table_switches_to_smiles_notation(tmp_path: Path):
    """Use the selected notation for every temporal molecule-state column."""

    table = atom_snapshot_table(_snapshot_simulation(tmp_path), 500, "smiles")

    assert table.rows[0][7] == "[A]"
    assert table.rows[0][8] == "[A][B]"


def test_molecule_snapshot_table_lists_component_labels_and_atoms(tmp_path: Path):
    """Describe each calculated component at exactly the selected timestep."""

    table = molecule_snapshot_table(_snapshot_simulation(tmp_path), 500, "formula")

    assert [column.heading for column in table.columns] == [
        "Calculated mol_id",
        "Chemical formula",
        "Constituent atom IDs",
    ]
    assert table.rows == (
        ("0", "AB", "[1, 2]"),
        ("1", "C", "[3]"),
        ("2", "D", "[4]"),
    )


def test_snapshot_requires_an_analyzed_timestep(tmp_path: Path):
    """Reject a timestep that is absent from bond-derived molecule observations."""

    with pytest.raises(ValueError, match="no bond-derived molecule observation"):
        molecule_snapshot_table(_snapshot_simulation(tmp_path), 550)


def _snapshot_simulation(tmp_path: Path) -> LoadedSimulation:
    """Create aligned trajectory and bond-derived molecule observations."""

    trajectory = tmp_path / "snapshot.lammpstrj"
    trajectory.write_text(
        """ITEM: TIMESTEP
500
ITEM: NUMBER OF ATOMS
4
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS id type x y z
1 2 0 0 0
2 3 1 0 0
3 4 2 0 0
4 5 3 0 0
""",
        encoding="utf-8",
    )
    simulation = LoadedSimulation(index=9, trajectory_path=trajectory)
    early_components = [["1"], ["2", "3"], ["4"]]
    late_components = [["1", "2"], ["3"], ["4"]]
    simulation.smiles_id = {
        timestep: early_components if timestep < 500 else late_components
        for timestep in range(0, 1100, 100)
    }
    simulation.chem_formulas = {
        timestep: ["A", "BC", "D"] if timestep < 500 else ["AB", "C", "D"]
        for timestep in range(0, 1100, 100)
    }
    simulation.smiles = {
        timestep: ["[A]", "[B][C]", "[D]"] if timestep < 500 else ["[A][B]", "[C]", "[D]"]
        for timestep in range(0, 1100, 100)
    }
    return simulation
