"""Tests for trajectory distance and angle analysis."""

from pathlib import Path

import pytest

from lammpalyze.analysis import LoadedSimulation
from lammpalyze.geometry import atom_id_groups, compute_geometry, parse_atom_ids
from lammpalyze.geometry_plotting import plot_geometry


def test_parse_atom_ids_accepts_single_ids_and_lists():
    """Accept the scalar and list forms exposed by the GUI fields."""

    assert parse_atom_ids("12") == [12]
    assert parse_atom_ids("[1, 4, 8]") == [1, 4, 8]
    assert atom_id_groups([1, 4], [2, 5]) == [(1, 2), (4, 5)]


def test_atom_id_groups_requires_matching_list_lengths():
    """Reject ambiguous positional pairing of unequal lists."""

    with pytest.raises(ValueError, match="same number"):
        atom_id_groups([1, 2], [3])


def test_compute_geometry_uses_minimum_image_distances(tmp_path: Path):
    """Do not produce a box-length jump when a pair straddles a boundary."""

    simulation = _simulation(tmp_path, _trajectory_text("9.5 0 0", "0.5 0 0", "0.5 1 0"))

    results = compute_geometry([simulation], "distance", [(1, 2)])

    assert results[0].timesteps.tolist() == [0]
    assert results[0].values == pytest.approx([1.0])


def test_compute_geometry_treats_second_atom_as_angle_vertex(tmp_path: Path):
    """Calculate the 1-2-3 angle with atom 2 at the vertex."""

    simulation = _simulation(tmp_path, _trajectory_text("9.5 0 0", "0.5 0 0", "0.5 1 0"))

    results = compute_geometry([simulation], "angle", [(1, 2, 3)])

    assert results[0].values == pytest.approx([90.0])


def test_compute_geometry_converts_scaled_coordinates(tmp_path: Path):
    """Interpret xs/ys/zs values as box-scaled rather than Cartesian."""

    text = _trajectory_text("0.1 0 0", "0.3 0 0", "0.3 0.1 0", columns="xs ys zs")
    simulation = _simulation(tmp_path, text)

    results = compute_geometry([simulation], "distance", [(1, 2)])

    assert results[0].values == pytest.approx([2.0])


def test_compute_geometry_reports_missing_atom_id(tmp_path: Path):
    """Identify the simulation and frame when a requested atom is absent."""

    simulation = _simulation(tmp_path, _trajectory_text("0 0 0", "1 0 0", "1 1 0"))

    with pytest.raises(ValueError, match=r"Simulation 7, timestep 0 lacks atom ID\(s\): 9"):
        compute_geometry([simulation], "distance", [(1, 9)])


def test_plot_geometry_labels_angle_units(tmp_path: Path):
    """Render computed geometry with atom IDs and physical units."""

    simulation = _simulation(tmp_path, _trajectory_text("0 0 0", "1 0 0", "1 1 0"))
    results = compute_geometry([simulation], "angle", [(1, 2, 3)])

    figure = plot_geometry(results, "angle")

    assert figure.axes[0].get_ylabel() == "Angle (degrees)"
    assert figure.axes[0].lines[0].get_label() == "Simulation 7 - 1-2-3"


def test_plot_geometry_adds_multiple_atom_molecule_tracks(tmp_path: Path):
    """Overlay selected atom molecule states on a shared secondary y-axis."""

    simulation = _simulation(tmp_path, _trajectory_text("0 0 0", "1 0 0", "1 1 0"))
    simulation.smiles_id = {0: [["1"], ["2"]], 10: [["1", "2"]]}
    simulation.smiles = {0: ["[Li]", "[O]"], 10: ["[Li]O"]}
    simulation.chem_formulas = {0: ["Li", "O"], 10: ["LiO"]}
    results = compute_geometry([simulation], "distance", [(1, 2)])

    figure = plot_geometry(
        results,
        "distance",
        simulations=[simulation],
        molecule_atom_ids=[1, 2],
        molecule_notation="formula",
        legend_location="upper left",
    )

    assert len(figure.axes) == 2
    molecule_axis = figure.axes[1]
    assert [list(line.get_ydata()) for line in molecule_axis.lines] == [[1, 2], [3, 2]]
    assert [tick.get_text() for tick in molecule_axis.get_yticklabels()] == [
        "1: Li",
        "2: LiO",
        "3: O",
    ]
    assert molecule_axis.get_ylabel() == "Molecule state (formula)"
    assert [text.get_text() for text in figure.axes[0].get_legend().get_texts()] == [
        "Simulation 7 - 1-2",
        "Simulation 7 - atom 1 molecule",
        "Simulation 7 - atom 2 molecule",
    ]


def _simulation(tmp_path: Path, text: str) -> LoadedSimulation:
    """Create one trajectory-backed simulation fixture."""

    trajectory = tmp_path / "trajectory.lammpstrj"
    trajectory.write_text(text, encoding="utf-8")
    return LoadedSimulation(index=7, trajectory_path=trajectory)


def _trajectory_text(first: str, second: str, third: str, columns: str = "x y z") -> str:
    """Return one orthogonal trajectory frame with three atoms."""

    return f"""ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
3
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS id type {columns}
1 1 {first}
2 1 {second}
3 1 {third}
"""
