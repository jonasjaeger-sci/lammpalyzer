"""Tests for trajectory distance and angle analysis."""

from pathlib import Path

import pytest

from lammpalyze.analysis import LoadedSimulation
from lammpalyze.geometry import (
    atom_id_groups,
    compute_distances,
    compute_geometry,
    compute_intramolecular_distances,
    distance_pairs,
    parse_atom_id_groups,
    parse_atom_ids,
    parse_distance_selections,
    parse_intramolecular_groups,
)
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


def test_distance_pairs_zip_equal_lists_and_expand_unequal_lists():
    """Preserve positional pairing unless list lengths require all combinations."""

    equal = distance_pairs(
        parse_distance_selections("[1, 2]", "atom"),
        parse_distance_selections("[10, 33]", "com_molecule"),
    )
    unequal = distance_pairs(
        parse_distance_selections("[1, 3]", "atom"),
        parse_distance_selections("[4, 5, 6, 7]", "atom"),
    )

    assert [(first.ids, second.ids) for first, second in equal] == [
        ((1,), (10,)),
        ((2,), (33,)),
    ]
    assert [(first.ids, second.ids) for first, second in unequal] == [
        ((1,), (4,)),
        ((1,), (5,)),
        ((1,), (6,)),
        ((1,), (7,)),
        ((3,), (4,)),
        ((3,), (5,)),
        ((3,), (6,)),
        ((3,), (7,)),
    ]


def test_parse_atom_id_groups_preserves_nested_groups():
    """Treat a flat list as one group and nested lists as independent groups."""

    assert parse_atom_id_groups("[1, 3, 4]") == [(1, 3, 4)]
    assert parse_atom_id_groups("[[1,3,4], [7,8,9]]") == [
        (1, 3, 4),
        (7, 8, 9),
    ]


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


def test_compute_distances_uses_periodic_mass_weighted_atom_com(tmp_path: Path):
    """Unwrap COM atoms across periodic boundaries before mass weighting."""

    simulation = _simulation(
        tmp_path,
        _custom_trajectory(
            [
                "1 1 2.0 0 0 1",
                "2 1 9.5 0 0 1",
                "3 1 0.5 0 0 1",
            ],
            columns="id type x y z mass",
        ),
    )
    pairs = distance_pairs(
        parse_distance_selections("1", "atom"),
        parse_distance_selections("[2,3]", "com_atoms"),
    )

    results = compute_distances([simulation], pairs)

    assert results[0].values == pytest.approx([2.0])
    assert results[0].label == "atom 1 - COM(atoms 2,3)"


def test_compute_distances_resolves_molecule_com_by_mol_column(tmp_path: Path):
    """Calculate molecule COMs from every atom carrying the selected molecule ID."""

    simulation = _simulation(
        tmp_path,
        _custom_trajectory(
            [
                "1 1 0 0 0 1 1",
                "2 1 2 0 0 1 10",
                "3 1 4 0 0 3 10",
            ],
            columns="id type x y z mass mol",
        ),
    )
    pairs = distance_pairs(
        parse_distance_selections("1", "atom"),
        parse_distance_selections("10", "com_molecule"),
    )

    results = compute_distances([simulation], pairs)

    assert results[0].values == pytest.approx([3.5])


def test_compute_distances_calculates_orthogonal_point_plane_distance(tmp_path: Path):
    """Use the normal of three plane atoms for the unsigned orthogonal distance."""

    simulation = _simulation(
        tmp_path,
        _custom_trajectory(
            [
                "1 1 0 0 3",
                "2 1 0 0 0",
                "3 1 1 0 0",
                "4 1 0 1 0",
            ]
        ),
    )
    pairs = distance_pairs(
        parse_distance_selections("1", "atom"),
        parse_distance_selections("[2,3,4]", "plane"),
    )

    results = compute_distances([simulation], pairs)

    assert results[0].values == pytest.approx([3.0])


def test_compute_intramolecular_distances_uses_only_pairs_within_each_group(tmp_path: Path):
    """Expand nested atom groups independently without cross-group pairs or duplicates."""

    simulation = _simulation(
        tmp_path,
        _custom_trajectory(
            [
                "1 1 0 0 0",
                "3 1 1 0 0",
                "4 1 3 0 0",
                "7 1 0 1 0",
                "8 1 0 3 0",
                "9 1 0 5 0",
            ]
        ),
    )
    groups = parse_intramolecular_groups("[[1,3,4],[7,8,9]]", "atoms")

    results = compute_intramolecular_distances([simulation], groups, "atoms")

    assert [result.atom_ids for result in results] == [
        (1, 3),
        (1, 4),
        (3, 4),
        (7, 8),
        (7, 9),
        (8, 9),
    ]


def test_compute_intramolecular_distances_expands_molecule_ids(tmp_path: Path):
    """Resolve each requested molecule to its constituent atom pairs."""

    simulation = _simulation(
        tmp_path,
        _custom_trajectory(
            [
                "1 1 0 0 0 10",
                "2 1 1 0 0 10",
                "3 1 3 0 0 10",
                "4 1 0 2 0 33",
                "5 1 0 4 0 33",
            ],
            columns="id type x y z mol",
        ),
    )
    groups = parse_intramolecular_groups("[10,33]", "molecules")

    results = compute_intramolecular_distances([simulation], groups, "molecules")

    assert [result.atom_ids for result in results] == [(1, 2), (1, 3), (2, 3), (4, 5)]
    assert results[0].label == "mol 10: atom 1 - atom 2"


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


def _custom_trajectory(rows: list[str], columns: str = "id type x y z") -> str:
    """Return one trajectory frame containing arbitrary atom rows and columns."""

    return f"""ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
{len(rows)}
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS {columns}
{chr(10).join(rows)}
"""
