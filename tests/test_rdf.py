"""Tests for radial distribution function calculations."""

from pathlib import Path

import numpy as np
import pytest

from lammpalyze.analysis import LoadedSimulation
from lammpalyze.parsers import list_lammpstrj_timesteps
from lammpalyze.rdf import compute_rdf, parse_rdf_ids


def test_parse_rdf_ids_accepts_lists_and_star_ranges():
    """Expand the molecule-ID syntax shown in the RDF selector."""

    assert parse_rdf_ids("1*11,15,17") == [*range(1, 12), 15, 17]
    assert parse_rdf_ids("3 1,2") == [1, 2, 3]


@pytest.mark.parametrize("value", ["", "4*2", "1-3", "a"])
def test_parse_rdf_ids_rejects_invalid_values(value: str):
    """Report malformed, empty, and descending RDF ID selections."""

    with pytest.raises(ValueError):
        parse_rdf_ids(value)


def test_compute_rdf_averages_selected_timestep_range(tmp_path: Path):
    """Average RDF values over the selected timestep range."""

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
    assert results[0].label == "Li - O"
    assert results[0].timesteps == [0, 10]
    assert np.all(np.isfinite(results[0].g_r))
    assert np.any(results[0].g_r > 0)


def test_compute_rdf_samples_from_start_through_inclusive_end(tmp_path: Path):
    """Use frames at start-anchored sampling intervals, including the end."""

    trajectory = tmp_path / "sampled.lammpstrj"
    trajectory.write_text(
        _multi_frame_trajectory_text(range(400000, 416001, 500)),
        encoding="utf-8",
    )
    simulation = LoadedSimulation(
        index=1,
        trajectory_path=trajectory,
        type_to_element={1: "Li", 2: "O"},
    )

    result = compute_rdf(
        [simulation],
        "Li",
        "O",
        (400000, 416000),
        1.0,
        sampling_frequency=1000,
    )[0]

    assert result.timesteps == list(range(400000, 416001, 1000))


@pytest.mark.parametrize("sampling_frequency", [0, -1, 1.5, True])
def test_compute_rdf_rejects_invalid_sampling_frequency(
    tmp_path: Path,
    sampling_frequency,
):
    """Require a positive integer timestep interval."""

    trajectory = tmp_path / "traj.lammpstrj"
    trajectory.write_text(_trajectory_text(box_length=10), encoding="utf-8")
    simulation = LoadedSimulation(
        index=1,
        trajectory_path=trajectory,
        type_to_element={1: "Li", 2: "O"},
    )

    with pytest.raises(ValueError, match="Sampling frequency"):
        compute_rdf(
            [simulation],
            "Li",
            "O",
            (0, 0),
            1.0,
            sampling_frequency=sampling_frequency,
        )


def test_compute_rdf_uses_per_simulation_radial_grid(tmp_path: Path):
    """Keep separate radial grids for simulations with different box sizes."""

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


def test_compute_rdf_uses_exact_timesteps_per_simulation(tmp_path: Path):
    """Select exact frames from simulations with non-overlapping timestep ranges."""

    trajectory_1 = tmp_path / "traj_1.lammpstrj"
    trajectory_2 = tmp_path / "traj_2.lammpstrj"
    trajectory_1.write_text(_two_frame_trajectory_text(0, 50), encoding="utf-8")
    trajectory_2.write_text(_two_frame_trajectory_text(160000, 160050), encoding="utf-8")
    simulations = [
        LoadedSimulation(index=1, trajectory_path=trajectory_1, type_to_element={1: "Li", 2: "O"}),
        LoadedSimulation(index=2, trajectory_path=trajectory_2, type_to_element={1: "Li", 2: "O"}),
    ]

    results = compute_rdf(
        simulations,
        "Li",
        "O",
        (50, 160050),
        1.0,
        timesteps_by_simulation={1: [50], 2: [160050]},
        sampling_frequency=7,
    )

    assert [result.simulation_index for result in results] == [1, 2]
    assert results[0].timesteps == [50]
    assert results[1].timesteps == [160050]


def test_compute_rdf_rejects_non_positive_bin_width(tmp_path: Path):
    """Reject zero or negative RDF bin widths."""

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
    """Report an empty RDF timestep range."""

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
    """Compute same-element RDFs without including self distances."""

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


def test_compute_rdf_selects_explicit_atom_types_with_the_same_element(tmp_path: Path):
    """Keep force-field atom types distinct even when they map to one element."""

    trajectory = tmp_path / "typed.lammpstrj"
    trajectory.write_text(
        """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
3
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS id type x y z
1 1 0 0 0
2 2 1 0 0
3 3 3 0 0
""",
        encoding="utf-8",
    )
    simulation = LoadedSimulation(
        index=1,
        trajectory_path=trajectory,
        type_to_element={1: "C", 2: "C", 3: "C"},
    )

    result = compute_rdf(
        [simulation],
        "Carbonyl C",
        "Ring C",
        (0, 0),
        1.0,
        atom_types_a=[1],
        atom_types_b=[2],
    )[0]

    assert result.g_r[1] > 0
    assert np.count_nonzero(result.g_r) == 1


def test_compute_rdf_uses_periodic_mass_weighted_molecule_centers(tmp_path: Path):
    """Calculate molecule RDF distances from periodic centers of mass."""

    oxygen_mass = 15.999
    hydrogen_mass = 1.008
    molecule_one_com = (
        hydrogen_mass * 9.8 + oxygen_mass * 10.2
    ) / (hydrogen_mass + oxygen_mass)
    molecule_two_x = molecule_one_com % 10.0 + 2.2
    trajectory = tmp_path / "molecules.lammpstrj"
    trajectory.write_text(
        f"""ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
3
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS id mol type x y z
1 1 1 9.8 0 0
2 1 2 0.2 0 0
3 2 2 {molecule_two_x} 0 0
""",
        encoding="utf-8",
    )
    simulation = LoadedSimulation(
        index=1,
        trajectory_path=trajectory,
        type_to_element={1: "H", 2: "O"},
    )

    result = compute_rdf(
        [simulation],
        "Solvent",
        "Ion",
        (0, 0),
        1.0,
        molecule_ids_a=[1],
        molecule_ids_b=[2],
    )[0]

    assert result.g_r[2] > 0
    assert np.count_nonzero(result.g_r) == 1


def test_compute_molecule_rdf_requires_mol_column(tmp_path: Path):
    """Reject molecule mode when the trajectory lacks molecule IDs."""

    trajectory = tmp_path / "atoms.lammpstrj"
    trajectory.write_text(_trajectory_text(box_length=10), encoding="utf-8")
    simulation = LoadedSimulation(
        index=1,
        trajectory_path=trajectory,
        type_to_element={1: "Li", 2: "O"},
    )

    with pytest.raises(ValueError, match="mol"):
        compute_rdf(
            [simulation],
            "A",
            "B",
            (0, 0),
            1.0,
            molecule_ids_a=[1],
            molecule_ids_b=[2],
        )


def _trajectory_text(box_length: int) -> str:
    """Return a minimal trajectory file with a configurable cubic box."""

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


def _two_frame_trajectory_text(first_timestep: int, second_timestep: int) -> str:
    """Return a two-frame trajectory with distinguishable timesteps."""

    return f"""ITEM: TIMESTEP
{first_timestep}
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
{second_timestep}
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
"""


def _multi_frame_trajectory_text(timesteps) -> str:
    """Return trajectory frames for an arbitrary timestep sequence."""

    frames = []
    for position, timestep in enumerate(timesteps):
        offset = position * 0.01
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
1 1 0 0 0 0
2 2 0 {1.0 + offset} 0 0
3 2 0 {2.0 + offset} 0 0
"""
        )
    return "".join(frames)
