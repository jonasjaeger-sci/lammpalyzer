"""Tests for LAMMPS and ReaxFF parsers."""

from pathlib import Path

import pytest

from lammpalyze.parsers import (
    copy_lammpstrj_until,
    eval_msd,
    eval_pairwise_dump,
    eval_species,
    eval_thermo,
    index_lammpstrj_frames,
    index_reax_bond_frames,
    iter_lammpstrj_frames,
    parse_bond_observations,
    parse_bonds,
    parse_traj,
    read_lammpstrj_frame,
    read_reax_bonds_frame,
)
from lammpalyze.parsers.bonds import (
    _RawBondFrame,
    _iter_temporally_filtered_bond_frames,
    _temporally_filtered_bond_order_frames,
)


def _raw_bond_frame(timestep: int, bond_orders: dict[tuple[int, int], float]) -> _RawBondFrame:
    """Create a minimal raw bond frame for temporal-filter tests."""

    return _RawBondFrame(
        timestep=timestep,
        atoms={"1": "C", "2": "H"},
        atom_types={"1": 1, "2": 2},
        bond_orders=bond_orders,
        charges={"1": 0.0, "2": 0.0},
    )


def test_eval_species_handles_changing_headers(tmp_path: Path):
    """Merge species columns across changing species-file headers."""

    species_file = tmp_path / "species.out"
    species_file.write_text(
        """
        # Timestep No_Moles No_Specs A B
        0 2 2 1 1
        # Timestep No_Moles No_Specs A C
        1 2 2 0 2
        """,
        encoding="utf-8",
    )

    species, _, frame = eval_species(species_file)

    assert species == ["A", "B", "C"]
    assert frame["No_Moles"].tolist() == [2, 2]
    assert frame["No_Specs"].tolist() == [2, 2]
    assert frame["A"].tolist() == [1, 0]
    assert frame["B"].tolist() == [1, 0]
    assert frame["C"].tolist() == [0, 2]


def test_eval_thermo_extracts_table(tmp_path: Path):
    """Extract the thermo table from a LAMMPS log file."""

    thermo_file = tmp_path / "thermo.log"
    thermo_file.write_text(
        """
        preamble
        Step Temp PotEng
        0 300 -10
        1 301 -11
        Loop time of 1 on 1 procs
        """,
        encoding="utf-8",
    )

    _, frame = eval_thermo(thermo_file)

    assert frame["Step"].tolist() == [0.0, 1.0]
    assert frame["Temp"].tolist() == [300.0, 301.0]


def test_eval_thermo_collects_multiple_run_tables_and_skips_warnings(tmp_path: Path):
    """Continue parsing after LAMMPS loop footers and ignore warning text."""

    thermo_file = tmp_path / "thermo.log"
    thermo_file.write_text(
        """
        run 10
        Step Temp PotEng
        0 300 -10
        WARNING: not numeric
        10 301 -11
        Loop time of 1 on 1 procs
        run 20
        Step Temp PotEng
        20 302 -12
        30 303 -13
        Loop time of 1 on 1 procs
        """,
        encoding="utf-8",
    )

    _, frame = eval_thermo(thermo_file)

    assert frame["Step"].tolist() == [0.0, 10.0, 20.0, 30.0]
    assert frame["Temp"].tolist() == [300.0, 301.0, 302.0, 303.0]
    assert frame["PotEng"].tolist() == [-10.0, -11.0, -12.0, -13.0]


def test_eval_thermo_ignores_later_tables_with_different_columns(tmp_path: Path):
    """Keep one coherent thermo time series when logs include rerun tables."""

    thermo_file = tmp_path / "thermo.log"
    thermo_file.write_text(
        """
        Step Temp
        0 300
        Loop time of 1 on 1 procs
        Step Press E_pair
        10 5 -7
        Loop time of 1 on 1 procs
        """,
        encoding="utf-8",
    )

    _, frame = eval_thermo(thermo_file)

    assert frame.columns.tolist() == ["Step", "Temp"]
    assert frame["Step"].tolist() == [0.0]
    assert frame["Temp"].tolist() == [300.0]


def test_eval_msd_preserves_computed_column_names(tmp_path: Path):
    """Read fix-averaged MSD output with every computed component selectable."""

    msd_file = tmp_path / "msd.dat"
    msd_file.write_text(
        """# Time-averaged data for fix msdout
# TimeStep c_msd_C[1] c_msd_C[4] c_msd_Li[4]
0 0 0 0
100 0.1 0.4 1.2
""",
        encoding="utf-8",
    )

    frame = eval_msd(msd_file)

    assert frame.columns.tolist() == ["Timestep", "c_msd_C[1]", "c_msd_C[4]", "c_msd_Li[4]"]
    assert frame["Timestep"].tolist() == [0.0, 100.0]
    assert frame["c_msd_Li[4]"].tolist() == [0.0, 1.2]


def test_eval_pairwise_dump_fuses_particle_ids_and_exposes_data_columns(tmp_path: Path):
    """Discard local indexes and keep stable pair labels across reversed ID order."""

    pairwise_file = tmp_path / "pairs.dump"
    pairwise_file.write_text(
        """ITEM: TIMESTEP
0
ITEM: NUMBER OF ENTRIES
2
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ENTRIES index c_pid[1] c_pid[2] c_pdist c_energy
1 654 653 3.5 -0.2
2 651 652 1.4 -0.5
ITEM: TIMESTEP
100
ITEM: NUMBER OF ENTRIES
1
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ENTRIES index c_pid[1] c_pid[2] c_pdist c_energy
1 653 654 3.6 -0.1
""",
        encoding="utf-8",
    )

    frame = eval_pairwise_dump(pairwise_file)

    assert frame.columns.tolist() == [
        "Timestep",
        "Pair",
        "Particle 1",
        "Particle 2",
        "c_pdist",
        "c_energy",
    ]
    assert frame["Pair"].tolist() == ["653-654", "651-652", "653-654"]
    assert frame["Particle 1"].tolist() == [653, 651, 653]
    assert frame["c_energy"].tolist() == [-0.2, -0.5, -0.1]


def test_iter_lammpstrj_frames_filters_inclusive_timestep_range(tmp_path: Path):
    """Yield only trajectory frames inside an inclusive timestep range."""

    trajectory = tmp_path / "traj.lammpstrj"
    trajectory.write_text(
        """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS id type q xu yu zu
1 2 0 1 2 3
ITEM: TIMESTEP
10
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS pp pp pp
0 20
0 20
0 20
ITEM: ATOMS id type q xu yu zu
2 3 0 4 5 6
""",
        encoding="utf-8",
    )

    frames = list(iter_lammpstrj_frames(trajectory, (5, 10)))

    assert len(frames) == 1
    assert frames[0].timestep == 10
    assert frames[0].bounds.tolist() == [[0.0, 20.0], [0.0, 20.0], [0.0, 20.0]]
    assert frames[0].atoms[0].atom_id == 2
    assert frames[0].atoms[0].atom_type == 3
    assert (frames[0].atoms[0].x, frames[0].atoms[0].y, frames[0].atoms[0].z) == (4.0, 5.0, 6.0)


def test_trajectory_frame_index_supports_direct_frame_reads(tmp_path: Path):
    """Seek directly to an indexed frame instead of rescanning prior frames."""

    trajectory = tmp_path / "traj.lammpstrj"
    trajectory.write_text(
        """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS id type x y z
1 1 1 2 3
ITEM: TIMESTEP
10
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS id type x y z
1 1 4 5 6
""",
        encoding="utf-8",
    )

    offsets = index_lammpstrj_frames(trajectory)
    frame = read_lammpstrj_frame(trajectory, 10, frame_offset=offsets[10])

    assert list(offsets) == [0, 10]
    assert (frame.atoms[0].x, frame.atoms[0].y, frame.atoms[0].z) == (4.0, 5.0, 6.0)


def test_parse_traj_accepts_reordered_wrapped_columns(tmp_path: Path):
    """Build legacy arrays from flexible atom tables without requiring xu."""

    trajectory = tmp_path / "traj.lammpstrj"
    trajectory.write_text(
        """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS z id q y type x fx
13 1 -0.2 -1 2 11 4.5
""",
        encoding="utf-8",
    )

    frame = next(parse_traj(trajectory))

    assert frame.tolist() == [[-0.2, 1.0, 9.0, 3.0]]


def test_copy_lammpstrj_until_preserves_raw_frames(tmp_path: Path):
    """Copy trajectory frames from the beginning through a selected timestep."""

    trajectory = tmp_path / "traj.lammpstrj"
    text = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS id type x y z
1 1 1 2 3
ITEM: TIMESTEP
10
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS id type x y z
1 1 4 5 6
ITEM: TIMESTEP
20
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS id type x y z
1 1 7 8 9
"""
    trajectory.write_text(text, encoding="utf-8")
    output = tmp_path / "cut.lammpstrj"

    frames = copy_lammpstrj_until(trajectory, output, 10)

    assert frames == 2
    assert output.read_text(encoding="utf-8") == "\n".join(text.splitlines()[:20]) + "\n"


def test_parse_bonds_applies_pair_cutoff_before_rdkit_connectivity(tmp_path: Path):
    """Apply default and pair-specific cutoffs before RDKit connectivity."""

    pytest.importorskip("rdkit")
    bond_file = tmp_path / "bonds.reax"
    bond_file.write_text(
        """# Timestep 0
# Number of particles 2
1 1 1 2 1 0.4 0.4 0 0
2 2 1 1 1 0.4 0.4 0 0
""",
        encoding="utf-8",
    )

    _, _, default_smiles_atoms, _ = parse_bonds(bond_file, {1: "C", 2: "H"})
    _, _, pair_cutoff_smiles_atoms, _ = parse_bonds(
        bond_file,
        {1: "C", 2: "H"},
        bond_order_cutoffs={(1, 2): 0.3},
    )

    assert default_smiles_atoms[0] == [["1"], ["2"]]
    assert pair_cutoff_smiles_atoms[0] == [["1", "2"]]


def test_parse_bonds_keeps_bond_equal_to_cutoff(tmp_path: Path):
    """Treat the cutoff as inclusive because only lower orders are excluded."""

    pytest.importorskip("rdkit")
    bond_file = tmp_path / "bonds.reax"
    bond_file.write_text(
        """# Timestep 0
# Number of particles 2
1 1 1 2 1 0.5 0.5 0 0
2 2 1 1 1 0.5 0.5 0 0
""",
        encoding="utf-8",
    )

    _, _, smiles_atoms, _ = parse_bonds(
        bond_file,
        {1: "C", 2: "H"},
        bond_order_cutoffs={(1, 2): 0.5},
    )

    assert smiles_atoms[0] == [["1", "2"]]


def test_temporal_filter_backdates_accepted_bond_state_changes():
    """Apply a persistent state change starting at the first candidate frame."""

    frames = [
        _raw_bond_frame(0, {}),
        _raw_bond_frame(10, {(1, 2): 0.8}),
        _raw_bond_frame(20, {(1, 2): 0.8}),
    ]

    bond_orders = _temporally_filtered_bond_order_frames(
        frames,
        {},
        0.3,
        {},
        persistence_frames=2,
        persistence_timesteps=0,
        hysteresis=0.0,
    )

    assert bond_orders == [{}, {(1, 2): 1}, {(1, 2): 1}]


def test_temporal_filter_streams_frames_without_pending_changes():
    """Release each finalized frame without consuming the full input stream."""

    consumed = []

    def frames():
        for timestep in (0, 10, 20):
            consumed.append(timestep)
            yield _raw_bond_frame(timestep, {(1, 2): 0.8})

    filtered = _iter_temporally_filtered_bond_frames(
        frames(),
        {},
        0.3,
        {},
        persistence_frames=1,
        persistence_timesteps=0,
        hysteresis=0.0,
    )

    frame, bond_orders = next(filtered)

    assert frame.timestep == 0
    assert bond_orders == {(1, 2): 1}
    assert consumed == [0]


def test_temporal_filter_keeps_stable_state_for_unresolved_flickers():
    """Do not backdate a candidate state that disappears before acceptance."""

    frames = [
        _raw_bond_frame(0, {(1, 2): 0.8}),
        _raw_bond_frame(10, {}),
        _raw_bond_frame(20, {(1, 2): 0.8}),
    ]

    bond_orders = _temporally_filtered_bond_order_frames(
        frames,
        {},
        0.3,
        {},
        persistence_frames=2,
        persistence_timesteps=0,
        hysteresis=0.0,
    )

    assert bond_orders == [{(1, 2): 1}, {(1, 2): 1}, {(1, 2): 1}]


def test_parse_bonds_filters_one_frame_bond_state_flicker(tmp_path: Path):
    """Keep an accepted bond when its disappearance does not persist."""

    pytest.importorskip("rdkit")
    bond_file = tmp_path / "bonds.reax"
    bond_file.write_text(
        """# Timestep 0
# Number of particles 2
1 1 1 2 0 0.8 0.8 0 0.1
2 2 1 1 0 0.8 0.8 0 -0.1
# Timestep 10
# Number of particles 2
1 1 0 0 0 0 0.1
2 2 0 0 0 0 -0.1
# Timestep 20
# Number of particles 2
1 1 1 2 0 0.8 0.8 0 0.1
2 2 1 1 0 0.8 0.8 0 -0.1
""",
        encoding="utf-8",
    )

    result = parse_bond_observations(
        bond_file,
        {1: "C", 2: "H"},
        bond_state_persistence_frames=2,
    )

    assert result.smiles_atoms == {0: [["1", "2"]], 10: [["1", "2"]], 20: [["1", "2"]]}
    offsets = index_reax_bond_frames(bond_file)
    bonds = read_reax_bonds_frame(
        bond_file,
        20,
        frame_offset=offsets[20],
    )
    assert [(bond.atom_i, bond.atom_j) for bond in bonds] == [(1, 2)]


def test_parse_bonds_accepts_transition_after_configured_timestep_duration(tmp_path: Path):
    """Backdate accepted transitions after the configured timestep duration."""

    pytest.importorskip("rdkit")
    bond_file = tmp_path / "bonds.reax"
    unbonded = "1 1 0 0 0 0 0.1\n2 2 0 0 0 0 -0.1\n"
    bonded = "1 1 1 2 0 0.8 0.8 0 0.1\n2 2 1 1 0 0.8 0.8 0 -0.1\n"
    bond_file.write_text(
        "# Timestep 0\n# Number of particles 2\n"
        + unbonded
        + "# Timestep 10\n# Number of particles 2\n"
        + bonded
        + "# Timestep 20\n# Number of particles 2\n"
        + bonded
        + "# Timestep 110\n# Number of particles 2\n"
        + bonded,
        encoding="utf-8",
    )

    result = parse_bond_observations(
        bond_file,
        {1: "C", 2: "H"},
        bond_state_persistence_timesteps=100,
    )

    assert result.smiles_atoms[10] == [["1", "2"]]
    assert result.smiles_atoms[20] == [["1", "2"]]
    assert result.smiles_atoms[110] == [["1", "2"]]


def test_parse_bonds_calculates_component_and_element_charge_statistics(tmp_path: Path):
    """Aggregate continuous ReaxFF charges without assigning formal charges."""

    pytest.importorskip("rdkit")
    bond_file = tmp_path / "bonds.reax"
    bond_file.write_text(
        """# Timestep 0
# Number of particles 3
1 1 2 2 3 0 0.8 0.8 1.6 0 0.8
2 2 1 1 0 0.8 0.8 0 0.1
3 2 1 1 0 0.8 0.8 0 -0.1
""",
        encoding="utf-8",
    )

    result = parse_bond_observations(bond_file, {1: "C", 2: "H"})

    assert result.atom_charges[0] == {"1": 0.8, "2": 0.1, "3": -0.1}
    assert result.charge_statistics[0]["H"].mean == 0.0
    assert result.charge_statistics[0]["H"].std == pytest.approx(0.1)
    assert result.component_properties[0][0].charge == pytest.approx(0.8)
    assert result.component_properties[0][0].ion_candidate == "cation candidate"


def test_parse_bonds_flags_and_excludes_high_valence_components(tmp_path: Path):
    """Record conservative quality flags and exclusion indexes for reactions."""

    pytest.importorskip("rdkit")
    bond_file = tmp_path / "bonds.reax"
    hydrogen_rows = "".join(
        f"{atom_id} 2 1 1 0 1.0 1.0 0 0.0\n" for atom_id in range(2, 7)
    )
    bond_file.write_text(
        "# Timestep 0\n# Number of particles 6\n"
        "1 1 5 2 3 4 5 6 0 1.0 1.0 1.0 1.0 1.0 5.0 0 0.0\n"
        + hydrogen_rows,
        encoding="utf-8",
    )

    result = parse_bond_observations(
        bond_file,
        {1: "C", 2: "H"},
        structure_quality_mode="exclude",
    )

    assert result.component_properties[0][0].suspicious
    assert "exceeds supported valence 4" in result.component_properties[0][0].warnings[0]
    assert result.excluded_components[0] == {0}

    skip_result = parse_bond_observations(
        bond_file,
        {1: "C", 2: "H"},
        structure_quality_mode="skip",
    )
    assert skip_result.excluded_components[0] == {0}
