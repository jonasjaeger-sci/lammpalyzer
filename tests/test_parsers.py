"""Tests for LAMMPS and ReaxFF parsers."""

from pathlib import Path

from lammpalyze.parsers import eval_species, eval_thermo, iter_lammpstrj_frames


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
