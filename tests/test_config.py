"""Tests for lammpalyze input-file parsing."""

from pathlib import Path

import pytest

from lammpalyze.config import parse_input_file


def test_parse_input_file_groups_simulations(tmp_path: Path):
    """Parse grouped simulation paths from a sample input file."""

    for name in [
        "bonds_R1.reax",
        "species_R1.out",
        "thermo_R1.log",
        "traj_R1.lammpstrj",
        "species_R2.out",
    ]:
        (tmp_path / name).write_text("", encoding="utf-8")

    input_file = tmp_path / "lmplyz.inp"
    input_file.write_text(
        """
        element_list = ["C", "H"]
        BF1 = bonds_R1.reax
        SF1 = species_R1.out
        ThermoF1 = thermo_R1.log
        TrajF1 = traj_R1.lammpstrj
        SF2 = species_R2.out
        """,
        encoding="utf-8",
    )

    config = parse_input_file(input_file)

    assert config.element_list == ["C", "H"]
    assert [simulation.index for simulation in config.simulations] == [1, 2]
    assert config.simulations[0].bond == tmp_path / "bonds_R1.reax"
    assert config.simulations[1].species == tmp_path / "species_R2.out"


def test_parse_input_file_reports_missing_referenced_files(tmp_path: Path):
    """Report a helpful error when referenced output files are missing."""

    input_file = tmp_path / "lmplyz.inp"
    input_file.write_text(
        """
        element_list = ["C", "H"]
        SF1 = missing_species.out
        """,
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="simulation 1 species"):
        parse_input_file(input_file)
