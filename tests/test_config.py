"""Tests for lammpalyze input-file parsing."""

from pathlib import Path

import pytest

from lammpalyze.config import parse_input_file
from lammpalyze.validation import validate_config


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

    config = parse_input_file(input_file)

    with pytest.raises(FileNotFoundError, match="Simulation 1 species"):
        validate_config(config)


def test_parse_input_file_reads_bond_order_cutoffs_and_type_ranges(tmp_path: Path):
    """Read default, pair-specific, symmetric, and compact-range cutoffs."""

    input_file = tmp_path / "lmplyz.inp"
    input_file.write_text(
        """
        element_list = ["C", "H", "Li", "O"]

        # Bond Order cutoffs
        default 0.5
        3 1*2 0.4
        4 3 0.55

        # Bond Files
        BF1 = bonds.reax
        """,
        encoding="utf-8",
    )

    config = parse_input_file(input_file)

    assert config.default_bond_order_cutoff == 0.5
    assert config.bond_order_cutoffs == {(1, 3): 0.4, (2, 3): 0.4, (3, 4): 0.55}
    assert config.bond_order_cutoff(3, 1) == 0.4
    assert config.bond_order_cutoff(1, 4) == 0.5


def test_parse_input_file_uses_default_bond_order_cutoff_without_section(tmp_path: Path):
    """Use 0.3 for every pair when no cutoff section is supplied."""

    input_file = tmp_path / "lmplyz.inp"
    input_file.write_text(
        'element_list = ["C", "H"]\nBF1 = bonds.reax\n',
        encoding="utf-8",
    )

    config = parse_input_file(input_file)

    assert config.default_bond_order_cutoff == 0.3
    assert config.bond_order_cutoffs == {}
    assert config.bond_order_cutoff(1, 2) == 0.3


def test_parse_input_file_rejects_cutoff_types_outside_element_list(tmp_path: Path):
    """Reject cutoff selectors that do not map to a declared element."""

    input_file = tmp_path / "lmplyz.inp"
    input_file.write_text(
        """
        element_list = ["C", "H"]
        # Bond Order cutoffs
        1 3 0.4
        # Bond Files
        BF1 = bonds.reax
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="element_list defines types 1 through 2"):
        parse_input_file(input_file)
