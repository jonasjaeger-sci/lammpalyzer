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
        "pairs_R1.dump",
        "msd_R1.dat",
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
        Dump1 = pairs_R1.dump
        MSD1 = msd_R1.dat
        SF2 = species_R2.out
        """,
        encoding="utf-8",
    )

    config = parse_input_file(input_file)

    assert config.element_list == ["C", "H"]
    assert [simulation.index for simulation in config.simulations] == [1, 2]
    assert config.simulations[0].bond == tmp_path / "bonds_R1.reax"
    assert config.simulations[0].pairwise == tmp_path / "pairs_R1.dump"
    assert config.simulations[0].msd == tmp_path / "msd_R1.dat"
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
    """Use 0.5 for every pair when no cutoff section is supplied."""

    input_file = tmp_path / "lmplyz.inp"
    input_file.write_text(
        'element_list = ["C", "H"]\nBF1 = bonds.reax\n',
        encoding="utf-8",
    )

    config = parse_input_file(input_file)

    assert config.default_bond_order_cutoff == 0.5
    assert config.bond_order_cutoffs == {}
    assert config.bond_order_cutoff(1, 2) == 0.5


@pytest.mark.parametrize("setting", ["boundary p p f", "boundary = p p f"])
def test_parse_input_file_reads_boundary_modes(tmp_path: Path, setting: str):
    """Accept command-style and assignment-style boundary settings."""

    input_file = tmp_path / "lmplyz.inp"
    input_file.write_text(
        f'element_list = ["C", "H"]\n{setting}\nBF1 = bonds.reax\n',
        encoding="utf-8",
    )

    assert parse_input_file(input_file).boundary == ("p", "p", "f")


def test_parse_input_file_defaults_to_fully_periodic_boundaries(tmp_path: Path):
    """Assume periodic boundaries on every axis when no setting is present."""

    input_file = tmp_path / "lmplyz.inp"
    input_file.write_text(
        'element_list = ["C", "H"]\nBF1 = bonds.reax\n',
        encoding="utf-8",
    )

    assert parse_input_file(input_file).boundary == ("p", "p", "p")


@pytest.mark.parametrize("setting", ["boundary p f", "boundary p s f", "boundary p p p f"])
def test_parse_input_file_rejects_invalid_boundary_modes(tmp_path: Path, setting: str):
    """Require exactly three periodic or fixed boundary values."""

    input_file = tmp_path / "lmplyz.inp"
    input_file.write_text(
        f'element_list = ["C", "H"]\n{setting}\nBF1 = bonds.reax\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="boundary must contain exactly three"):
        parse_input_file(input_file)


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


def test_parse_input_file_reads_temporal_quality_and_charge_settings(tmp_path: Path):
    """Read configurable filtering, quality, and ion-candidate settings."""

    input_file = tmp_path / "lmplyz.inp"
    input_file.write_text(
        """
        element_list = ["C", "H"]
        bond_state_persistence_frames = 3
        bond_state_persistence_timesteps = 100
        bond_order_hysteresis = 0.05
        structure_quality_mode = exclude
        ion_charge_threshold = 0.7
        BF1 = bonds.reax
        """,
        encoding="utf-8",
    )

    config = parse_input_file(input_file)

    assert config.bond_state_persistence_frames == 3
    assert config.bond_state_persistence_timesteps == 100
    assert config.bond_order_hysteresis == 0.05
    assert config.structure_quality_mode == "exclude"
    assert config.ion_charge_threshold == 0.7


def test_parse_input_file_accepts_skip_structure_quality_mode(tmp_path: Path):
    """Allow suspicious intermediates to be bridged between clean states."""

    input_file = tmp_path / "lmplyz.inp"
    input_file.write_text(
        'element_list = ["C", "H"]\nstructure_quality_mode = skip\nBF1 = bonds.reax\n',
        encoding="utf-8",
    )

    assert parse_input_file(input_file).structure_quality_mode == "skip"


@pytest.mark.parametrize(
    ("setting", "message"),
    [
        ("bond_state_persistence_frames = 0", "must be at least 1"),
        ("bond_state_persistence_timesteps = -1", "must be at least 0"),
        ("structure_quality_mode = discard", "must be one of"),
    ],
)
def test_parse_input_file_rejects_invalid_analysis_settings(tmp_path: Path, setting: str, message: str):
    """Reject invalid temporal and structure-quality options."""

    input_file = tmp_path / "lmplyz.inp"
    input_file.write_text(
        f'element_list = ["C", "H"]\n{setting}\nBF1 = bonds.reax\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        parse_input_file(input_file)
