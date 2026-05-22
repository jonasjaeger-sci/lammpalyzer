from pathlib import Path

from lammpalyze.config import parse_input_file


def test_parse_input_file_groups_simulations(tmp_path: Path):
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
