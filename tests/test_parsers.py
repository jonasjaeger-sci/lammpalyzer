from pathlib import Path

from lammpalyze.parsers import eval_species, eval_thermo


def test_eval_species_handles_changing_headers(tmp_path: Path):
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
