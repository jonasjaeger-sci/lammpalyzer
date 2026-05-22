from pathlib import Path

from lammpalyze.analysis import LoadedSimulation
from lammpalyze.ovito import (
    _find_ovito_executable,
    create_reaction_scene,
    launch_ovito_scene,
    normalize_reaction_path,
)
from lammpalyze.reactions import ReactionOccurrence


def test_normalize_reaction_path_accepts_paths_out_row():
    row = "['AB'] -> ['A', 'B']\t3"

    assert normalize_reaction_path(row) == "['AB'] -> ['A', 'B']"


def test_create_reaction_scene_writes_ball_and_stick_data_file(tmp_path: Path):
    trajectory = tmp_path / "traj.lammpstrj"
    trajectory.write_text(
        """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS id type q xu yu zu
1 1 0.0 1.0 2.0 3.0
2 1 0.0 12.0 2.0 3.0
ITEM: TIMESTEP
1
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS id type q xu yu zu
1 1 0.0 4.0 5.0 6.0
2 1 0.0 5.0 5.0 6.0
""",
        encoding="utf-8",
    )
    bond_file = tmp_path / "bonds.reax"
    bond_file.write_text(
        """# Timestep 0
#
# Number of particles 2
#
1 1 1 2 0 1.0 0 0 0
2 1 1 1 0 1.0 0 0 0
# Timestep 1
#
# Number of particles 2
#
1 1 0 0 0 0
2 1 0 0 0 0
""",
        encoding="utf-8",
    )
    simulation = LoadedSimulation(index=1, trajectory_path=trajectory, bond_path=bond_file, type_to_element={1: "H"})
    occurrence = ReactionOccurrence(
        reaction="['AB'] -> ['A', 'B']",
        timestep_reactants=0,
        timestep_products=1,
        reactants=["AB"],
        products=["A", "B"],
        reactant_atom_ids=["1", "2"],
        product_atom_ids=["1"],
        simulation_index=1,
    )

    scene = create_reaction_scene(simulation, occurrence, output_dir=tmp_path / "scene")

    assert scene.dump_file.exists()
    assert scene.info_file.exists()
    dump_text = scene.dump_file.read_text(encoding="utf-8")
    assert "4 atoms\n" in dump_text
    assert "1 bonds\n" in dump_text
    assert "-9.75000000 15.75000000 xlo xhi" in dump_text
    assert "Atoms # sphere" in dump_text
    assert "1 2 0.6000 1.0000 -5.75000000 2.00000000 3.00000000" in dump_text
    assert "Bonds" in dump_text
    assert "1 1 1 2" in dump_text
    assert "1 1.000000 # X" in dump_text
    assert "2 1.008000 # H" in dump_text
    info_text = scene.info_file.read_text(encoding="utf-8")
    assert "['AB'] -> ['A', 'B']" in info_text
    assert "Bonds section" in info_text


def test_find_ovito_executable_accepts_mixed_case_env(monkeypatch):
    monkeypatch.setenv("OVITO_bin", "/custom/ovito")

    assert _find_ovito_executable() == "/custom/ovito"


def test_launch_ovito_scene_opens_dump_without_script_flag(monkeypatch, tmp_path: Path):
    calls = []

    class DummyProcess:
        pass

    def fake_popen(command):
        calls.append(command)
        return DummyProcess()

    data_file = tmp_path / "scene.data"
    info_file = tmp_path / "scene.txt"
    data_file.write_text("", encoding="utf-8")
    info_file.write_text("", encoding="utf-8")

    monkeypatch.setattr("subprocess.Popen", fake_popen)

    launch_ovito_scene(
        scene=type("Scene", (), {"data_file": data_file, "info_file": info_file})(),
        ovito_executable="/usr/bin/ovito",
    )

    assert calls == [["/usr/bin/ovito", str(data_file)]]
