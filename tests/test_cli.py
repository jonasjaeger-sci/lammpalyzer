"""Tests for the command-line entry point."""

import csv
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from lammpalyze import cli


def test_main_loads_project_writes_paths_and_skips_gui(monkeypatch, tmp_path: Path):
    """Load a project, write paths, and skip GUI startup when requested."""

    calls = {}
    config = SimpleNamespace(name="config", input_file=tmp_path / "lmplyz.inp")
    paths = [SimpleNamespace(reaction="A -> B", count=2)]
    project = SimpleNamespace(
        simulations=[object(), object()],
        reaction_path_table=lambda: ([1, 2], paths, {"A -> B": {1: 2, 2: 0}}),
    )
    output_path = tmp_path / "paths.csv"

    def fake_parse_input_file(input_path):
        """Record the requested input path and return a fake config."""

        calls["input_path"] = input_path
        return config

    def fake_load_project(loaded_config, progress_callback=None):
        """Record the loaded config and return a fake project."""

        calls["config"] = loaded_config
        calls["progress_callback"] = progress_callback
        return project

    def fake_write_reaction_paths_csv(written_paths, target_path, **kwargs):
        """Record CSV writer arguments and return the requested path."""

        calls["paths"] = written_paths
        calls["target_path"] = target_path
        calls["writer_kwargs"] = kwargs
        return target_path

    def fake_validate_config(loaded_config):
        """Record the config passed through preflight validation."""

        calls["validated_config"] = loaded_config

    monkeypatch.setattr(cli, "parse_input_file", fake_parse_input_file)
    monkeypatch.setattr(cli, "validate_config", fake_validate_config)
    monkeypatch.setattr(cli, "load_project", fake_load_project)
    monkeypatch.setattr(cli, "write_reaction_paths_csv", fake_write_reaction_paths_csv)

    exit_code = cli.main(["-i", "lmplyz.inp", "--no-gui", "-o", str(output_path)])

    assert exit_code == 0
    assert calls == {
        "input_path": "lmplyz.inp",
        "validated_config": config,
        "config": config,
        "progress_callback": calls["progress_callback"],
        "paths": paths,
        "target_path": output_path,
        "writer_kwargs": {
            "simulation_indices": [1, 2],
            "counts_by_reaction": {"A -> B": {1: 2, 2: 0}},
            "metadata": {
                "input_file": config.input_file,
                "run_date": calls["writer_kwargs"]["metadata"]["run_date"],
                "simulation_ids": [1, 2],
                "software_version": cli.VERSION,
            },
        },
    }


def test_lammpalyze_example_cli_writes_expected_paths(tmp_path: Path):
    """Run an example through the CLI and assert the generated paths file."""

    repo_root = Path(__file__).resolve().parents[1]
    example_dir = repo_root / "examples" / "example_NVT_vs_NPT"
    output_path = tmp_path / "paths.csv"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lammpalyze.cli",
            "-i",
            str(example_dir / "lmplyz.inp"),
            "--no-gui",
            "-o",
            str(output_path),
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Loaded 2 simulation(s)." in result.stdout
    assert "Wrote 388 reaction path(s)" in result.stdout
    assert b"\r\n" not in output_path.read_bytes()

    rows = list(csv.reader(output_path.open(encoding="utf-8", newline="")))
    metadata = dict(rows[1:5])
    assert rows[0] == ["Metadata", "Value"]
    assert metadata["input_file"] == str((example_dir / "lmplyz.inp").resolve())
    assert metadata["simulation_ids"] == "1;2"
    assert metadata["software_version"] == cli.VERSION

    table = rows[6:]
    assert table[0] == ["Reaction", "Simulation 1", "Simulation 2", "Sum"]
    assert len(table[1:]) == 388
    assert all(int(row[1]) + int(row[2]) == int(row[3]) for row in table[1:])


def test_progress_bar_writes_to_stderr(capsys):
    """The CLI progress bar is intentionally small and terminal-friendly."""

    progress = cli._ProgressBar(enabled=True, width=4)

    progress.update(1, 2, "Loaded simulation 1")
    progress.update(2, 2, "Loaded simulation 2")

    assert "[##--] 1/2 Loaded simulation 1" in capsys.readouterr().err


def test_validate_command_reports_clean_input(tmp_path: Path, capsys):
    """Validate a minimal project without running the expensive parsers."""

    (tmp_path / "traj.lammpstrj").write_text(
        """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS id type x y z
1 1 0.0 0.0 0.0
""",
        encoding="utf-8",
    )
    input_file = tmp_path / "lmplyz.inp"
    input_file.write_text(
        """
        element_list = ["C"]
        TrajF1 = traj.lammpstrj
        """,
        encoding="utf-8",
    )

    exit_code = cli.main(["validate", "-i", str(input_file)])

    assert exit_code == 0
    assert "OK: Input file passed validation checks." in capsys.readouterr().out


def test_validate_command_reports_preflight_errors(tmp_path: Path, capsys):
    """Surface common input problems before full analysis begins."""

    (tmp_path / "bad_traj.lammpstrj").write_text(
        """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS id type vx vy vz
1 3 0.0 0.0 0.0
""",
        encoding="utf-8",
    )
    input_file = tmp_path / "lmplyz.inp"
    input_file.write_text(
        """
        element_list = ["C", "H"]
        BF1 = missing_bonds.reax
        TrajF2 = bad_traj.lammpstrj
        """,
        encoding="utf-8",
    )

    exit_code = cli.main(["validate", "-i", str(input_file)])
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "ERROR: Simulation 1 bond file is missing" in output
    assert "WARNING: Simulation 1 has bond but is missing trajectory" in output
    assert "ERROR: simulation 2 trajectory file uses atom type(s) 3" in output
    assert "ERROR: Simulation 2 trajectory uses unsupported atom columns" in output
