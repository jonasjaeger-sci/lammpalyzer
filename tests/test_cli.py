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

    def fake_load_project(loaded_config):
        """Record the loaded config and return a fake project."""

        calls["config"] = loaded_config
        return project

    def fake_write_reaction_paths_csv(written_paths, target_path, **kwargs):
        """Record CSV writer arguments and return the requested path."""

        calls["paths"] = written_paths
        calls["target_path"] = target_path
        calls["writer_kwargs"] = kwargs
        return target_path

    monkeypatch.setattr(cli, "parse_input_file", fake_parse_input_file)
    monkeypatch.setattr(cli, "load_project", fake_load_project)
    monkeypatch.setattr(cli, "write_reaction_paths_csv", fake_write_reaction_paths_csv)

    exit_code = cli.main(["-i", "lmplyz.inp", "--no-gui", "-o", str(output_path)])

    assert exit_code == 0
    assert calls == {
        "input_path": "lmplyz.inp",
        "config": config,
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
