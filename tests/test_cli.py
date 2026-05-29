"""Tests for the command-line entry point."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from lammpalyze import cli


def test_main_loads_project_writes_paths_and_skips_gui(monkeypatch, tmp_path: Path):
    """Load a project, write paths, and skip GUI startup when requested."""

    calls = {}
    config = SimpleNamespace(name="config")
    paths = [SimpleNamespace(reaction="A -> B", count=2)]
    project = SimpleNamespace(simulations=[object(), object()], reaction_paths=lambda: paths)
    output_path = tmp_path / "paths.out"

    def fake_parse_input_file(input_path):
        """Record the requested input path and return a fake config."""

        calls["input_path"] = input_path
        return config

    def fake_load_project(loaded_config):
        """Record the loaded config and return a fake project."""

        calls["config"] = loaded_config
        return project

    def fake_write_reaction_paths(written_paths, target_path):
        """Record reaction-path write arguments and return the target path."""

        calls["paths"] = written_paths
        calls["target_path"] = target_path
        return target_path

    monkeypatch.setattr(cli, "parse_input_file", fake_parse_input_file)
    monkeypatch.setattr(cli, "load_project", fake_load_project)
    monkeypatch.setattr(cli, "write_reaction_paths", fake_write_reaction_paths)

    exit_code = cli.main(["-i", "lmplyz.inp", "--no-gui", "-o", str(output_path)])

    assert exit_code == 0
    assert calls == {
        "input_path": "lmplyz.inp",
        "config": config,
        "paths": paths,
        "target_path": output_path,
    }


def test_lammpalyze_example_cli_writes_expected_paths(tmp_path: Path):
    """Run an example through the CLI and assert the generated paths file."""

    repo_root = Path(__file__).resolve().parents[1]
    example_dir = repo_root / "examples" / "example_NVT_vs_NPT"
    output_path = tmp_path / "paths.out"

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
    assert output_path.read_bytes() == (example_dir / "paths.out").read_bytes()
