"""Command-line entry point for lammpalyze."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from lammpalyze.analysis import load_project
from lammpalyze.config import parse_input_file
from lammpalyze.reactions import write_reaction_paths


def build_parser() -> argparse.ArgumentParser:
    """Build the lammpalyze argument parser."""

    parser = argparse.ArgumentParser(description="Analyze LAMMPS/ReaxFF output files.")
    parser.add_argument("-i", "--input", required=True, help="Path to lammpalyze input file.")
    parser.add_argument(
        "-o",
        "--output",
        default="paths.out",
        help="Reaction path output file. Default: paths.out",
    )
    gui_group = parser.add_mutually_exclusive_group()
    gui_group.add_argument("--gui", action="store_true", help="Open the graphical interface after loading data.")
    gui_group.add_argument("--no-gui", action="store_true", help="Do not open the graphical interface.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the lammpalyze CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = parse_input_file(args.input)
        project = load_project(config)
        paths = project.reaction_paths()
        output_path = write_reaction_paths(paths, Path(args.output))
        print(f"Loaded {len(project.simulations)} simulation(s).")
        print(f"Wrote {len(paths)} reaction path(s) to {output_path}.")

        if _should_launch_gui(args.gui, args.no_gui):
            from lammpalyze.gui import launch_gui

            launch_gui(project)
        return 0
    except Exception as exc:
        print(f"lammpalyze: error: {exc}", file=sys.stderr)
        return 1


def _should_launch_gui(force_gui: bool, no_gui: bool) -> bool:
    if no_gui:
        return False
    if force_gui:
        return True
    return bool(os.environ.get("DISPLAY") or sys.platform == "win32")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
