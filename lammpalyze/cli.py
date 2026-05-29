"""Command-line entry point for lammpalyze."""

from __future__ import annotations

import argparse
import datetime
import os
import sys
from pathlib import Path

from lammpalyze.analysis import load_project
from lammpalyze.config import parse_input_file
from lammpalyze.info import LOGO, PROGRAM_NAME, VERSION
from lammpalyze.reactions import write_reaction_paths_csv

_DATE_FMT = "%d.%m.%Y %H:%M:%S"


def build_parser() -> argparse.ArgumentParser:
    """Build the lammpalyze argument parser."""

    parser = argparse.ArgumentParser(description="Analyze LAMMPS/ReaxFF output files.")
    parser.add_argument("-i", "--input", required=True, help="Path to lammpalyze input file.")
    parser.add_argument(
        "-o",
        "--output",
        default="paths.csv",
        help="Reaction path CSV output file. Default: paths.csv",
    )
    gui_group = parser.add_mutually_exclusive_group()
    gui_group.add_argument("--gui", action="store_true", help="Open the graphical interface after loading data.")
    gui_group.add_argument("--no-gui", action="store_true", help="Do not open the graphical interface.")
    return parser


def hello_world() -> None:
    """Print program banner and startup metadata."""

    timestart = datetime.datetime.now().strftime(_DATE_FMT)
    pyversion = sys.version.split()[0]
    print(LOGO, flush=True)
    print(f"{PROGRAM_NAME} version: {VERSION}", flush=True)
    print(f"Start of execution: {timestart}", flush=True)
    print(f"Python version: {pyversion}", flush=True)


def bye_world() -> None:
    """Print program shutdown timestamp."""

    timeend = datetime.datetime.now().strftime(_DATE_FMT)
    print(f"End of {PROGRAM_NAME} execution: {timeend}", flush=True)


def main(argv: list[str] | None = None) -> int:
    """Run the lammpalyze CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    hello_world()

    try:
        config = parse_input_file(args.input)
        project = load_project(config)
        simulation_indices, paths, counts_by_reaction = project.reaction_path_table()
        output_path = write_reaction_paths_csv(
            paths,
            Path(args.output),
            simulation_indices=simulation_indices,
            counts_by_reaction=counts_by_reaction,
            metadata={
                "input_file": config.input_file,
                "run_date": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                "simulation_ids": simulation_indices,
                "software_version": VERSION,
            },
        )
        print(f"Loaded {len(project.simulations)} simulation(s).")
        print(f"Wrote {len(paths)} reaction path(s) to {output_path}.")

        if _should_launch_gui(args.gui, args.no_gui):
            from lammpalyze.gui import launch_gui

            launch_gui(project)
        exit_code = 0
    except Exception as exc:
        print(f"lammpalyze: error: {exc}", file=sys.stderr)
        exit_code = 1
    finally:
        bye_world()
    return exit_code


def _should_launch_gui(force_gui: bool, no_gui: bool) -> bool:
    """Return whether the GUI should open for the requested CLI flags."""

    if no_gui:
        return False
    if force_gui:
        return True
    return bool(os.environ.get("DISPLAY") or sys.platform == "win32")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
