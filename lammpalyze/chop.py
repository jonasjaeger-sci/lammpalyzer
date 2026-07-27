"""Streaming tools for extracting timestep ranges from large LAMMPS files."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Sequence, TextIO


def chop_lammpstrj(
    source: str | Path,
    start_timestep: int,
    end_timestep: int,
    destination: str | Path | None = None,
) -> tuple[Path, int]:
    """Copy complete trajectory frames in the inclusive timestep range."""

    source_path = Path(source)
    destination_path = Path(destination) if destination is not None else default_segment_path(
        source_path,
        start_timestep,
        end_timestep,
    )
    _validate_range(start_timestep, end_timestep)

    frames_written = 0
    saw_frame = False
    preamble: list[str] = []

    with source_path.open(encoding="utf-8", newline="") as input_handle, destination_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_handle:
        while True:
            line = input_handle.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                if not saw_frame:
                    preamble.append(line)
                continue

            saw_frame = True
            timestep_line = input_handle.readline()
            if not timestep_line:
                if frames_written > 0:
                    break
                raise ValueError(f"Missing timestep value in {source_path}")
            timestep = _parse_timestep_value(timestep_line, source_path)
            selected = start_timestep <= timestep <= end_timestep

            if selected and frames_written == 0:
                output_handle.writelines(preamble)
            if selected:
                output_handle.write(line)
                output_handle.write(timestep_line)

            _copy_lammpstrj_frame_body(input_handle, output_handle, source_path, timestep, selected)
            if selected:
                frames_written += 1
            if timestep >= end_timestep:
                break

    if frames_written == 0:
        raise ValueError(
            f"No trajectory frames found from timestep {start_timestep} through {end_timestep} in {source_path}"
        )
    return destination_path, frames_written


def chop_thermo_log(
    source: str | Path,
    start_timestep: int,
    end_timestep: int,
    destination: str | Path | None = None,
) -> tuple[Path, int]:
    """Copy thermo rows whose first column is in the inclusive timestep range."""

    source_path = Path(source)
    destination_path = Path(destination) if destination is not None else default_segment_path(
        source_path,
        start_timestep,
        end_timestep,
    )
    _validate_range(start_timestep, end_timestep)

    rows_written = 0
    seen_numeric_row = False
    leading_headers: list[str] = []
    pending_table_headers: list[str] = []

    with source_path.open(encoding="utf-8", newline="") as input_handle, destination_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_handle:
        for line in input_handle:
            timestep = _first_column_timestep(line)
            if timestep is None:
                if not seen_numeric_row and rows_written == 0:
                    leading_headers.append(line)
                elif _is_thermo_header(line):
                    pending_table_headers.append(line)
                continue

            seen_numeric_row = True
            if start_timestep <= timestep <= end_timestep:
                if rows_written == 0:
                    output_handle.writelines(leading_headers)
                output_handle.writelines(pending_table_headers)
                pending_table_headers.clear()
                output_handle.write(line)
                rows_written += 1

    if rows_written == 0:
        raise ValueError(
            f"No thermo rows found from timestep {start_timestep} through {end_timestep} in {source_path}"
        )
    return destination_path, rows_written


def default_segment_path(source: str | Path, start_timestep: int, end_timestep: int) -> Path:
    """Return ``file_start_end.ext`` beside the input path."""

    source_path = Path(source)
    return source_path.with_name(f"{source_path.stem}_{start_timestep}_{end_timestep}{source_path.suffix}")


def pieceoftraj_main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for extracting trajectory frames."""

    parser = _build_segment_parser("Extract complete frames from a LAMMPS trajectory dump.")
    args = parser.parse_args(_normalize_legacy_end_arg(argv))
    try:
        output_path, frames_written = chop_lammpstrj(args.input, args.start, args.end, args.output)
    except Exception as exc:
        print(f"pieceoftraj: error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {frames_written} trajectory frame(s) to {output_path}")
    return 0


def chopthermo_main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for extracting thermo rows."""

    parser = _build_segment_parser("Extract timestep rows from a LAMMPS thermo log.")
    args = parser.parse_args(_normalize_legacy_end_arg(argv))
    try:
        output_path, rows_written = chop_thermo_log(args.input, args.start, args.end, args.output)
    except Exception as exc:
        print(f"chopthermo: error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {rows_written} thermo row(s) to {output_path}")
    return 0


def _build_segment_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-i", "--input", required=True, help="Input file to slice.")
    parser.add_argument("-s", "--start", required=True, type=int, help="First timestep to include.")
    parser.add_argument("-e", "--end", required=True, type=int, help="Last timestep to include.")
    parser.add_argument(
        "-o",
        "--output",
        help="Output path. Default: inputname_start_end.ext beside the input file.",
    )
    return parser


def _normalize_legacy_end_arg(argv: Sequence[str] | None) -> Sequence[str] | None:
    if argv is None:
        argv = sys.argv[1:]

    normalized: list[str] = []
    for arg in argv:
        if arg.startswith("e-") and arg[2:].isdigit():
            normalized.extend(["-e", arg[2:]])
        elif arg.startswith("e=") and arg[2:].isdigit():
            normalized.extend(["-e", arg[2:]])
        else:
            normalized.append(arg)
    return normalized


def _validate_range(start_timestep: int, end_timestep: int) -> None:
    if start_timestep > end_timestep:
        raise ValueError("start timestep must be less than or equal to end timestep")


def _copy_lammpstrj_frame_body(
    input_handle: TextIO,
    output_handle: TextIO,
    source_path: Path,
    timestep: int,
    selected: bool,
) -> None:
    number_header = _read_required(input_handle, f"Malformed trajectory frame at timestep {timestep} in {source_path}")
    if not number_header.startswith("ITEM: NUMBER OF ATOMS"):
        raise ValueError(f"Malformed trajectory frame at timestep {timestep} in {source_path}")
    n_atoms_line = _read_required(input_handle, f"Missing atom count at timestep {timestep} in {source_path}")
    n_atoms = int(n_atoms_line.strip())
    _write_if_selected(output_handle, selected, number_header, n_atoms_line)

    bounds_header = _read_required(input_handle, f"Missing box bounds at timestep {timestep} in {source_path}")
    if not bounds_header.startswith("ITEM: BOX BOUNDS"):
        raise ValueError(f"Missing box bounds at timestep {timestep} in {source_path}")
    _write_if_selected(output_handle, selected, bounds_header)
    for _ in range(3):
        _write_if_selected(
            output_handle,
            selected,
            _read_required(input_handle, f"Missing box-bound row at timestep {timestep} in {source_path}"),
        )

    atoms_header = _read_required(input_handle, f"Missing atom table at timestep {timestep} in {source_path}")
    if not atoms_header.startswith("ITEM: ATOMS"):
        raise ValueError(f"Missing atom table at timestep {timestep} in {source_path}")
    _write_if_selected(output_handle, selected, atoms_header)
    for _ in range(n_atoms):
        _write_if_selected(
            output_handle,
            selected,
            _read_required(input_handle, f"Missing atom row at timestep {timestep} in {source_path}"),
        )


def _write_if_selected(output_handle: TextIO, selected: bool, *lines: str) -> None:
    if selected:
        output_handle.writelines(lines)


def _read_required(handle: TextIO, message: str) -> str:
    line = handle.readline()
    if not line:
        raise ValueError(message)
    return line


def _parse_timestep_value(line: str, source_path: Path) -> int:
    try:
        return int(line.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid timestep value {line.strip()!r} in {source_path}") from exc


def _first_column_timestep(line: str) -> int | None:
    parts = line.split()
    if not parts:
        return None
    try:
        value = float(parts[0])
    except ValueError:
        return None
    if not math.isfinite(value) or not value.is_integer():
        return None
    return int(value)


def _is_thermo_header(line: str) -> bool:
    parts = line.split()
    return bool(parts) and parts[0] == "Step"
