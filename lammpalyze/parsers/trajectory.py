"""Parsers for LAMMPS trajectory files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from lammpalyze.parsers.models import TrajectoryAtom, TrajectoryFrame


def parse_traj(filename: str | Path) -> Iterator[np.ndarray]:
    """Yield wrapped ``[q, x, y, z]`` arrays from flexible atom tables.

    Missing charge values are represented by ``NaN``. Coordinate columns may
    be wrapped, unwrapped, or scaled coordinates in any table position.
    """

    for frame in iter_lammpstrj_frames(filename):
        coordinates = np.array(
            [
                [atom.x, atom.y, atom.z]
                for atom in frame.atoms
            ],
            dtype=float,
        )
        minimum = frame.bounds[:, 0]
        box_lengths = frame.bounds[:, 1] - minimum
        wrapped = minimum + (coordinates - minimum) % box_lengths
        charges = np.array(
            [atom.values.get("q", np.nan) for atom in frame.atoms],
            dtype=float,
        )
        yield np.column_stack((charges, wrapped))


def list_lammpstrj_timesteps(filename: str | Path) -> list[int]:
    """Return all timesteps present in a LAMMPS trajectory file."""

    return list(index_lammpstrj_frames(filename))


def index_lammpstrj_frames(filename: str | Path) -> dict[int, int]:
    """Return byte offsets for every trajectory timestep."""

    frame_offsets = {}
    with Path(filename).open(encoding="utf-8") as handle:
        n_atoms = 0
        while True:
            line_offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if line.startswith("ITEM: TIMESTEP"):
                frame_offsets[int(handle.readline().strip())] = line_offset
                continue
            if line.startswith("ITEM: NUMBER OF ATOMS"):
                n_atoms = int(handle.readline().strip())
                continue
            if line.startswith("ITEM: ATOMS"):
                for _ in range(n_atoms):
                    handle.readline()
    return frame_offsets


def trajectory_atom_columns(filename: str | Path) -> list[str]:
    """Return the columns declared by the first trajectory atom table."""

    with Path(filename).open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("ITEM: ATOMS"):
                return line.split()[2:]
    raise ValueError(f"No ITEM: ATOMS table found in trajectory file {filename}")


def copy_lammpstrj_until(
    source: str | Path,
    destination: str | Path,
    end_timestep: int,
) -> int:
    """Copy raw trajectory frames from the beginning through ``end_timestep``."""

    frames_written = 0
    with Path(source).open(encoding="utf-8") as input_handle, Path(destination).open(
        "w",
        encoding="utf-8",
    ) as output_handle:
        while True:
            line = input_handle.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue

            timestep_line = input_handle.readline()
            if not timestep_line:
                raise ValueError(f"Malformed trajectory frame in {source}")
            timestep = int(timestep_line.strip())
            frame_lines = [line, timestep_line]

            number_header = input_handle.readline()
            if not number_header.startswith("ITEM: NUMBER OF ATOMS"):
                raise ValueError(f"Malformed trajectory frame at timestep {timestep} in {source}")
            n_atoms_line = input_handle.readline()
            n_atoms = int(n_atoms_line.strip())
            frame_lines.extend([number_header, n_atoms_line])

            bounds_header = input_handle.readline()
            if not bounds_header.startswith("ITEM: BOX BOUNDS"):
                raise ValueError(f"Missing box bounds at timestep {timestep} in {source}")
            frame_lines.append(bounds_header)
            frame_lines.extend(input_handle.readline() for _ in range(3))

            atoms_header = input_handle.readline()
            if not atoms_header.startswith("ITEM: ATOMS"):
                raise ValueError(f"Missing atom table at timestep {timestep} in {source}")
            frame_lines.append(atoms_header)
            frame_lines.extend(input_handle.readline() for _ in range(n_atoms))

            if timestep <= end_timestep:
                output_handle.writelines(frame_lines)
                frames_written += 1
            if timestep >= end_timestep:
                break

    if frames_written == 0:
        raise ValueError(f"No trajectory frames found through timestep {end_timestep} in {source}")
    return frames_written


def iter_lammpstrj_frames(
    filename: str | Path,
    timestep_range: tuple[int, int] | None = None,
) -> Iterator[TrajectoryFrame]:
    """Yield trajectory frames, optionally limited to an inclusive timestep range."""

    with Path(filename).open(encoding="utf-8") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue

            timestep = int(handle.readline().strip())
            number_header = handle.readline()
            if not number_header.startswith("ITEM: NUMBER OF ATOMS"):
                raise ValueError(f"Malformed trajectory frame at timestep {timestep} in {filename}")
            n_atoms = int(handle.readline().strip())

            bounds_header = handle.readline()
            if not bounds_header.startswith("ITEM: BOX BOUNDS"):
                raise ValueError(f"Missing box bounds at timestep {timestep} in {filename}")
            bounds = np.array([[float(value) for value in handle.readline().split()[:2]] for _ in range(3)])

            atoms_header = handle.readline()
            if not atoms_header.startswith("ITEM: ATOMS"):
                raise ValueError(f"Missing atom table at timestep {timestep} in {filename}")
            columns = atoms_header.split()[2:]

            if timestep_range is not None:
                start, end = sorted(timestep_range)
                if timestep < start or timestep > end:
                    for _ in range(n_atoms):
                        handle.readline()
                    continue

            atoms = [
                _trajectory_atom_from_values(columns, handle.readline().split(), bounds)
                for _ in range(n_atoms)
            ]
            yield TrajectoryFrame(timestep=timestep, bounds=bounds, atoms=atoms)


def read_lammpstrj_frame(
    filename: str | Path,
    target_timestep: int,
    frame_offset: int | None = None,
) -> TrajectoryFrame:
    """Read one trajectory frame by timestep for external visualization."""

    with Path(filename).open(encoding="utf-8") as handle:
        if frame_offset is not None:
            handle.seek(frame_offset)
        while True:
            line = handle.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue

            timestep = int(handle.readline().strip())
            number_header = handle.readline()
            if not number_header.startswith("ITEM: NUMBER OF ATOMS"):
                raise ValueError(f"Malformed trajectory frame at timestep {timestep} in {filename}")
            n_atoms = int(handle.readline().strip())

            bounds_header = handle.readline()
            if not bounds_header.startswith("ITEM: BOX BOUNDS"):
                raise ValueError(f"Missing box bounds at timestep {timestep} in {filename}")
            bounds = np.array([[float(value) for value in handle.readline().split()[:2]] for _ in range(3)])

            atoms_header = handle.readline()
            if not atoms_header.startswith("ITEM: ATOMS"):
                raise ValueError(f"Missing atom table at timestep {timestep} in {filename}")
            columns = atoms_header.split()[2:]

            atoms = []
            for _ in range(n_atoms):
                values = handle.readline().split()
                if timestep == target_timestep:
                    atoms.append(_trajectory_atom_from_values(columns, values, bounds))

            if timestep == target_timestep:
                return TrajectoryFrame(timestep=timestep, bounds=bounds, atoms=atoms)
            if frame_offset is not None:
                break

    raise ValueError(f"Timestep {target_timestep} not found in trajectory file {filename}")


def _trajectory_atom_from_values(
    columns: list[str],
    values: list[str],
    bounds: np.ndarray | None = None,
) -> TrajectoryAtom:
    """Build a trajectory atom from one LAMMPS atom-table row."""

    if len(values) < len(columns):
        raise ValueError(
            f"Trajectory atom row has {len(values)} values for {len(columns)} columns."
        )
    column_index = {column: index for index, column in enumerate(columns)}
    x_column = _first_available_column(column_index, ("xu", "x", "xs"))
    y_column = _first_available_column(column_index, ("yu", "y", "ys"))
    z_column = _first_available_column(column_index, ("zu", "z", "zs"))
    coordinates = np.array(
        [
            float(values[column_index[x_column]]),
            float(values[column_index[y_column]]),
            float(values[column_index[z_column]]),
        ]
    )
    if bounds is not None and (x_column, y_column, z_column) == ("xs", "ys", "zs"):
        coordinates = bounds[:, 0] + coordinates * (bounds[:, 1] - bounds[:, 0])
    numeric_values = {}
    for column, index in column_index.items():
        try:
            numeric_values[column] = float(values[index])
        except ValueError:
            continue
    element_index = column_index.get("element")
    return TrajectoryAtom(
        atom_id=int(float(values[column_index["id"]])),
        atom_type=int(float(values[column_index.get("type", column_index["id"])])),
        x=float(coordinates[0]),
        y=float(coordinates[1]),
        z=float(coordinates[2]),
        element=values[element_index] if element_index is not None else None,
        values=numeric_values,
    )


def _first_available_column(column_index: dict[str, int], candidates: tuple[str, ...]) -> str:
    """Return the first candidate column present in the atom-table header."""

    for column in candidates:
        if column in column_index:
            return column
    raise ValueError(f"Trajectory atom table lacks coordinate columns {candidates}")
