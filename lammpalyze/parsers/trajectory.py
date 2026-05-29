"""Parsers for LAMMPS trajectory files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from lammpalyze.parsers.models import TrajectoryAtom, TrajectoryFrame


def parse_traj(filename: str | Path) -> Iterator[np.ndarray]:
    """Yield wrapped ``[q, x, y, z]`` atom arrays from a LAMMPS trajectory."""

    with Path(filename).open(encoding="utf-8") as handle:
        n_atoms = 0
        min_coords = np.zeros(3)
        box_lengths = np.ones(3)
        while True:
            line = handle.readline()
            if not line:
                break

            if "NUMBER" in line:
                n_atoms = int(handle.readline())

            if "ITEM: BOX BOUNDS" in line:
                bounds = np.array([[float(x) for x in handle.readline().split()] for _ in range(3)])
                min_coords = bounds[:, 0]
                box_lengths = bounds[:, 1] - bounds[:, 0]

            if "ITEM: ATOMS" in line:
                cols = line.split()[2:]
                frame_data = []
                for _ in range(n_atoms):
                    atom_line = handle.readline().split()
                    frame_data.append(
                        [
                            float(atom_line[cols.index("q")]),
                            float(atom_line[cols.index("xu")]),
                            float(atom_line[cols.index("yu")]),
                            float(atom_line[cols.index("zu")]),
                        ]
                    )
                unwrapped = np.array(frame_data)
                wrapped = unwrapped.copy()
                wrapped[:, 1:] = min_coords + (unwrapped[:, 1:] - min_coords) % box_lengths
                yield wrapped


def list_lammpstrj_timesteps(filename: str | Path) -> list[int]:
    """Return all timesteps present in a LAMMPS trajectory file."""

    timesteps = []
    with Path(filename).open(encoding="utf-8") as handle:
        n_atoms = 0
        while True:
            line = handle.readline()
            if not line:
                break
            if line.startswith("ITEM: TIMESTEP"):
                timesteps.append(int(handle.readline().strip()))
                continue
            if line.startswith("ITEM: NUMBER OF ATOMS"):
                n_atoms = int(handle.readline().strip())
                continue
            if line.startswith("ITEM: ATOMS"):
                for _ in range(n_atoms):
                    handle.readline()
    return timesteps


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
                _trajectory_atom_from_values(columns, handle.readline().split())
                for _ in range(n_atoms)
            ]
            yield TrajectoryFrame(timestep=timestep, bounds=bounds, atoms=atoms)


def read_lammpstrj_frame(filename: str | Path, target_timestep: int) -> TrajectoryFrame:
    """Read one trajectory frame by timestep for external visualization."""

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

            atoms = []
            for _ in range(n_atoms):
                values = handle.readline().split()
                if timestep == target_timestep:
                    atoms.append(_trajectory_atom_from_values(columns, values))

            if timestep == target_timestep:
                return TrajectoryFrame(timestep=timestep, bounds=bounds, atoms=atoms)

    raise ValueError(f"Timestep {target_timestep} not found in trajectory file {filename}")


def _trajectory_atom_from_values(columns: list[str], values: list[str]) -> TrajectoryAtom:
    """Build a trajectory atom from one LAMMPS atom-table row."""

    column_index = {column: index for index, column in enumerate(columns)}
    x_column = _first_available_column(column_index, ("xu", "x", "xs"))
    y_column = _first_available_column(column_index, ("yu", "y", "ys"))
    z_column = _first_available_column(column_index, ("zu", "z", "zs"))
    return TrajectoryAtom(
        atom_id=int(float(values[column_index["id"]])),
        atom_type=int(float(values[column_index.get("type", column_index["id"])])),
        x=float(values[column_index[x_column]]),
        y=float(values[column_index[y_column]]),
        z=float(values[column_index[z_column]]),
    )


def _first_available_column(column_index: dict[str, int], candidates: tuple[str, ...]) -> str:
    """Return the first candidate column present in the atom-table header."""

    for column in candidates:
        if column in column_index:
            return column
    raise ValueError(f"Trajectory atom table lacks coordinate columns {candidates}")
