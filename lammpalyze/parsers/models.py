"""Shared parser data models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrajectoryAtom:
    """One atom entry from a LAMMPS trajectory frame."""

    atom_id: int
    atom_type: int
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class TrajectoryFrame:
    """A LAMMPS trajectory frame with bounds and atom positions."""

    timestep: int
    bounds: np.ndarray
    atoms: list[TrajectoryAtom]


@dataclass(frozen=True)
class ReaxBond:
    """One bond from a ReaxFF bonds frame."""

    atom_i: int
    atom_j: int
    order: float
