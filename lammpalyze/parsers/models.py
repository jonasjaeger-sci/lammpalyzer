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


@dataclass(frozen=True)
class ChargeStatistics:
    """Atomic partial-charge summary for one element in one bond frame."""

    mean: float
    std: float
    count: int


@dataclass(frozen=True)
class ComponentProperties:
    """Charge and quality information for one parsed molecular component."""

    charge: float
    ion_candidate: str | None
    warnings: tuple[str, ...]

    @property
    def suspicious(self) -> bool:
        """Return whether any structural quality warning was recorded."""

        return bool(self.warnings)


@dataclass(frozen=True)
class BondParseResult:
    """Complete bond parsing result, including optional analysis metadata."""

    atom_evolution: dict[str, list[str]]
    smiles: dict[int, list[str]]
    smiles_atoms: dict[int, list[list[str]]]
    chem_formulas: dict[int, list[str]]
    atom_charges: dict[int, dict[str, float]]
    charge_statistics: dict[int, dict[str, ChargeStatistics]]
    component_properties: dict[int, list[ComponentProperties]]
    excluded_components: dict[int, set[int]]
