"""Load configured LAMMPS outputs and expose project-level analysis helpers."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from lammpalyze.config import LammpalyzeConfig
from lammpalyze.parsers import eval_species, eval_thermo, parse_bonds, parse_traj
from lammpalyze.reactions import ReactionOccurrence, ReactionPath, build_reaction_path_table, find_reaction_occurrences

LOGGER = logging.getLogger(__name__)
ProgressCallback = Callable[[int, int, str], None]


@dataclass
class LoadedSimulation:
    """Container for the data parsed from one replica or simulation run."""

    index: int
    species: list[str] | None = None
    species_df: pd.DataFrame | None = None
    thermo_df: pd.DataFrame | None = None
    atom_evolution: dict[str, list[str]] | None = None
    smiles: dict[int, list[str]] | None = None
    smiles_id: dict[int, list[list[str]]] | None = None
    chem_formulas: dict[int, list[str]] | None = None
    trajectory_path: Path | None = None
    bond_path: Path | None = None
    type_to_element: dict[int, str] | None = None

    @property
    def has_bond_data(self) -> bool:
        """Check whether reaction-related bond parsing has been completed."""

        return self.smiles is not None and self.smiles_id is not None and self.chem_formulas is not None

    def iter_trajectory(self):
        """Stream trajectory frames from disk when a caller actually needs them.

        Trajectory files can be very large, so lammpalyze keeps the validated
        path in the project and streams frames instead of loading everything
        into memory during CLI startup.
        """

        if self.trajectory_path is None:
            raise ValueError(f"Simulation {self.index} has no trajectory file.")
        return parse_traj(self.trajectory_path)

    def load_trajectory(self) -> list[np.ndarray]:
        """Read every trajectory frame into memory for small-file workflows."""

        return list(self.iter_trajectory())


@dataclass
class LammpalyzeProject:
    """A loaded analysis session built from one lammpalyze input file."""

    config: LammpalyzeConfig
    simulations: list[LoadedSimulation]

    def reaction_paths(self) -> list[ReactionPath]:
        """Collapse the per-run reaction counts into one ranked list."""

        _, paths, _ = self.reaction_path_table()
        return paths

    def reaction_path_table(self) -> tuple[list[int], list[ReactionPath], dict[str, dict[int, int]]]:
        """Prepare reaction counts in the same shape used by the GUI table."""

        return build_reaction_path_table(self.simulations)

    def first_reaction_occurrence(self, reaction: str) -> tuple[LoadedSimulation, ReactionOccurrence]:
        """Find a concrete event for a reaction path, scanning runs in order."""

        for simulation in self.simulations:
            if simulation.smiles is None or simulation.smiles_id is None:
                continue
            occurrences = find_reaction_occurrences(
                simulation.smiles,
                simulation.smiles_id,
                reaction_filter=reaction,
                first_only=True,
                simulation_index=simulation.index,
            )
            if occurrences:
                return simulation, occurrences[0]
        raise ValueError(f"No occurrence found for reaction path: {reaction}")

    def simulation(self, index: int) -> LoadedSimulation:
        """Look up one loaded simulation by the index used in ``lmplyz.inp``."""

        for simulation in self.simulations:
            if simulation.index == index:
                return simulation
        raise KeyError(f"Simulation {index} was not loaded.")


def load_project(
    config: LammpalyzeConfig,
    progress_callback: ProgressCallback | None = None,
) -> LammpalyzeProject:
    """Parse every file referenced by ``config`` and assemble a project object."""

    simulations: list[LoadedSimulation] = []
    total = len(config.simulations)
    for position, files in enumerate(config.simulations, start=1):
        if progress_callback is not None:
            progress_callback(position - 1, total, f"Loading simulation {files.index}")
        LOGGER.info("Loading simulation %s", files.index)

        loaded = LoadedSimulation(index=files.index)
        loaded.bond_path = files.bond
        loaded.trajectory_path = files.trajectory
        loaded.type_to_element = config.type_to_element

        if files.species is not None:
            species, _, species_df = eval_species(files.species)
            loaded.species = species
            loaded.species_df = species_df

        if files.thermo is not None:
            _, loaded.thermo_df = eval_thermo(files.thermo)

        if files.bond is not None:
            atom_evolution, smiles, smiles_id, chem_formulas = parse_bonds(
                files.bond,
                config.type_to_element,
            )
            loaded.atom_evolution = atom_evolution
            loaded.smiles = smiles
            loaded.smiles_id = smiles_id
            loaded.chem_formulas = chem_formulas

        simulations.append(loaded)
        if progress_callback is not None:
            progress_callback(position, total, f"Loaded simulation {files.index}")

    return LammpalyzeProject(config=config, simulations=simulations)


def aggregate_thermo(
    simulations: list[LoadedSimulation],
    parameter: str,
    x_column: str = "Step",
) -> pd.DataFrame:
    """Align and average one thermodynamic column across compatible runs.

    The returned frame contains ``x_column``, ``mean``, and ``std`` columns.
    Values are aligned on the x-axis column before averaging.
    """

    series_by_simulation = []
    for simulation in simulations:
        if simulation.thermo_df is None:
            continue
        if parameter not in simulation.thermo_df.columns:
            continue
        if x_column not in simulation.thermo_df.columns:
            raise ValueError(f"Thermo data for simulation {simulation.index} lacks {x_column!r}.")
        series = simulation.thermo_df[[x_column, parameter]].rename(
            columns={parameter: f"sim_{simulation.index}"}
        )
        series_by_simulation.append(series)

    if not series_by_simulation:
        raise ValueError(f"No thermo data found for parameter {parameter!r}.")

    merged = series_by_simulation[0]
    for series in series_by_simulation[1:]:
        merged = merged.merge(series, on=x_column, how="inner")

    value_columns = [column for column in merged.columns if column != x_column]
    result = pd.DataFrame(
        {
            x_column: merged[x_column],
            "mean": merged[value_columns].mean(axis=1),
            "std": merged[value_columns].std(axis=1).fillna(0.0),
        }
    )
    return result
