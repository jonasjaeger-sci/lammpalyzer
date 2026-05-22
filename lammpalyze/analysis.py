"""High-level project loading and numerical analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from lammpalyze.config import LammpalyzeConfig
from lammpalyze.parsers import eval_species, eval_thermo, parse_bonds, parse_traj
from lammpalyze.reactions import ReactionOccurrence, ReactionPath, count_reaction_paths, find_reaction_occurrences


@dataclass
class LoadedSimulation:
    """Parsed data for one simulation run."""

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
        """Whether this simulation has parsed SMILES/formula data."""

        return self.smiles is not None and self.smiles_id is not None and self.chem_formulas is not None

    def iter_trajectory(self):
        """Yield trajectory frames on demand.

        Trajectory files can be very large, so lammpalyze keeps the validated
        path in the project and streams frames instead of loading everything
        into memory during CLI startup.
        """

        if self.trajectory_path is None:
            raise ValueError(f"Simulation {self.index} has no trajectory file.")
        return parse_traj(self.trajectory_path)

    def load_trajectory(self) -> list[np.ndarray]:
        """Load all trajectory frames into memory."""

        return list(self.iter_trajectory())


@dataclass
class LammpalyzeProject:
    """A loaded lammpalyze project containing one or more simulations."""

    config: LammpalyzeConfig
    simulations: list[LoadedSimulation]

    def reaction_paths(self) -> list[ReactionPath]:
        """Return counted reaction paths across all simulations."""

        all_paths: dict[str, int] = {}
        for simulation in self.simulations:
            if simulation.smiles is None or simulation.smiles_id is None:
                continue
            for path in count_reaction_paths(simulation.smiles, simulation.smiles_id):
                all_paths[path.reaction] = all_paths.get(path.reaction, 0) + path.count
        return [
            ReactionPath(reaction, count)
            for reaction, count in sorted(all_paths.items(), key=lambda item: item[1], reverse=True)
        ]

    def first_reaction_occurrence(self, reaction: str) -> tuple[LoadedSimulation, ReactionOccurrence]:
        """Return the first concrete occurrence matching ``reaction``."""

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
        """Return a loaded simulation by its input index."""

        for simulation in self.simulations:
            if simulation.index == index:
                return simulation
        raise KeyError(f"Simulation {index} was not loaded.")


def load_project(config: LammpalyzeConfig) -> LammpalyzeProject:
    """Load all simulation data referenced in ``config``."""

    simulations: list[LoadedSimulation] = []
    for files in config.simulations:
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

    return LammpalyzeProject(config=config, simulations=simulations)


def aggregate_thermo(
    simulations: list[LoadedSimulation],
    parameter: str,
    x_column: str = "Step",
) -> pd.DataFrame:
    """Average a thermodynamic parameter across simulations.

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
