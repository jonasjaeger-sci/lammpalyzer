"""Load configured LAMMPS outputs and expose project-level analysis helpers."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from lammpalyze.config import DEFAULT_BOUNDARY, LammpalyzeConfig
from lammpalyze.parsers import (
    ChargeStatistics,
    ComponentProperties,
    ReaxBond,
    TrajectoryFrame,
    eval_msd,
    eval_pairwise_dump,
    eval_species,
    eval_thermo,
    index_lammpstrj_frames,
    index_reax_bond_frames,
    parse_bond_observations,
    parse_traj,
    read_lammpstrj_frame,
    read_reax_bonds_frame,
)
from lammpalyze.reactions import (
    ConnectedReactionPathway,
    ConnectedReactionOccurrence,
    ConnectedReactionStep,
    PathwayEdgeData,
    ReactionOccurrence,
    ReactionPath,
    build_connected_reaction_pathways_from_edge_data,
    build_reaction_path_table,
    collect_connected_pathway_edge_data,
    find_reaction_occurrences,
    index_connected_reaction_occurrences,
)

LOGGER = logging.getLogger(__name__)
ProgressCallback = Callable[[int, int, str], None]


@dataclass
class LoadedSimulation:
    """Container for the data parsed from one replica or simulation run."""

    index: int
    species: list[str] | None = None
    species_df: pd.DataFrame | None = None
    thermo_df: pd.DataFrame | None = None
    pairwise_df: pd.DataFrame | None = None
    msd_df: pd.DataFrame | None = None
    atom_evolution: dict[str, list[str]] | None = None
    smiles: dict[int, list[str]] | None = None
    smiles_id: dict[int, list[list[str]]] | None = None
    chem_formulas: dict[int, list[str]] | None = None
    atom_charges: dict[int, dict[str, float]] | None = None
    charge_statistics: dict[int, dict[str, ChargeStatistics]] | None = None
    component_properties: dict[int, list[ComponentProperties]] | None = None
    excluded_components: dict[int, set[int]] | None = None
    structure_quality_mode: str = "flag"
    trajectory_path: Path | None = None
    bond_path: Path | None = None
    type_to_element: dict[int, str] | None = None
    boundary: tuple[str, str, str] = DEFAULT_BOUNDARY
    _trajectory_frame_offsets: dict[int, int] | None = field(default=None, init=False, repr=False)
    _bond_frame_offsets: dict[int, int] | None = field(default=None, init=False, repr=False)

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

    def trajectory_timesteps(self) -> list[int]:
        """Return trajectory timesteps, scanning the file at most once."""

        if self._trajectory_frame_offsets is None:
            if self.trajectory_path is None:
                self._trajectory_frame_offsets = {}
            else:
                self._trajectory_frame_offsets = index_lammpstrj_frames(self.trajectory_path)
        return list(self._trajectory_frame_offsets)

    def read_trajectory_frame(self, timestep: int) -> TrajectoryFrame:
        """Read one trajectory frame using the cached file index."""

        if self.trajectory_path is None:
            raise ValueError(f"Simulation {self.index} has no trajectory file.")
        self.trajectory_timesteps()
        frame_offset = (self._trajectory_frame_offsets or {}).get(timestep)
        return read_lammpstrj_frame(self.trajectory_path, timestep, frame_offset=frame_offset)

    def read_bond_frame(self, timestep: int) -> list[ReaxBond]:
        """Read one bond frame using offsets captured during project loading."""

        if self.bond_path is None:
            raise ValueError(f"Simulation {self.index} has no ReaxFF bond file.")
        if self._bond_frame_offsets is None:
            self._bond_frame_offsets = index_reax_bond_frames(self.bond_path)
        return read_reax_bonds_frame(
            self.bond_path,
            timestep,
            frame_offset=(self._bond_frame_offsets or {}).get(timestep),
        )


@dataclass
class LammpalyzeProject:
    """A loaded analysis session built from one lammpalyze input file."""

    config: LammpalyzeConfig
    simulations: list[LoadedSimulation]
    _reaction_path_table_cache: (
        tuple[list[int], list[ReactionPath], dict[str, dict[int, int]]] | None
    ) = field(default=None, init=False, repr=False)
    _connected_pathway_cache: dict[
        tuple[str, int],
        list[ConnectedReactionPathway],
    ] = field(default_factory=dict, init=False, repr=False)
    _connected_pathway_edge_cache: dict[str, PathwayEdgeData] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _first_reaction_occurrences_cache: (
        dict[str, tuple[LoadedSimulation, ReactionOccurrence]] | None
    ) = field(default=None, init=False, repr=False)
    _connected_occurrence_index_cache: dict[
        str,
        dict[int, dict[tuple[str, str], ReactionOccurrence]],
    ] = field(default_factory=dict, init=False, repr=False)

    def reaction_paths(self) -> list[ReactionPath]:
        """Collapse the per-run reaction counts into one ranked list."""

        _, paths, _ = self.reaction_path_table()
        return paths

    def reaction_path_table(self) -> tuple[list[int], list[ReactionPath], dict[str, dict[int, int]]]:
        """Prepare reaction counts in the same shape used by the GUI table."""

        if self._reaction_path_table_cache is None:
            self._reaction_path_table_cache = build_reaction_path_table(self.simulations)
        return self._reaction_path_table_cache

    def connected_reaction_pathways(
        self,
        notation: str = "formula",
        min_count: int = 1,
    ) -> list[ConnectedReactionPathway]:
        """Prepare connected reaction-state pathways for GUI display."""

        if notation not in {"formula", "smiles"}:
            raise ValueError("notation must be 'formula' or 'smiles'.")
        if min_count < 1:
            raise ValueError("min_count must be at least 1.")
        cache_key = (notation, min_count)
        if cache_key not in self._connected_pathway_cache:
            if notation not in self._connected_pathway_edge_cache:
                self._connected_pathway_edge_cache[notation] = (
                    collect_connected_pathway_edge_data(self.simulations, notation)
                )
            self._connected_pathway_cache[cache_key] = (
                build_connected_reaction_pathways_from_edge_data(
                    self._connected_pathway_edge_cache[notation],
                    min_count=min_count,
                )
            )
        return self._connected_pathway_cache[cache_key]

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
                excluded_components=simulation.excluded_components,
                quality_mode=simulation.structure_quality_mode,
            )
            if occurrences:
                return simulation, occurrences[0]
        raise ValueError(f"No occurrence found for reaction path: {reaction}")

    def first_reaction_occurrences(self) -> dict[str, tuple[LoadedSimulation, ReactionOccurrence]]:
        """Return the first concrete event for each observed reaction path."""

        if self._first_reaction_occurrences_cache is not None:
            return self._first_reaction_occurrences_cache
        occurrences_by_reaction = {}
        for simulation in self.simulations:
            if simulation.smiles is None or simulation.smiles_id is None:
                continue
            for occurrence in find_reaction_occurrences(
                simulation.smiles,
                simulation.smiles_id,
                simulation_index=simulation.index,
                excluded_components=simulation.excluded_components,
                quality_mode=simulation.structure_quality_mode,
            ):
                occurrences_by_reaction.setdefault(occurrence.reaction, (simulation, occurrence))
        self._first_reaction_occurrences_cache = occurrences_by_reaction
        return self._first_reaction_occurrences_cache

    def first_connected_reaction_occurrence(
        self,
        step: ConnectedReactionStep,
        notation: str = "formula",
    ) -> tuple[LoadedSimulation, ConnectedReactionOccurrence]:
        """Find a concrete event matching one displayed connected pathway step."""

        if notation not in self._connected_occurrence_index_cache:
            self._connected_occurrence_index_cache[notation] = {
                simulation.index: index_connected_reaction_occurrences(
                    simulation,
                    notation=notation,
                )
                for simulation in self.simulations
            }
        simulation_indexes = set(step.simulations)
        for simulation in self.simulations:
            if simulation_indexes and simulation.index not in simulation_indexes:
                continue
            occurrence_index = self._connected_occurrence_index_cache[notation][simulation.index]
            occurrence = occurrence_index.get((step.source, step.target))
            direction = "forward"
            if occurrence is None and step.arrow == "<->":
                occurrence = occurrence_index.get((step.target, step.source))
                direction = "reverse"
            if occurrence is not None:
                return simulation, ConnectedReactionOccurrence(
                    step=step,
                    occurrence=occurrence,
                    matched_direction=direction,
                )
        raise ValueError(
            f"No occurrence found for connected pathway {step.label}: "
            f"{step.source} {step.arrow} {step.target}"
        )

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
        loaded.structure_quality_mode = config.structure_quality_mode
        loaded.boundary = config.boundary

        if files.species is not None:
            species, _, species_df = eval_species(files.species)
            loaded.species = species
            loaded.species_df = species_df

        if files.thermo is not None:
            _, loaded.thermo_df = eval_thermo(files.thermo)

        if files.pairwise is not None:
            loaded.pairwise_df = eval_pairwise_dump(files.pairwise)

        if files.msd is not None:
            loaded.msd_df = eval_msd(files.msd)

        if files.bond is not None:
            bond_result = parse_bond_observations(
                files.bond,
                config.type_to_element,
                default_bond_order_cutoff=config.default_bond_order_cutoff,
                bond_order_cutoffs=config.bond_order_cutoffs,
                bond_state_persistence_frames=config.bond_state_persistence_frames,
                bond_state_persistence_timesteps=config.bond_state_persistence_timesteps,
                bond_order_hysteresis=config.bond_order_hysteresis,
                structure_quality_mode=config.structure_quality_mode,
                ion_charge_threshold=config.ion_charge_threshold,
            )
            loaded.atom_evolution = bond_result.atom_evolution
            loaded.smiles = bond_result.smiles
            loaded.smiles_id = bond_result.smiles_atoms
            loaded.chem_formulas = bond_result.chem_formulas
            loaded.atom_charges = bond_result.atom_charges
            loaded.charge_statistics = bond_result.charge_statistics
            loaded.component_properties = bond_result.component_properties
            loaded.excluded_components = bond_result.excluded_components
            suspicious_count = sum(
                properties.suspicious
                for frame_properties in bond_result.component_properties.values()
                for properties in frame_properties
            )
            if suspicious_count and config.structure_quality_mode in {"flag", "exclude", "skip"}:
                LOGGER.warning(
                    "Simulation %s contains %s suspicious component observation(s); mode=%s",
                    files.index,
                    suspicious_count,
                    config.structure_quality_mode,
                )

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
