"""Tabular atom and molecule snapshots from analyzed trajectory states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from lammpalyze.analysis import LoadedSimulation

SnapshotNotation = Literal["formula", "smiles"]


@dataclass(frozen=True)
class SnapshotColumn:
    """One column in a snapshot table."""

    key: str
    heading: str


@dataclass(frozen=True)
class SnapshotTable:
    """GUI-independent table data for one selected system state."""

    columns: tuple[SnapshotColumn, ...]
    rows: tuple[tuple[str, ...], ...]
    observation_timesteps: tuple[int, ...]


def is_snapshot_notation_column(column_key: str) -> bool:
    """Return whether a snapshot column contains a copyable formula or SMILES."""

    return column_key == "molecule" or column_key.startswith("state_")


def snapshot_timestep_window(
    timesteps: list[int] | tuple[int, ...],
    selected_timestep: int,
    radius: int = 5,
) -> list[int]:
    """Return the selected observation plus up to ``radius`` observations on each side."""

    if radius < 0:
        raise ValueError("Snapshot timestep radius must not be negative.")
    ordered = sorted(set(timesteps))
    try:
        selected_index = ordered.index(selected_timestep)
    except ValueError as exc:
        raise ValueError(
            f"Timestep {selected_timestep} has no bond-derived molecule observation."
        ) from exc
    start = max(0, selected_index - radius)
    end = selected_index + radius + 1
    return ordered[start:end]


def atom_snapshot_table(
    simulation: LoadedSimulation,
    selected_timestep: int,
    notation: SnapshotNotation = "formula",
    radius: int = 5,
) -> SnapshotTable:
    """Return atom types, calculated component IDs, and nearby molecule states."""

    molecule_values = _molecule_values(simulation, notation)
    observation_timesteps = snapshot_timestep_window(
        list(simulation.smiles_id or {}), selected_timestep, radius
    )
    frame = simulation.read_trajectory_frame(selected_timestep)
    state_by_timestep = {
        timestep: _atom_component_states(simulation, timestep, molecule_values)
        for timestep in observation_timesteps
    }
    selected_states = state_by_timestep[selected_timestep]

    columns = [
        SnapshotColumn("atom_id", "Atom ID"),
        SnapshotColumn("atom_type", "Atom type"),
        SnapshotColumn("mol_id", "Calculated mol_id"),
    ]
    columns.extend(
        SnapshotColumn(
            f"state_{timestep}",
            f"{timestep} (selected)" if timestep == selected_timestep else str(timestep),
        )
        for timestep in observation_timesteps
    )

    rows = []
    for atom in sorted(frame.atoms, key=lambda value: value.atom_id):
        selected_state = selected_states.get(atom.atom_id)
        row = [
            str(atom.atom_id),
            str(atom.atom_type),
            str(selected_state[0]) if selected_state is not None else "—",
        ]
        row.extend(
            state_by_timestep[timestep].get(atom.atom_id, (None, "—"))[1]
            for timestep in observation_timesteps
        )
        rows.append(tuple(row))
    return SnapshotTable(tuple(columns), tuple(rows), tuple(observation_timesteps))


def molecule_snapshot_table(
    simulation: LoadedSimulation,
    selected_timestep: int,
    notation: SnapshotNotation = "formula",
) -> SnapshotTable:
    """Return calculated component IDs, molecule labels, and constituent atoms."""

    molecule_values = _molecule_values(simulation, notation)
    components = (simulation.smiles_id or {}).get(selected_timestep)
    if components is None:
        raise ValueError(
            f"Timestep {selected_timestep} has no bond-derived molecule observation."
        )
    labels = molecule_values.get(selected_timestep)
    if labels is None:
        raise ValueError(
            f"Simulation {simulation.index} has no {notation} labels at timestep "
            f"{selected_timestep}."
        )

    notation_heading = "Chemical formula" if notation == "formula" else "SMILES"
    columns = (
        SnapshotColumn("mol_id", "Calculated mol_id"),
        SnapshotColumn("molecule", notation_heading),
        SnapshotColumn("atoms", "Constituent atom IDs"),
    )
    rows = tuple(
        (
            str(component_index),
            labels[component_index] if component_index < len(labels) else "—",
            f"[{', '.join(str(atom_id) for atom_id in component_atoms)}]",
        )
        for component_index, component_atoms in enumerate(components)
    )
    return SnapshotTable(columns, rows, (selected_timestep,))


def _molecule_values(
    simulation: LoadedSimulation,
    notation: SnapshotNotation,
) -> dict[int, list[str]]:
    """Return validated formula or SMILES observations."""

    normalized_notation = notation.strip().lower()
    if normalized_notation not in {"formula", "smiles"}:
        raise ValueError("Snapshot notation must be 'formula' or 'smiles'.")
    if not simulation.smiles_id:
        raise ValueError(f"Simulation {simulation.index} has no bond-derived molecule data.")
    molecule_values = (
        simulation.chem_formulas if normalized_notation == "formula" else simulation.smiles
    )
    if not molecule_values:
        raise ValueError(
            f"Simulation {simulation.index} has no bond-derived {normalized_notation} data."
        )
    return molecule_values


def _atom_component_states(
    simulation: LoadedSimulation,
    timestep: int,
    molecule_values: dict[int, list[str]],
) -> dict[int, tuple[int, str]]:
    """Map atom IDs to calculated component indexes and labels at one timestep."""

    components = (simulation.smiles_id or {}).get(timestep, [])
    labels = molecule_values.get(timestep, [])
    states = {}
    for component_index, component_atoms in enumerate(components):
        label = labels[component_index] if component_index < len(labels) else "—"
        for atom_id in component_atoms:
            states[int(atom_id)] = (component_index, label)
    return states
