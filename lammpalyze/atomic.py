"""Streaming analysis of flexible per-atom trajectory data."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pstdev
from typing import TYPE_CHECKING, Iterable

from lammpalyze.parsers import iter_lammpstrj_frames, trajectory_atom_columns

if TYPE_CHECKING:
    from lammpalyze.analysis import LoadedSimulation
    from lammpalyze.parsers import TrajectoryAtom


STRUCTURAL_ATOM_COLUMNS = {
    "id",
    "mol",
    "type",
    "element",
    "x",
    "y",
    "z",
    "xu",
    "yu",
    "zu",
    "xs",
    "ys",
    "zs",
    "ix",
    "iy",
    "iz",
}
PREFERRED_ATOMIC_PROPERTIES = ("q", "vx", "vy", "vz", "v", "fx", "fy", "fz", "f")
DERIVED_COMPONENTS = {
    "v": ("vx", "vy", "vz"),
    "f": ("fx", "fy", "fz"),
}
ATOMIC_PROPERTY_NAMES = {
    "q": "Atomic charge",
    "vx": "Velocity x",
    "vy": "Velocity y",
    "vz": "Velocity z",
    "v": "Velocity magnitude",
    "fx": "Force x",
    "fy": "Force y",
    "fz": "Force z",
    "f": "Force magnitude",
}
ATOMIC_PROPERTY_UNITS = {
    "q": "e",
    "vx": "Angstroms/femtosecond",
    "vy": "Angstroms/femtosecond",
    "vz": "Angstroms/femtosecond",
    "v": "Angstroms/femtosecond",
    "fx": "(kcal/mol)/Angstrom",
    "fy": "(kcal/mol)/Angstrom",
    "fz": "(kcal/mol)/Angstrom",
    "f": "(kcal/mol)/Angstrom",
}
ATOM_ID_TOKEN = re.compile(r"^(\d+)(?:-(\d+))?$")


@dataclass(frozen=True)
class AtomicSeries:
    """One trajectory-derived atomic property series."""

    label: str
    timesteps: list[int]
    means: list[float]
    deviations: list[float]
    counts: list[int]


@dataclass(frozen=True)
class ElementAtomicSeries:
    """Aggregate and individual series collected in one trajectory pass."""

    aggregate: list[AtomicSeries]
    individual: list[AtomicSeries]


def trajectory_atomic_properties(filename) -> list[str]:
    """Return selectable scalar and derived properties in one trajectory."""

    columns = trajectory_atom_columns(filename)
    available = {column for column in columns if column not in STRUCTURAL_ATOM_COLUMNS}
    for derived, components in DERIVED_COMPONENTS.items():
        if all(component in available for component in components):
            available.add(derived)
    preferred = [name for name in PREFERRED_ATOMIC_PROPERTIES if name in available]
    return preferred + sorted(available - set(preferred))


def available_atomic_properties(simulations: Iterable[LoadedSimulation]) -> list[str]:
    """Return the union of selectable properties across trajectory simulations."""

    available = set()
    for simulation in simulations:
        if simulation.trajectory_path is not None:
            available.update(trajectory_atomic_properties(simulation.trajectory_path))
    preferred = [name for name in PREFERRED_ATOMIC_PROPERTIES if name in available]
    return preferred + sorted(available - set(preferred))


def collect_atomic_series(
    simulations: Iterable[LoadedSimulation],
    property_name: str,
    *,
    elements: Iterable[str] | None = None,
    atom_ids: Iterable[int] | None = None,
    step_range: tuple[float, float] | None = None,
) -> list[AtomicSeries]:
    """Stream and aggregate one atomic property by element or atom ID."""

    selected_elements = tuple(dict.fromkeys(elements or ()))
    selected_atom_ids = tuple(dict.fromkeys(atom_ids or ()))
    if bool(selected_elements) == bool(selected_atom_ids):
        raise ValueError("Select either one or more elements or one or more atom IDs.")

    if selected_elements:
        result = []
        for simulation in simulations:
            if simulation.trajectory_path is None:
                continue
            if property_name not in trajectory_atomic_properties(simulation.trajectory_path):
                continue
            result.extend(
                _collect_element_aggregate_series(
                    simulation,
                    property_name,
                    selected_elements,
                    step_range,
                )
            )
        if not result:
            raise ValueError(
                "No atomic observations match the selected property, simulations, atoms, and range."
            )
        return result

    result = []
    for simulation in simulations:
        if simulation.trajectory_path is None:
            continue
        if property_name not in trajectory_atomic_properties(simulation.trajectory_path):
            continue
        selectors = selected_elements or selected_atom_ids
        observations = {selector: [] for selector in selectors}
        for frame in iter_lammpstrj_frames(
            simulation.trajectory_path,
            _trajectory_step_range(step_range),
        ):
            if not _in_step_range(frame.timestep, step_range):
                continue
            grouped_values = {selector: [] for selector in selectors}
            for atom in frame.atoms:
                selector = _atom_selector(atom, simulation, selected_elements, selected_atom_ids)
                if selector not in grouped_values:
                    continue
                value = atomic_property_value(atom, property_name)
                if value is not None:
                    grouped_values[selector].append(value)
            for selector, values in grouped_values.items():
                if values:
                    observations[selector].append(
                        (frame.timestep, fmean(values), pstdev(values), len(values))
                    )

        for selector, values in observations.items():
            if not values:
                continue
            selector_label = str(selector) if selected_elements else f"atom {selector}"
            result.append(
                AtomicSeries(
                    label=f"Simulation {simulation.index} {selector_label}",
                    timesteps=[value[0] for value in values],
                    means=[value[1] for value in values],
                    deviations=[value[2] for value in values],
                    counts=[value[3] for value in values],
                )
            )
    if not result:
        raise ValueError(
            "No atomic observations match the selected property, simulations, atoms, and range."
        )
    return result


def collect_element_atomic_series(
    simulations: Iterable[LoadedSimulation],
    property_name: str,
    elements: Iterable[str],
    *,
    step_range: tuple[float, float] | None = None,
    max_individual_series: int | None = None,
) -> ElementAtomicSeries:
    """Collect element means and matching individual atoms in one pass."""

    selected_elements = tuple(dict.fromkeys(elements))
    if not selected_elements:
        raise ValueError("Select at least one element.")

    aggregate_result = []
    individual_result = []
    for simulation in simulations:
        if simulation.trajectory_path is None:
            continue
        if property_name not in trajectory_atomic_properties(simulation.trajectory_path):
            continue
        aggregate_observations = {element: [] for element in selected_elements}
        individual_observations = {}
        for frame in iter_lammpstrj_frames(
            simulation.trajectory_path,
            _trajectory_step_range(step_range),
        ):
            if not _in_step_range(frame.timestep, step_range):
                continue
            grouped_values = {element: [] for element in selected_elements}
            for atom in frame.atoms:
                element = _atom_element(atom, simulation)
                if element not in grouped_values:
                    continue
                value = atomic_property_value(atom, property_name)
                if value is None:
                    continue
                grouped_values[element].append(value)
                individual_key = (element, atom.atom_id)
                if individual_key not in individual_observations:
                    if (
                        max_individual_series is not None
                        and len(individual_observations) >= max_individual_series
                    ):
                        raise ValueError(
                            "Individual atom plotting would create more than "
                            f"{max_individual_series} series. Select fewer elements "
                            "or use Atom IDs for a focused individual-atom plot."
                        )
                    individual_observations[individual_key] = []
                individual_observations[individual_key].append((frame.timestep, value, 0.0, 1))
            for element, values in grouped_values.items():
                if values:
                    aggregate_observations[element].append(
                        (frame.timestep, fmean(values), pstdev(values), len(values))
                    )

        for element, values in aggregate_observations.items():
            if values:
                aggregate_result.append(
                    _atomic_series(f"Simulation {simulation.index} {element}", values)
                )
        for (element, atom_id), values in sorted(individual_observations.items()):
            individual_result.append(
                _atomic_series(
                    f"Simulation {simulation.index} {element} atom {atom_id}",
                    values,
                )
            )

    if not aggregate_result:
        raise ValueError(
            "No atomic observations match the selected property, simulations, elements, and range."
        )
    return ElementAtomicSeries(
        aggregate=aggregate_result,
        individual=individual_result,
    )


def atomic_property_value(atom: TrajectoryAtom, property_name: str) -> float | None:
    """Return one stored or vector-magnitude property for an atom."""

    components = DERIVED_COMPONENTS.get(property_name)
    if components is None:
        return atom.values.get(property_name)
    values = [atom.values.get(component) for component in components]
    if any(value is None for value in values):
        return None
    return math.sqrt(sum(value * value for value in values if value is not None))


def atomic_property_label(property_name: str) -> str:
    """Return a display label with a unit when the property has a known unit."""

    name = ATOMIC_PROPERTY_NAMES.get(property_name, property_name)
    unit = ATOMIC_PROPERTY_UNITS.get(property_name)
    return f"{name} [{unit}]" if unit else name


def parse_atom_ids(value: str) -> list[int]:
    """Parse comma/space-separated atom IDs and inclusive ID ranges."""

    tokens = value.replace(",", " ").split()
    atom_ids = set()
    for token in tokens:
        match = ATOM_ID_TOKEN.fullmatch(token)
        if match is None:
            raise ValueError(f"Invalid atom ID or range: {token!r}.")
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if start <= 0 or end <= 0:
            raise ValueError("Atom IDs must be positive integers.")
        if end < start:
            raise ValueError(f"Atom ID range ends before it starts: {token!r}.")
        atom_ids.update(range(start, end + 1))
    if not atom_ids:
        raise ValueError("Enter at least one atom ID.")
    return sorted(atom_ids)


def _atom_selector(atom, simulation, elements, atom_ids):
    """Return the selected element name or atom ID for one atom."""

    if elements:
        return _atom_element(atom, simulation)
    return atom.atom_id if atom.atom_id in atom_ids else None


def _atom_element(atom, simulation) -> str | None:
    """Return an atom's explicit or type-mapped element name."""

    if atom.element is not None:
        return atom.element
    if simulation.type_to_element is None:
        return None
    return simulation.type_to_element.get(atom.atom_type)


def _atomic_series(label: str, values) -> AtomicSeries:
    """Build an atomic series from timestep/statistic tuples."""

    return AtomicSeries(
        label=label,
        timesteps=[value[0] for value in values],
        means=[value[1] for value in values],
        deviations=[value[2] for value in values],
        counts=[value[3] for value in values],
    )


def _collect_element_aggregate_series(
    simulation: LoadedSimulation,
    property_name: str,
    selected_elements: tuple[str, ...],
    step_range: tuple[float, float] | None,
) -> list[AtomicSeries]:
    """Collect per-element means without constructing per-atom objects."""

    observations = {element: [] for element in selected_elements}
    selected_element_set = set(selected_elements)
    step_upper = None if step_range is None else max(step_range)
    trajectory_path = Path(simulation.trajectory_path)
    with trajectory_path.open(encoding="utf-8") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue

            timestep = int(handle.readline().strip())
            number_header = handle.readline()
            if not number_header.startswith("ITEM: NUMBER OF ATOMS"):
                raise ValueError(f"Malformed trajectory frame at timestep {timestep} in {trajectory_path}")
            n_atoms = int(handle.readline().strip())

            bounds_header = handle.readline()
            if not bounds_header.startswith("ITEM: BOX BOUNDS"):
                raise ValueError(f"Missing box bounds at timestep {timestep} in {trajectory_path}")
            for _ in range(3):
                handle.readline()

            atoms_header = handle.readline()
            if not atoms_header.startswith("ITEM: ATOMS"):
                raise ValueError(f"Missing atom table at timestep {timestep} in {trajectory_path}")
            columns = atoms_header.split()[2:]

            if step_upper is not None and timestep > step_upper:
                break
            if not _in_step_range(timestep, step_range):
                for _ in range(n_atoms):
                    handle.readline()
                continue

            stats = {element: [0, 0.0, 0.0] for element in selected_elements}
            row_reader = _fast_atom_row_reader(columns, property_name, simulation.type_to_element)
            for _ in range(n_atoms):
                values = handle.readline().split()
                element, value = row_reader(values)
                if element not in selected_element_set or value is None:
                    continue
                element_stats = stats[element]
                element_stats[0] += 1
                element_stats[1] += value
                element_stats[2] += value * value

            for element, (count, total, total_squared) in stats.items():
                if count:
                    mean = total / count
                    variance = max(total_squared / count - mean * mean, 0.0)
                    observations[element].append(
                        (timestep, mean, math.sqrt(variance), count)
                    )

    return [
        AtomicSeries(
            label=f"Simulation {simulation.index} {element}",
            timesteps=[value[0] for value in values],
            means=[value[1] for value in values],
            deviations=[value[2] for value in values],
            counts=[value[3] for value in values],
        )
        for element, values in observations.items()
        if values
    ]


def _fast_atom_row_reader(
    columns: list[str],
    property_name: str,
    type_to_element: dict[int, str] | None,
):
    """Build a row reader for element/type selection and one numeric property."""

    column_index = {column: index for index, column in enumerate(columns)}
    element_index = column_index.get("element")
    type_index = column_index.get("type")
    property_indexes = _fast_property_indexes(column_index, property_name)
    required_indexes = [
        index
        for index in (element_index, type_index, *property_indexes)
        if index is not None
    ]
    minimum_values = max(required_indexes, default=-1) + 1

    def read_row(values: list[str]) -> tuple[str | None, float | None]:
        if len(values) < minimum_values:
            return None, None
        element = None
        if element_index is not None:
            element = values[element_index]
        elif type_index is not None and type_to_element is not None:
            try:
                element = type_to_element.get(int(float(values[type_index])))
            except ValueError:
                element = None
        value = _fast_property_value(values, property_indexes)
        return element, value

    return read_row


def _fast_property_indexes(column_index: dict[str, int], property_name: str) -> tuple[int, ...]:
    """Return direct or vector-component column indexes for a property."""

    components = DERIVED_COMPONENTS.get(property_name)
    if components is not None:
        return tuple(column_index[component] for component in components)
    return (column_index[property_name],)


def _fast_property_value(values: list[str], property_indexes: tuple[int, ...]) -> float | None:
    """Return a scalar or vector magnitude from raw atom-table values."""

    try:
        property_values = [float(values[index]) for index in property_indexes]
    except (IndexError, ValueError):
        return None
    if len(property_values) == 1:
        return property_values[0]
    return math.sqrt(sum(value * value for value in property_values))


def _in_step_range(timestep: int, step_range: tuple[float, float] | None) -> bool:
    """Return whether a timestep is inside an optional inclusive range."""

    if step_range is None:
        return True
    lower, upper = sorted(step_range)
    return lower <= timestep <= upper


def _trajectory_step_range(step_range: tuple[float, float] | None) -> tuple[int, int] | None:
    """Return a parser-friendly timestep range that still includes edge floats."""

    if step_range is None:
        return None
    lower, upper = sorted(step_range)
    return math.floor(lower), math.ceil(upper)
