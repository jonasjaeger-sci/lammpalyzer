"""Matplotlib plotting helpers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from importlib import import_module
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lammpalyze.analysis import LoadedSimulation, aggregate_thermo
from lammpalyze.atomic import (
    AtomicSeries,
    atomic_property_label,
    collect_atomic_series,
    collect_element_atomic_series,
)

ReferenceLines = tuple[list[float], list[float]]
TIME_UNITS_IN_FS = {
    "fs": 1.0,
    "ps": 1_000.0,
    "ns": 1_000_000.0,
    "us": 1_000_000_000.0,
    "µs": 1_000_000_000.0,
    "ms": 1_000_000_000_000.0,
    "s": 1_000_000_000_000_000.0,
}
LEGEND_LOCATIONS = {
    "best",
    "upper right",
    "upper left",
    "lower left",
    "lower right",
    "center left",
    "center right",
    "lower center",
    "upper center",
    "center",
}

SPECIES_DARK_COLORS = [
    "#4cc9f0",
    "#f72585",
    "#f9c74f",
    "#90be6d",
    "#f9844a",
    "#b5179e",
    "#43aa8b",
    "#577590",
    "#ff6b6b",
    "#c77dff",
    "#80ed99",
    "#ffd166",
]

THERMO_DARK_COLORS = {
    "line": "#4cc9f0",
    "mean": "#f9c74f",
    "std": "#f72585",
    "figure": "#0b1020",
    "axes": "#111827",
    "text": "#e5e7eb",
    "title": "#f9fafb",
    "tick": "#d1d5db",
    "grid": "#374151",
    "spine": "#6b7280",
}

THERMO_BRIGHT_COLORS = {
    "line": "#2563eb",
    "mean": "#b45309",
    "std": "#be123c",
    "figure": "#f8fafc",
    "axes": "#ffffff",
    "text": "#111827",
    "title": "#020617",
    "tick": "#1f2937",
    "grid": "#cbd5e1",
    "spine": "#64748b",
}

THERMO_THEMES = {
    "dark": THERMO_DARK_COLORS,
    "bright": THERMO_BRIGHT_COLORS,
}

THERMO_LINE_COLORS = [
    "#4cc9f0",
    "#f72585",
    "#90be6d",
    "#f9844a",
    "#c77dff",
    "#ffd166",
]

CHARGE_LINE_COLORS = SPECIES_DARK_COLORS

THERMO_UNITS = {
    "PotEng": "kcal/mol",
    "KinEng": "kcal/mol",
    "TotEng": "kcal/mol",
    "E_vdwl": "kcal/mol",
    "E_coul": "kcal/mol",
    "Temp": "K",
    "Press": "atm",
    "Vol": "A³",
    "Volume": "A³",
}


@dataclass(frozen=True)
class PlotSettings:
    """Global plot display settings shared by GUI plots."""

    x_axis: str = "timestep"
    timestep_size_fs: float = 0.5
    time_unit: str = "ps"
    reset_x_origin: bool = False
    log_x: bool = False
    log_y: bool = False


def plot_species(
    simulations: list[LoadedSimulation],
    species: list[str],
    reference_lines: ReferenceLines | None = None,
    step_range: tuple[float, float] | None = None,
    excluded_timesteps: list[int] | None = None,
    theme: str = "dark",
    data_source: str = "lammps",
    plot_settings: PlotSettings | None = None,
):
    """Plot selected species counts and total molecule count over time."""

    style = _theme_colors(theme)
    settings = _normalize_plot_settings(plot_settings)
    normalized_source = _normalize_species_data_source(data_source)
    return [
        _plot_species_counts(
            simulations,
            species,
            reference_lines,
            step_range,
            excluded_timesteps,
            style,
            normalized_source,
            settings,
        ),
        _plot_species_molecule_counts(
            simulations,
            reference_lines,
            step_range,
            excluded_timesteps,
            style,
            normalized_source,
            settings,
        ),
    ]


def simulation_has_species_source(simulation: LoadedSimulation, data_source: str) -> bool:
    """Return whether a simulation has data for a species plot source."""

    normalized_source = _normalize_species_data_source(data_source)
    if normalized_source == "lammps":
        return simulation.species_df is not None
    molecule_values = simulation.chem_formulas if normalized_source == "formula" else simulation.smiles
    return bool(molecule_values)


def species_names_for_source(simulations: list[LoadedSimulation], data_source: str = "lammps") -> list[str]:
    """Return available species names for the selected species plot source."""

    normalized_source = _normalize_species_data_source(data_source)
    if normalized_source == "lammps":
        return sorted(
            {
                name
                for simulation in simulations
                if simulation.species
                for name in simulation.species
            }
        )
    names = set()
    for simulation in simulations:
        names.update(_bond_species_labels(simulation, normalized_source))
    return sorted(names)


def _plot_species_counts(
    simulations: list[LoadedSimulation],
    species: list[str],
    reference_lines: ReferenceLines | None = None,
    step_range: tuple[float, float] | None = None,
    excluded_timesteps: list[int] | None = None,
    style: dict[str, str] | None = None,
    data_source: str = "lammps",
    plot_settings: PlotSettings | None = None,
):
    """Plot selected species counts over time for each simulation."""

    style = style or THERMO_DARK_COLORS
    settings = _normalize_plot_settings(plot_settings)
    fig, ax = plt.subplots(facecolor=style["figure"])
    color_cycle = cycle(SPECIES_DARK_COLORS)
    plotted_lines = 0
    plotted_frames = []

    for simulation in simulations:
        species_df = _filtered_species_frame(simulation, data_source, step_range, excluded_timesteps)
        if species_df is None or species_df.empty:
            continue
        plotted_frames.append((simulation, species_df))
    x_origin = _time_axis_origin([frame["Timestep"] for _simulation, frame in plotted_frames], step_range)

    for simulation, species_df in plotted_frames:
        available_species = [name for name in species if name in species_df.columns]
        for name in available_species:
            ax.plot(
                _display_time_values(species_df["Timestep"], settings, x_origin),
                species_df[name],
                label=f"R{simulation.index} {name}",
                color=next(color_cycle),
                linewidth=2.0,
            )
            plotted_lines += 1

    title = "Species evolution" if data_source == "lammps" else f"Bond-derived species ({data_source})"
    _style_axes(ax, title, "Count", style, x_label=_time_axis_label(settings), plot_settings=settings)
    _apply_step_range(ax, step_range, settings, x_origin)
    _add_reference_lines(ax, _display_reference_lines(reference_lines, settings, x_origin), color=style["text"])

    if plotted_lines:
        legend_columns = min(max(1, plotted_lines // 6 + 1), 5)
        handles, labels = ax.get_legend_handles_labels()
        legend = fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            ncol=legend_columns,
            frameon=False,
            fontsize="small",
        )
        for text in legend.get_texts():
            text.set_color(style["text"])

    fig.tight_layout(rect=(0, 0, 1, 0.82))
    return fig


def _plot_species_molecule_counts(
    simulations: list[LoadedSimulation],
    reference_lines: ReferenceLines | None = None,
    step_range: tuple[float, float] | None = None,
    excluded_timesteps: list[int] | None = None,
    style: dict[str, str] | None = None,
    data_source: str = "lammps",
    plot_settings: PlotSettings | None = None,
):
    """Plot the total number of molecules over time for each simulation."""

    style = style or THERMO_DARK_COLORS
    settings = _normalize_plot_settings(plot_settings)
    fig, ax = plt.subplots(facecolor=style["figure"])
    color_cycle = cycle(SPECIES_DARK_COLORS)
    plotted_lines = 0
    plotted_frames = []

    for simulation in simulations:
        species_df = _filtered_species_frame(simulation, data_source, step_range, excluded_timesteps)
        if species_df is None or species_df.empty or "No_Moles" not in species_df.columns:
            continue
        plotted_frames.append((simulation, species_df))
    x_origin = _time_axis_origin([frame["Timestep"] for _simulation, frame in plotted_frames], step_range)

    for simulation, species_df in plotted_frames:
        ax.plot(
            _display_time_values(species_df["Timestep"], settings, x_origin),
            species_df["No_Moles"],
            label=f"R{simulation.index} No_Moles",
            color=next(color_cycle),
            linewidth=2.0,
        )
        plotted_lines += 1

    _style_axes(ax, "Total molecules", "Molecules", style, x_label=_time_axis_label(settings), plot_settings=settings)
    _apply_step_range(ax, step_range, settings, x_origin)
    _add_reference_lines(ax, _display_reference_lines(reference_lines, settings, x_origin), color=style["text"])

    if plotted_lines:
        legend = ax.legend(frameon=False)
        for text in legend.get_texts():
            text.set_color(style["text"])

    fig.tight_layout()
    return fig


def _filtered_species_frame(
    simulation: LoadedSimulation,
    data_source: str,
    step_range: tuple[float, float] | None,
    excluded_timesteps: list[int] | None,
):
    """Return species data filtered to the requested visible timesteps."""

    normalized_source = _normalize_species_data_source(data_source)
    if normalized_source == "lammps":
        if simulation.species_df is None:
            return None
        frame = simulation.species_df
    else:
        frame = _bond_species_frame(simulation, normalized_source)
        if frame is None:
            return None
    if excluded_timesteps:
        frame = frame[~frame["Timestep"].isin(set(excluded_timesteps))]
    if step_range is not None:
        lower, upper = sorted(step_range)
        frame = frame[(frame["Timestep"] >= lower) & (frame["Timestep"] <= upper)]
    return frame


def _bond_species_frame(simulation: LoadedSimulation, notation: str) -> pd.DataFrame | None:
    """Build a species-count table from parsed bond-analysis molecule labels."""

    molecule_values = simulation.chem_formulas if notation == "formula" else simulation.smiles
    if not molecule_values:
        return None

    rows = []
    for timestep, labels in sorted(molecule_values.items()):
        counts = Counter(_included_bond_species_labels(simulation, timestep, labels))
        row = {
            "Timestep": timestep,
            "No_Moles": sum(counts.values()),
            "No_Specs": len(counts),
        }
        row.update(counts)
        rows.append(row)
    if not rows:
        return None
    return pd.DataFrame(rows).fillna(0)


def _bond_species_labels(simulation: LoadedSimulation, notation: str) -> set[str]:
    """Return bond-derived molecule labels visible under the current quality policy."""

    molecule_values = simulation.chem_formulas if notation == "formula" else simulation.smiles
    if not molecule_values:
        return set()
    labels = set()
    for timestep, timestep_labels in molecule_values.items():
        labels.update(_included_bond_species_labels(simulation, timestep, timestep_labels))
    return labels


def _included_bond_species_labels(
    simulation: LoadedSimulation,
    timestep: int,
    labels: list[str],
) -> list[str]:
    """Return molecule labels after applying the suspicious-structure exclusion policy."""

    quality_mode = getattr(simulation, "structure_quality_mode", "flag")
    excluded_indexes = (
        (getattr(simulation, "excluded_components", None) or {}).get(timestep, set())
        if quality_mode in {"exclude", "skip"}
        else set()
    )
    return [
        label
        for component_index, label in enumerate(labels)
        if component_index not in excluded_indexes
    ]


def _normalize_species_data_source(data_source: str) -> str:
    """Normalize user-facing species data source choices."""

    source = data_source.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "lammps": "lammps",
        "species": "lammps",
        "lammps_species": "lammps",
        "species_file": "lammps",
        "formula": "formula",
        "chemical_formula": "formula",
        "bond_formula": "formula",
        "bond_formulas": "formula",
        "bond_analysis_formula": "formula",
        "smiles": "smiles",
        "bond_smiles": "smiles",
        "bond_analysis_smiles": "smiles",
    }
    if source not in aliases:
        raise ValueError("Species data source must be 'lammps', 'formula', or 'smiles'.")
    return aliases[source]


def plot_rdf(*args, **kwargs):
    """Forward RDF rendering to its dedicated module for API compatibility."""

    rdf_plotting = import_module("lammpalyze.rdf_plotting")
    return rdf_plotting.plot_rdf(*args, **kwargs)


def plot_charge_evolution(
    simulations: list[LoadedSimulation],
    elements: list[str],
    *,
    uncertainty: str = "band",
    step_range: tuple[float, float] | None = None,
    theme: str = "dark",
    plot_settings: PlotSettings | None = None,
):
    """Plot mean atomic partial charge by element, with optional deviations."""

    if uncertainty not in {"band", "errorbar", "none"}:
        raise ValueError("uncertainty must be 'band', 'errorbar', or 'none'.")
    if not elements:
        raise ValueError("Select at least one element for charge plotting.")

    style = _theme_colors(theme)
    settings = _normalize_plot_settings(plot_settings)
    fig, ax = plt.subplots(figsize=(9.0, 5.2), facecolor=style["figure"])
    color_cycle = cycle(CHARGE_LINE_COLORS)
    origin_values = []
    for simulation in simulations:
        if not simulation.charge_statistics:
            continue
        for element in elements:
            origin_values.append(
                [
                    timestep
                    for timestep, summaries in sorted(simulation.charge_statistics.items())
                    if element in summaries
                ]
            )
    x_origin = _time_axis_origin(origin_values, step_range)
    plotted = 0
    for simulation in simulations:
        if not simulation.charge_statistics:
            continue
        for element in elements:
            observations = [
                (timestep, summaries[element])
                for timestep, summaries in sorted(simulation.charge_statistics.items())
                if element in summaries
            ]
            if step_range is not None:
                lower, upper = sorted(step_range)
                observations = [item for item in observations if lower <= item[0] <= upper]
            if not observations:
                continue
            timesteps = [item[0] for item in observations]
            means = [item[1].mean for item in observations]
            deviations = [item[1].std for item in observations]
            color = next(color_cycle)
            label = f"Simulation {simulation.index} {element}"
            if uncertainty == "errorbar":
                ax.errorbar(
                    _display_time_values(timesteps, settings, x_origin),
                    means,
                    yerr=deviations,
                    label=label,
                    color=color,
                    linewidth=1.8,
                    capsize=2,
                )
            else:
                x_values = _display_time_values(timesteps, settings, x_origin)
                ax.plot(x_values, means, label=label, color=color, linewidth=2.0)
                if uncertainty == "band":
                    lower_values = [mean - std for mean, std in zip(means, deviations, strict=False)]
                    upper_values = [mean + std for mean, std in zip(means, deviations, strict=False)]
                    ax.fill_between(x_values, lower_values, upper_values, color=color, alpha=0.18)
            plotted += 1

    if not plotted:
        raise ValueError("No charge observations match the selected simulations, elements, and range.")
    _style_axes(
        ax,
        "Atomic partial charges",
        "Mean partial charge [e]",
        style,
        x_label=_time_axis_label(settings),
        plot_settings=settings,
    )
    _apply_step_range(ax, step_range, settings, x_origin)
    legend = ax.legend(frameon=False, fontsize="small")
    for text in legend.get_texts():
        text.set_color(style["text"])
    fig.tight_layout()
    return fig


def plot_atomic_data(
    simulations: list[LoadedSimulation],
    property_name: str,
    *,
    elements: list[str] | None = None,
    atom_ids: list[int] | None = None,
    uncertainty: str = "band",
    step_range: tuple[float, float] | None = None,
    theme: str = "dark",
    plot_settings: PlotSettings | None = None,
):
    """Plot a trajectory atom property by element or individual atom ID."""

    if uncertainty not in {"band", "errorbar", "none"}:
        raise ValueError("uncertainty must be 'band', 'errorbar', or 'none'.")
    series = collect_atomic_series(
        simulations,
        property_name,
        elements=elements,
        atom_ids=atom_ids,
        step_range=step_range,
    )
    property_label = atomic_property_label(property_name)
    return _plot_atomic_series(
        series,
        property_label,
        property_label,
        uncertainty=uncertainty,
        show_uncertainty=bool(elements),
        step_range=step_range,
        theme=theme,
        plot_settings=plot_settings,
    )


def plot_atomic_data_figures(
    simulations: list[LoadedSimulation],
    property_name: str,
    *,
    elements: list[str] | None = None,
    atom_ids: list[int] | None = None,
    include_individual_element_atoms: bool = False,
    max_individual_element_atoms: int | None = 200,
    uncertainty: str = "band",
    step_range: tuple[float, float] | None = None,
    theme: str = "dark",
    plot_settings: PlotSettings | None = None,
) -> list:
    """Create aggregate and optional per-element-atom figures."""

    if not elements or not include_individual_element_atoms:
        return [
            plot_atomic_data(
                simulations,
                property_name,
                elements=elements,
                atom_ids=atom_ids,
                uncertainty=uncertainty,
                step_range=step_range,
                theme=theme,
                plot_settings=plot_settings,
            )
        ]
    if uncertainty not in {"band", "errorbar", "none"}:
        raise ValueError("uncertainty must be 'band', 'errorbar', or 'none'.")

    series = collect_element_atomic_series(
        simulations,
        property_name,
        elements,
        step_range=step_range,
        max_individual_series=max_individual_element_atoms,
    )
    property_label = atomic_property_label(property_name)
    element_label = ", ".join(elements)
    return [
        _plot_atomic_series(
            series.aggregate,
            property_label,
            property_label,
            uncertainty=uncertainty,
            show_uncertainty=True,
            step_range=step_range,
            theme=theme,
            plot_settings=plot_settings,
        ),
        _plot_atomic_series(
            series.individual,
            f"Individual {element_label} atoms: {property_label}",
            property_label,
            uncertainty="none",
            show_uncertainty=False,
            step_range=step_range,
            theme=theme,
            individual_legend=True,
            plot_settings=plot_settings,
        ),
    ]


def plot_collected_atomic_series(
    series: list[AtomicSeries],
    property_name: str,
    *,
    uncertainty: str = "band",
    show_uncertainty: bool = True,
    step_range: tuple[float, float] | None = None,
    theme: str = "dark",
    title: str | None = None,
    individual_legend: bool = False,
    plot_settings: PlotSettings | None = None,
):
    """Plot already-collected trajectory atom series."""

    property_label = atomic_property_label(property_name)
    return _plot_atomic_series(
        series,
        title or property_label,
        property_label,
        uncertainty=uncertainty,
        show_uncertainty=show_uncertainty,
        step_range=step_range,
        theme=theme,
        individual_legend=individual_legend,
        plot_settings=plot_settings,
    )


def _plot_atomic_series(
    series: list[AtomicSeries],
    title: str,
    y_label: str,
    *,
    uncertainty: str,
    show_uncertainty: bool,
    step_range: tuple[float, float] | None,
    theme: str,
    individual_legend: bool = False,
    plot_settings: PlotSettings | None = None,
):
    """Render already-collected atomic series without rereading a trajectory."""

    style = _theme_colors(theme)
    settings = _normalize_plot_settings(plot_settings)
    fig, ax = plt.subplots(figsize=(9.0, 5.2), facecolor=style["figure"])
    color_cycle = cycle(CHARGE_LINE_COLORS)
    x_origin = _time_axis_origin([observation.timesteps for observation in series], step_range)
    for observation in series:
        color = next(color_cycle)
        if uncertainty == "errorbar" and show_uncertainty:
            ax.errorbar(
                _display_time_values(observation.timesteps, settings, x_origin),
                observation.means,
                yerr=observation.deviations,
                label=observation.label,
                color=color,
                linewidth=1.8,
                capsize=2,
            )
        else:
            x_values = _display_time_values(observation.timesteps, settings, x_origin)
            ax.plot(
                x_values,
                observation.means,
                label=observation.label,
                color=color,
                linewidth=2.0,
            )
            if uncertainty == "band" and show_uncertainty:
                lower = [
                    mean - deviation
                    for mean, deviation in zip(
                        observation.means,
                        observation.deviations,
                        strict=False,
                    )
                ]
                upper = [
                    mean + deviation
                    for mean, deviation in zip(
                        observation.means,
                        observation.deviations,
                        strict=False,
                    )
                ]
                ax.fill_between(
                    x_values,
                    lower,
                    upper,
                    color=color,
                    alpha=0.18,
                )

    _style_axes(ax, title, y_label, style, x_label=_time_axis_label(settings), plot_settings=settings)
    _apply_step_range(ax, step_range, settings, x_origin)
    legend_options = {"frameon": False, "fontsize": "small"}
    if individual_legend:
        legend_options.update(
            {"bbox_to_anchor": (1.02, 1.0), "loc": "upper left", "fontsize": "x-small"}
        )
    legend = ax.legend(**legend_options)
    for text in legend.get_texts():
        text.set_color(style["text"])
    fig.tight_layout()
    return fig


def plot_msd(
    simulations: list[LoadedSimulation],
    selections: list[tuple[int, str]],
    *,
    step_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    running_average_points: int | None = None,
    reference_lines: ReferenceLines | None = None,
    average_groups: list[list[int]] | None = None,
    average_group_labels: list[str] | None = None,
    fit_range: tuple[float, float] | None = None,
    legend_location: str = "best",
    theme: str = "dark",
    plot_settings: PlotSettings | None = None,
):
    """Plot selected MSD series and a grouped mean/standard-deviation figure."""

    settings = _normalize_plot_settings(plot_settings)
    average_groups = average_groups or None
    simulation_by_index = {simulation.index: simulation for simulation in simulations}
    selected_series = []
    for simulation_index, column in selections:
        simulation = simulation_by_index.get(simulation_index)
        if simulation is None or simulation.msd_df is None or column not in simulation.msd_df.columns:
            continue
        selected_series.append(
            (
                simulation_index,
                simulation.msd_df["Timestep"],
                simulation.msd_df[column],
                f"MSD{simulation_index} - {column}",
            )
        )
    combined = _plot_computed_series(
        [(x_values, y_values, label) for _, x_values, y_values, label in selected_series],
        title="Mean-square displacement",
        y_label="Mean-square displacement [Å²]",
        step_range=step_range,
        y_range=y_range,
        running_average_points=running_average_points,
        reference_lines=reference_lines,
        legend_location=legend_location,
        theme=theme,
        linear_fit_range=fit_range,
        linear_fit_slope_scale=1.0 / 6.0,
        linear_fit_unit=f"Å²/{_time_axis_unit(settings)}",
        linear_fit_quantity="D",
        plot_settings=settings,
    )
    try:
        averaged = _plot_msd_averages(
            selected_series,
            average_groups=average_groups,
            average_group_labels=average_group_labels,
            step_range=step_range,
            y_range=y_range,
            reference_lines=reference_lines,
            legend_location=legend_location,
            theme=theme,
            plot_settings=settings,
        )
    except Exception:
        plt.close(combined)
        raise
    return [combined, averaged]


def plot_pairwise(
    simulations: list[LoadedSimulation],
    parameter: str,
    selections: list[tuple[int, str]],
    *,
    step_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    running_average_points: int | None = None,
    reference_lines: ReferenceLines | None = None,
    molecule_atom: tuple[int, int] | None = None,
    molecule_notation: str = "formula",
    legend_location: str = "best",
    theme: str = "dark",
    plot_settings: PlotSettings | None = None,
):
    """Plot one local-dump value over time for selected particle pairs."""

    simulation_by_index = {simulation.index: simulation for simulation in simulations}
    series = []
    for simulation_index, pair in selections:
        simulation = simulation_by_index.get(simulation_index)
        if simulation is None or simulation.pairwise_df is None:
            continue
        frame = simulation.pairwise_df
        if parameter not in frame.columns:
            continue
        pair_frame = frame[frame["Pair"] == pair].sort_values("Timestep")
        if pair_frame.empty:
            continue
        series.append(
            (
                pair_frame["Timestep"],
                pair_frame[parameter],
                f"Dump{simulation_index} - {pair}",
            )
        )
    figure = _plot_computed_series(
        series,
        title=f"Pairwise data: {parameter}",
        y_label=parameter,
        step_range=step_range,
        y_range=y_range,
        running_average_points=running_average_points,
        reference_lines=reference_lines,
        legend_location=legend_location,
        theme=theme,
        plot_settings=plot_settings,
    )
    if molecule_atom is not None:
        simulation_index, atom_id = molecule_atom
        simulation = simulation_by_index.get(simulation_index)
        if simulation is None:
            plt.close(figure)
            raise ValueError(f"Simulation {simulation_index} is unavailable for atom tracking.")
        try:
            _add_atom_molecule_axis(
                figure.axes[0],
                simulation,
                atom_id,
                notation=molecule_notation,
                legend_location=legend_location,
                theme=theme,
            )
        except Exception:
            plt.close(figure)
            raise
        figure.tight_layout()
    return figure


def _plot_msd_averages(
    selected_series,
    *,
    average_groups: list[list[int]] | None,
    average_group_labels: list[str] | None,
    step_range: tuple[float, float] | None,
    y_range: tuple[float, float] | None,
    reference_lines: ReferenceLines | None,
    legend_location: str,
    theme: str,
    plot_settings: PlotSettings | None = None,
):
    """Plot aligned MSD means and standard deviations for requested groups."""

    if not selected_series:
        raise ValueError("No matching MSD data found for the selected series.")
    style = _theme_colors(theme)
    settings = _normalize_plot_settings(plot_settings)
    specs = _msd_average_specs(selected_series, average_groups, average_group_labels)
    fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor=style["figure"])
    colors = _line_colors(len(specs), None)
    x_origin = _time_axis_origin([series[1] for series in selected_series], step_range)
    for (label, group_series), color in zip(specs, colors, strict=True):
        averaged = _average_computed_series(group_series)
        mean_label = "Mean" if average_groups is None else f"{label} mean"
        std_label = "Std. dev." if average_groups is None else f"{label} std. dev."
        ax.plot(
            _display_time_values(averaged["Timestep"], settings, x_origin),
            averaged["mean"],
            color=color,
            linewidth=2.2,
            label=mean_label,
        )
        ax.fill_between(
            _display_time_values(averaged["Timestep"], settings, x_origin),
            averaged["mean"] - averaged["std"],
            averaged["mean"] + averaged["std"],
            color=color,
            alpha=0.22,
            label=std_label,
        )
    _style_axes(
        ax,
        "Average mean-square displacement",
        "Mean-square displacement [Å²]",
        style,
        x_label=_time_axis_label(settings),
        plot_settings=settings,
    )
    _apply_step_range(ax, step_range, settings, x_origin)
    _apply_y_range(ax, y_range)
    _add_reference_lines(ax, _display_reference_lines(reference_lines, settings, x_origin), color=style["text"])
    legend = ax.legend(
        loc=_validated_legend_location(legend_location),
        frameon=False,
        fontsize="small",
    )
    for text in legend.get_texts():
        text.set_color(style["text"])
    fig.tight_layout()
    return fig


def _msd_average_specs(selected_series, average_groups, average_group_labels):
    """Build labelled selections for each requested MSD average group."""

    if not average_groups:
        return [("", selected_series)]
    specs = []
    for position, group in enumerate(average_groups):
        group_series = [series for series in selected_series if series[0] in group]
        if not group_series:
            continue
        label = "R" + ",R".join(str(index) for index in group)
        if average_group_labels is not None and position < len(average_group_labels):
            label = average_group_labels[position].strip() or label
        specs.append((label, group_series))
    if not specs:
        raise ValueError("No average groups match the selected MSD series.")
    return specs


def _average_computed_series(selected_series) -> pd.DataFrame:
    """Align selected computed curves by timestep and calculate mean and std."""

    frames = []
    for position, (_, x_values, y_values, _) in enumerate(selected_series):
        frame = pd.DataFrame({"Timestep": x_values, f"series_{position}": y_values}).dropna()
        frames.append(frame)
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="Timestep", how="inner")
    if merged.empty:
        raise ValueError("Selected MSD series have no common timesteps for averaging.")
    value_columns = [column for column in merged.columns if column != "Timestep"]
    return pd.DataFrame(
        {
            "Timestep": merged["Timestep"],
            "mean": merged[value_columns].mean(axis=1),
            "std": merged[value_columns].std(axis=1).fillna(0.0),
        }
    )


def atom_molecule_membership(
    simulation: LoadedSimulation,
    atom_id: int,
    notation: str = "formula",
) -> tuple[list[int], list[int], dict[int, str]]:
    """Numerize the molecule containing one atom at each parsed bond timestep."""

    timesteps, labels = atom_molecule_labels(simulation, atom_id, notation)
    label_to_number: dict[str, int] = {}
    numeric_values = [
        label_to_number.setdefault(label, len(label_to_number) + 1)
        for label in labels
    ]
    number_to_label = {number: label for label, number in label_to_number.items()}
    return timesteps, numeric_values, number_to_label


def atom_molecule_labels(
    simulation: LoadedSimulation,
    atom_id: int,
    notation: str = "formula",
) -> tuple[list[int], list[str]]:
    """Return the molecule label containing one atom at each parsed bond timestep."""

    normalized_notation = notation.strip().lower()
    if normalized_notation not in {"formula", "smiles"}:
        raise ValueError("Molecule notation must be 'formula' or 'smiles'.")
    molecule_values = (
        simulation.chem_formulas
        if normalized_notation == "formula"
        else simulation.smiles
    )
    if not simulation.smiles_id or not molecule_values:
        raise ValueError(f"Simulation {simulation.index} has no bond-derived molecule data.")

    atom_key = str(atom_id)
    timesteps = []
    molecule_labels = []
    for timestep, components in sorted(simulation.smiles_id.items()):
        labels = molecule_values.get(timestep, [])
        for component_index, component_atoms in enumerate(components):
            if (
                not any(str(value) == atom_key for value in component_atoms)
                or component_index >= len(labels)
            ):
                continue
            label = labels[component_index]
            timesteps.append(timestep)
            molecule_labels.append(label)
            break
    if not timesteps:
        raise ValueError(
            f"Atom {atom_id} was not found in simulation {simulation.index} molecule observations."
        )
    return timesteps, molecule_labels


def add_atom_molecule_axis(
    axis,
    simulations: list[LoadedSimulation],
    atom_ids: list[int],
    *,
    notation: str,
    legend_location: str,
    theme: str,
) -> None:
    """Add molecule-membership tracks to a secondary y-axis."""

    if not simulations:
        raise ValueError("Select at least one simulation for atom tracking.")
    if not atom_ids:
        raise ValueError("Select at least one atom ID for molecule tracking.")

    normalized_notation = notation.strip().lower()
    if normalized_notation not in {"formula", "smiles"}:
        raise ValueError("Molecule notation must be 'formula' or 'smiles'.")

    style = _theme_colors(theme)
    label_to_number: dict[str, int] = {}
    tracks = []
    for simulation in simulations:
        for atom_id in atom_ids:
            timesteps, labels = atom_molecule_labels(
                simulation,
                atom_id,
                normalized_notation,
            )
            values = [
                label_to_number.setdefault(label, len(label_to_number) + 1)
                for label in labels
            ]
            tracks.append((simulation.index, atom_id, timesteps, values))

    molecule_axis = axis.twinx()
    colors = _line_colors(len(tracks), None)
    for (simulation_index, atom_id, timesteps, values), color in zip(
        tracks,
        colors,
        strict=True,
    ):
        if len(simulations) == 1 and len(atom_ids) == 1:
            label = f"Atom {atom_id} molecule"
        else:
            label = f"Simulation {simulation_index} - atom {atom_id} molecule"
        molecule_axis.step(
            timesteps,
            values,
            where="post",
            color=color,
            linewidth=2.0,
            alpha=0.9,
            label=label,
        )

    number_to_label = {number: label for label, number in label_to_number.items()}
    ticks = sorted(number_to_label)
    molecule_axis.set_yticks(ticks)
    molecule_axis.set_yticklabels([f"{number}: {number_to_label[number]}" for number in ticks])
    notation_label = "formula" if normalized_notation == "formula" else "SMILES"
    axis_label = (
        f"Atom {atom_ids[0]} molecule ({notation_label})"
        if len(simulations) == 1 and len(atom_ids) == 1
        else f"Molecule state ({notation_label})"
    )
    molecule_axis.set_ylabel(
        axis_label,
        color=style["mean"],
        fontsize=13,
        fontweight="bold",
    )
    molecule_axis.tick_params(axis="y", colors=style["mean"])
    molecule_axis.spines["right"].set_color(style["mean"])
    _rebuild_combined_legend(axis, molecule_axis, legend_location, style)


def _add_atom_molecule_axis(
    axis,
    simulation: LoadedSimulation,
    atom_id: int,
    *,
    notation: str,
    legend_location: str,
    theme: str,
) -> None:
    """Add a numbered molecule-membership track to a pairwise plot."""

    add_atom_molecule_axis(
        axis,
        [simulation],
        [atom_id],
        notation=notation,
        legend_location=legend_location,
        theme=theme,
    )


def _rebuild_combined_legend(axis, molecule_axis, legend_location: str, style: dict[str, str]) -> None:
    """Merge primary and secondary-axis legend handles."""

    primary_legend = axis.get_legend()
    if primary_legend is not None:
        primary_legend.remove()
    primary_handles, primary_labels = axis.get_legend_handles_labels()
    molecule_handles, molecule_labels = molecule_axis.get_legend_handles_labels()
    legend = axis.legend(
        [*primary_handles, *molecule_handles],
        [*primary_labels, *molecule_labels],
        loc=_validated_legend_location(legend_location),
        frameon=False,
        fontsize="small",
    )
    for text in legend.get_texts():
        text.set_color(style["text"])


def _plot_computed_series(
    series,
    *,
    title: str,
    y_label: str,
    step_range: tuple[float, float] | None,
    y_range: tuple[float, float] | None,
    running_average_points: int | None,
    reference_lines: ReferenceLines | None,
    legend_location: str,
    theme: str,
    linear_fit_range: tuple[float, float] | None = None,
    linear_fit_slope_scale: float | None = None,
    linear_fit_unit: str | None = None,
    linear_fit_quantity: str = "slope",
    plot_settings: PlotSettings | None = None,
):
    """Render selected computed-data series using the thermo visual style."""

    if not series:
        raise ValueError("No matching computed data found for the selected series.")
    running_average_points = _validated_running_average_points(running_average_points)
    style = _theme_colors(theme)
    settings = _normalize_plot_settings(plot_settings)
    fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor=style["figure"])
    colors = _line_colors(len(series), None)
    x_origin = _time_axis_origin([x_values for x_values, _y_values, _label in series], step_range)
    plotted = 0
    for (x_values, y_values, label), color in zip(series, colors, strict=True):
        valid = x_values.notna() & y_values.notna()
        x_values = x_values[valid]
        y_values = y_values[valid]
        if x_values.empty:
            continue
        display_x_values = _display_time_values(x_values, settings, x_origin)
        ax.plot(display_x_values, y_values, color=color, linewidth=2.0, label=label)
        if linear_fit_range is not None:
            try:
                fit = _linear_fit_for_series(
                    x_values,
                    display_x_values,
                    y_values,
                    linear_fit_range,
                    label,
                )
            except Exception:
                plt.close(fig)
                raise
            fit_label = _linear_fit_label(
                label,
                fit["slope"],
                linear_fit_slope_scale,
                linear_fit_unit,
                linear_fit_quantity,
            )
            ax.plot(
                fit["x"],
                fit["y"],
                color=_inverse_hex_color(color),
                linestyle="--",
                linewidth=2.2,
                alpha=0.9,
                label=fit_label,
            )
        if running_average_points is not None:
            ax.plot(
                display_x_values,
                y_values.rolling(window=running_average_points, min_periods=1).mean(),
                color=color,
                linestyle="--",
                linewidth=1.8,
                label=f"{label} Avg",
            )
        plotted += 1
    if not plotted:
        plt.close(fig)
        raise ValueError("The selected computed series contain no numeric observations.")

    _style_axes(ax, title, y_label, style, x_label=_time_axis_label(settings), plot_settings=settings)
    _apply_step_range(ax, step_range, settings, x_origin)
    _apply_y_range(ax, y_range)
    _add_reference_lines(ax, _display_reference_lines(reference_lines, settings, x_origin), color=style["text"])
    legend = ax.legend(
        loc=_validated_legend_location(legend_location),
        frameon=False,
        fontsize="small",
    )
    for text in legend.get_texts():
        text.set_color(style["text"])
    fig.tight_layout()
    return fig


def _linear_fit_for_series(
    raw_x_values: pd.Series,
    display_x_values,
    y_values: pd.Series,
    fit_range: tuple[float, float],
    label: str,
) -> dict[str, np.ndarray | float]:
    """Return a first-order fit for one computed series inside a timestep range."""

    lower, upper = sorted(fit_range)
    fit_mask = (raw_x_values >= lower) & (raw_x_values <= upper)
    fit_x = np.asarray(display_x_values, dtype=float)[fit_mask]
    fit_y = y_values[fit_mask].to_numpy(dtype=float)
    if len(fit_x) < 2 or len(np.unique(fit_x)) < 2:
        raise ValueError(
            f"Series {label!r} needs at least two distinct finite timesteps "
            "inside the linear fit range."
        )
    slope, intercept = np.polyfit(fit_x, fit_y, deg=1)
    line_x = np.array([fit_x.min(), fit_x.max()])
    line_y = slope * line_x + intercept
    return {"x": line_x, "y": line_y, "slope": float(slope), "intercept": float(intercept)}


def _linear_fit_label(
    label: str,
    slope: float,
    slope_scale: float | None,
    unit: str | None,
    quantity: str,
) -> str:
    """Return a compact legend label for a linear fit and derived quantity."""

    if slope_scale is None:
        return f"{label} fit (slope={_format_scientific(slope)})"
    value = slope * slope_scale
    unit_text = f" {unit}" if unit else ""
    return f"{label} fit ({quantity}={_format_scientific(value)}{unit_text})"


def _format_scientific(value: float) -> str:
    """Format fitted values compactly for plot legends."""

    return f"{value:.4g}"


def plot_thermo(
    simulations: list[LoadedSimulation],
    parameter: str,
    legend_labels: dict[int, str] | None = None,
    step_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    running_average_points: int | None = None,
    reference_lines: ReferenceLines | None = None,
    average_groups: list[list[int]] | None = None,
    average_group_labels: list[str] | None = None,
    theme: str = "dark",
    gradient_colors: tuple[str, str] | None = None,
    plot_settings: PlotSettings | None = None,
):
    """Create one combined simulation figure and one averaged figure."""

    figures = []
    running_average_points = _validated_running_average_points(running_average_points)
    average_groups = average_groups or None
    style = _theme_colors(theme)
    settings = _normalize_plot_settings(plot_settings)
    y_label = thermo_axis_label(parameter)
    plottable = [
        simulation
        for simulation in simulations
        if simulation.thermo_df is not None and parameter in simulation.thermo_df.columns
    ]
    if not plottable:
        raise ValueError(f"No thermo data found for parameter {parameter!r}.")

    x_origin = _time_axis_origin([simulation.thermo_df["Step"] for simulation in plottable], step_range)
    fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor=style["figure"])
    line_colors = _line_colors(len(plottable), gradient_colors)
    for simulation, color in zip(plottable, line_colors, strict=False):
        label = _thermo_legend_label(simulation, legend_labels)
        ax.plot(
            _display_time_values(simulation.thermo_df["Step"], settings, x_origin),
            simulation.thermo_df[parameter],
            color=color,
            linewidth=2.0,
            label=label,
        )
        if running_average_points is not None:
            ax.plot(
                _display_time_values(simulation.thermo_df["Step"], settings, x_origin),
                simulation.thermo_df[parameter].rolling(window=running_average_points, min_periods=1).mean(),
                color=_inverse_hex_color(color),
                linestyle="-",
                linewidth=2.6,
                label=f"{label} Avg",
            )
    _style_axes(
        ax,
        f"Selected simulations: {parameter}",
        y_label,
        style,
        x_label=_time_axis_label(settings),
        plot_settings=settings,
    )
    _apply_step_range(ax, step_range, settings, x_origin)
    _apply_y_range(ax, y_range)
    _add_reference_lines(ax, _display_reference_lines(reference_lines, settings, x_origin), color=style["text"])
    legend = ax.legend(frameon=False)
    for text in legend.get_texts():
        text.set_color(style["text"])
    fig.tight_layout()
    figures.append(fig)

    average_specs = _thermo_average_specs(plottable, average_groups, average_group_labels)
    fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor=style["figure"])
    average_colors = _line_colors(len(average_specs), gradient_colors)
    for (label, simulations_for_average), average_color in zip(average_specs, average_colors, strict=False):
        averaged = aggregate_thermo(simulations_for_average, parameter)
        color = style["mean"] if average_groups is None and gradient_colors is None else average_color
        std_color = style["std"] if average_groups is None and gradient_colors is None else color
        mean_label = "Mean" if average_groups is None else f"{label} mean"
        std_label = "Std. dev." if average_groups is None else f"{label} std. dev."
        display_steps = _display_time_values(averaged["Step"], settings, x_origin)
        ax.plot(display_steps, averaged["mean"], color=color, linewidth=2.2, label=mean_label)
        ax.fill_between(
            display_steps,
            averaged["mean"] - averaged["std"],
            averaged["mean"] + averaged["std"],
            alpha=0.22,
            color=std_color,
            label=std_label,
        )
    _style_axes(
        ax,
        f"Average {parameter}",
        y_label,
        style,
        x_label=_time_axis_label(settings),
        plot_settings=settings,
    )
    _apply_step_range(ax, step_range, settings, x_origin)
    _apply_y_range(ax, y_range)
    _add_reference_lines(ax, _display_reference_lines(reference_lines, settings, x_origin), color=style["text"])
    legend = ax.legend(frameon=False)
    for text in legend.get_texts():
        text.set_color(style["text"])
    fig.tight_layout()
    figures.append(fig)
    return figures


def _thermo_legend_label(simulation: LoadedSimulation, legend_labels: dict[int, str] | None) -> str:
    """Return the user label for a thermo series or a simulation default."""

    if legend_labels is not None:
        label = legend_labels.get(simulation.index, "").strip()
        if label:
            return label
    return f"Simulation {simulation.index}"


def _thermo_average_specs(
    simulations: list[LoadedSimulation],
    average_groups: list[list[int]] | None,
    average_group_labels: list[str] | None = None,
) -> list[tuple[str, list[LoadedSimulation]]]:
    """Return labelled simulation groups for the thermo average figure."""

    if not average_groups:
        return [("", simulations)]

    simulations_by_index = {simulation.index: simulation for simulation in simulations}
    specs = []
    for position, group in enumerate(average_groups):
        grouped_simulations = [simulations_by_index[index] for index in group if index in simulations_by_index]
        if not grouped_simulations:
            continue
        label = "R" + ",R".join(str(simulation.index) for simulation in grouped_simulations)
        if average_group_labels is not None and position < len(average_group_labels):
            label = average_group_labels[position].strip() or label
        specs.append((label, grouped_simulations))
    if not specs:
        raise ValueError("No average groups match the selected thermodynamic simulations.")
    return specs


def _apply_step_range(
    ax,
    step_range: tuple[float, float] | None,
    plot_settings: PlotSettings | None = None,
    x_origin: float | None = None,
) -> None:
    """Apply an optional x-axis step range to a plot."""

    if step_range is None:
        return
    start, end = _display_step_range(step_range, _normalize_plot_settings(plot_settings), x_origin)
    if start == end:
        padding = max(abs(start) * 0.01, 1.0)
        start -= padding
        end += padding
    ax.set_xlim(start, end)


def _apply_y_range(ax, y_range: tuple[float, float] | None) -> None:
    """Apply an optional y-axis range to a plot."""

    if y_range is None:
        return
    start, end = sorted(y_range)
    if start == end:
        padding = max(abs(start) * 0.01, 1.0)
        start -= padding
        end += padding
    ax.set_ylim(start, end)


def _validated_running_average_points(points: int | None) -> int | None:
    """Validate and return the optional running-average window size."""

    if points is None:
        return None
    if points < 1:
        raise ValueError("Running-average points must be at least 1.")
    return points


def _validated_legend_location(location: str) -> str:
    """Normalize and validate a Matplotlib legend location."""

    normalized = location.strip().lower()
    if normalized not in LEGEND_LOCATIONS:
        expected = ", ".join(sorted(LEGEND_LOCATIONS))
        raise ValueError(f"Legend location must be one of {expected}; received {location!r}.")
    return normalized


def _normalize_plot_settings(plot_settings: PlotSettings | None = None) -> PlotSettings:
    """Validate and normalize shared plot display settings."""

    if plot_settings is None:
        return PlotSettings()
    x_axis = plot_settings.x_axis.strip().lower().replace(" ", "_")
    if x_axis in {"timesteps", "step", "steps"}:
        x_axis = "timestep"
    if x_axis in {"real_time", "time"}:
        x_axis = "time"
    if x_axis not in {"timestep", "time"}:
        raise ValueError("x_axis must be 'timestep' or 'time'.")
    if plot_settings.timestep_size_fs <= 0:
        raise ValueError("Time step size must be greater than zero.")
    time_unit = _normalize_time_unit(plot_settings.time_unit)
    return PlotSettings(
        x_axis=x_axis,
        timestep_size_fs=float(plot_settings.timestep_size_fs),
        time_unit=time_unit,
        reset_x_origin=bool(plot_settings.reset_x_origin),
        log_x=bool(plot_settings.log_x),
        log_y=bool(plot_settings.log_y),
    )


def _normalize_time_unit(unit: str) -> str:
    """Return a supported abbreviated time unit."""

    normalized = unit.strip()
    aliases = {
        "femtoseconds": "fs",
        "femtosecond": "fs",
        "picoseconds": "ps",
        "picosecond": "ps",
        "nanoseconds": "ns",
        "nanosecond": "ns",
        "microseconds": "us",
        "microsecond": "us",
        "µs": "us",
        "milliseconds": "ms",
        "millisecond": "ms",
        "seconds": "s",
        "second": "s",
    }
    normalized = aliases.get(normalized.lower(), normalized)
    if normalized not in TIME_UNITS_IN_FS:
        expected = ", ".join(["fs", "ps", "ns", "us", "ms", "s"])
        raise ValueError(f"Time unit must be one of {expected}.")
    return "us" if normalized == "µs" else normalized


def _time_axis_unit(plot_settings: PlotSettings | None = None) -> str:
    """Return the displayed x-axis unit for timestep-like plots."""

    settings = _normalize_plot_settings(plot_settings)
    return settings.time_unit if settings.x_axis == "time" else "timestep"


def _time_axis_label(plot_settings: PlotSettings | None = None) -> str:
    """Return the x-axis label for timestep-like plots."""

    settings = _normalize_plot_settings(plot_settings)
    if settings.x_axis == "time":
        return f"Time [{settings.time_unit}]"
    return "Timestep"


def _display_time_values(
    values,
    plot_settings: PlotSettings | None = None,
    x_origin: float | None = None,
):
    """Return timestep values converted to the configured display axis."""

    settings = _normalize_plot_settings(plot_settings)
    shifted_values = _reset_time_origin(values, x_origin) if settings.reset_x_origin else values
    if settings.x_axis != "time":
        return shifted_values
    factor = settings.timestep_size_fs / TIME_UNITS_IN_FS[settings.time_unit]
    if isinstance(shifted_values, pd.Series):
        return shifted_values * factor
    return np.asarray(shifted_values, dtype=float) * factor


def _display_step_range(
    step_range: tuple[float, float],
    plot_settings: PlotSettings | None = None,
    x_origin: float | None = None,
) -> tuple[float, float]:
    """Convert a timestep x-limit pair into display-axis units."""

    settings = _normalize_plot_settings(plot_settings)
    lower, upper = sorted(step_range)
    if settings.reset_x_origin:
        origin = lower if x_origin is None else x_origin
        lower, upper = lower - origin, upper - origin
    if settings.x_axis != "time":
        return lower, upper
    factor = settings.timestep_size_fs / TIME_UNITS_IN_FS[settings.time_unit]
    return lower * factor, upper * factor


def _display_reference_lines(
    reference_lines: ReferenceLines | None,
    plot_settings: PlotSettings | None = None,
    x_origin: float | None = None,
) -> ReferenceLines | None:
    """Convert vertical reference lines to display-axis units."""

    if reference_lines is None:
        return None
    vertical_lines, horizontal_lines = reference_lines
    settings = _normalize_plot_settings(plot_settings)
    converted_vertical = list(_display_time_values(pd.Series(vertical_lines), settings, x_origin))
    return converted_vertical, horizontal_lines


def _time_axis_origin(series_values, step_range: tuple[float, float] | None = None) -> float | None:
    """Return the raw timestep value to display as zero."""

    if step_range is not None:
        return min(step_range)
    minima = []
    for values in series_values:
        series = values if isinstance(values, pd.Series) else pd.Series(values)
        finite_values = pd.to_numeric(series, errors="coerce").dropna()
        if not finite_values.empty:
            minima.append(float(finite_values.min()))
    return min(minima) if minima else None


def _reset_time_origin(values, x_origin: float | None = None):
    """Shift finite timestep values so the first plotted origin is zero."""

    if isinstance(values, pd.Series):
        finite_values = values.dropna()
        if finite_values.empty:
            return values
        origin = finite_values.min() if x_origin is None else x_origin
        return values - origin
    array = np.asarray(values, dtype=float)
    finite = np.isfinite(array)
    if not finite.any():
        return array
    origin = float(array[finite].min()) if x_origin is None else x_origin
    return array - origin


def _apply_axis_scales(ax, plot_settings: PlotSettings | None = None) -> None:
    """Apply optional logarithmic x/y axis scaling."""

    settings = _normalize_plot_settings(plot_settings)
    if settings.log_x:
        ax.set_xscale("log", nonpositive="mask")
    if settings.log_y:
        ax.set_yscale("log", nonpositive="mask")


def _inverse_hex_color(color: str) -> str:
    """Return the RGB inverse of a ``#rrggbb`` color."""

    try:
        channels = [255 - channel for channel in _hex_to_rgb(color)]
    except ValueError:
        return THERMO_DARK_COLORS["text"]
    return "#" + "".join(f"{channel:02x}" for channel in channels)


def _line_colors(count: int, gradient_colors: tuple[str, str] | None) -> list[str]:
    """Return plot colors, optionally interpolated across a color gradient."""

    if count <= 0:
        return []
    if gradient_colors is None:
        return [color for _, color in zip(range(count), cycle(THERMO_LINE_COLORS), strict=False)]

    start = _hex_to_rgb(gradient_colors[0])
    end = _hex_to_rgb(gradient_colors[1])
    if count == 1:
        return [_rgb_to_hex(start)]
    colors = []
    for index in range(count):
        fraction = index / (count - 1)
        rgb = tuple(round(start[channel] + (end[channel] - start[channel]) * fraction) for channel in range(3))
        colors.append(_rgb_to_hex(rgb))
    return colors


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    """Convert a ``#rrggbb`` color to RGB integer channels."""

    stripped = color.strip().lstrip("#")
    if len(stripped) != 6:
        raise ValueError(f"Expected a #rrggbb color, got {color!r}.")
    try:
        return tuple(int(stripped[index:index + 2], 16) for index in range(0, 6, 2))
    except ValueError as exc:
        raise ValueError(f"Expected a #rrggbb color, got {color!r}.") from exc


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    """Convert RGB integer channels to a ``#rrggbb`` color."""

    return "#" + "".join(f"{channel:02x}" for channel in rgb)


def _theme_colors(theme: str) -> dict[str, str]:
    """Return thermo plot colors for a named theme."""

    normalized = theme.strip().lower()
    if normalized not in THERMO_THEMES:
        raise ValueError(f"Unknown thermo plot theme {theme!r}.")
    return THERMO_THEMES[normalized]


def _add_reference_lines(
    ax,
    reference_lines: ReferenceLines | None,
    color: str = THERMO_DARK_COLORS["text"],
) -> None:
    """Add optional vertical and horizontal reference lines to an axis."""

    if reference_lines is None:
        return
    vertical_lines, horizontal_lines = reference_lines
    for value in vertical_lines:
        ax.axvline(
            value,
            color=color,
            linestyle=":",
            linewidth=1.5,
            alpha=0.85,
            label="_nolegend_",
        )
    for value in horizontal_lines:
        ax.axhline(
            value,
            color=color,
            linestyle=":",
            linewidth=1.5,
            alpha=0.85,
            label="_nolegend_",
        )


def plot_thermo_per_simulation(simulations: list[LoadedSimulation], parameter: str):
    """Create one figure per simulation and one averaged figure.

    Kept for callers that still want the old behavior.
    """

    figures = []
    y_label = thermo_axis_label(parameter)
    for simulation in simulations:
        if simulation.thermo_df is None or parameter not in simulation.thermo_df.columns:
            continue
        fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor=THERMO_DARK_COLORS["figure"])
        ax.plot(
            simulation.thermo_df["Step"],
            simulation.thermo_df[parameter],
            color=THERMO_DARK_COLORS["line"],
            linewidth=2.0,
        )
        _style_dark_axes(ax, f"Simulation {simulation.index}: {parameter}", y_label)
        fig.tight_layout()
        figures.append(fig)

    averaged = aggregate_thermo(simulations, parameter)
    fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor=THERMO_DARK_COLORS["figure"])
    ax.plot(averaged["Step"], averaged["mean"], color=THERMO_DARK_COLORS["mean"], linewidth=2.2, label="Mean")
    ax.fill_between(
        averaged["Step"],
        averaged["mean"] - averaged["std"],
        averaged["mean"] + averaged["std"],
        alpha=0.25,
        color=THERMO_DARK_COLORS["std"],
        label="Std. dev.",
    )
    _style_dark_axes(ax, f"Average {parameter}", y_label)
    legend = ax.legend(frameon=False)
    for text in legend.get_texts():
        text.set_color(THERMO_DARK_COLORS["text"])
    fig.tight_layout()
    figures.append(fig)
    return figures


def thermo_axis_label(parameter: str) -> str:
    """Return a thermodynamic y-axis label with units when known."""

    unit = THERMO_UNITS.get(parameter)
    if unit is None:
        return parameter
    return f"{parameter} [{unit}]"


def _style_dark_axes(ax, title: str, y_label: str, x_label: str = "Step") -> None:
    """Apply the shared dark Matplotlib axis style."""

    _style_axes(ax, title, y_label, THERMO_DARK_COLORS, x_label=x_label)


def _style_axes(
    ax,
    title: str,
    y_label: str,
    colors: dict[str, str],
    x_label: str = "Step",
    plot_settings: PlotSettings | None = None,
) -> None:
    """Apply a Matplotlib axis style from a color theme."""

    ax.set_facecolor(colors["axes"])
    ax.set_xlabel(x_label, color=colors["text"], fontsize=16, fontweight="bold")
    ax.set_ylabel(y_label, color=colors["text"], fontsize=16, fontweight="bold")
    ax.set_title(title, color=colors["title"], pad=12)
    ax.tick_params(axis="both", colors=colors["tick"])
    ax.grid(True, color=colors["grid"], alpha=0.55, linewidth=0.8)
    _apply_axis_scales(ax, plot_settings)
    for spine in ax.spines.values():
        spine.set_color(colors["spine"])
