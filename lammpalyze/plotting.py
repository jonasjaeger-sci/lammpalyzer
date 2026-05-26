"""Matplotlib plotting helpers."""

from __future__ import annotations

from itertools import cycle

import matplotlib.pyplot as plt

from lammpalyze.analysis import LoadedSimulation, aggregate_thermo
from lammpalyze.rdf import RDFResult

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

THERMO_LINE_COLORS = [
    "#4cc9f0",
    "#f72585",
    "#90be6d",
    "#f9844a",
    "#c77dff",
    "#ffd166",
]

THERMO_UNITS = {
    "PotEng": "kcal/mol",
    "KinEng": "kcal/mol",
    "TotEng": "kcal/mol",
    "Temp": "K",
    "Press": "atm",
    "Vol": "A³",
    "Volume": "A³",
}


def plot_species(simulations: list[LoadedSimulation], species: list[str]):
    """Plot selected species counts over time for each simulation."""

    fig, ax = plt.subplots(facecolor="#0b1020")
    ax.set_facecolor("#111827")
    color_cycle = cycle(SPECIES_DARK_COLORS)
    plotted_lines = 0

    for simulation in simulations:
        if simulation.species_df is None:
            continue
        available_species = [name for name in species if name in simulation.species_df.columns]
        for name in available_species:
            ax.plot(
                simulation.species_df["Timestep"],
                simulation.species_df[name],
                label=f"R{simulation.index} {name}",
                color=next(color_cycle),
                linewidth=2.0,
            )
            plotted_lines += 1

    ax.set_xlabel("Timestep", color="#e5e7eb", fontsize=16, fontweight="bold")
    ax.set_ylabel("Count", color="#e5e7eb", fontsize=16, fontweight="bold")
    ax.set_title("Species evolution", color="#f9fafb", pad=12)
    ax.tick_params(axis="both", colors="#d1d5db")
    ax.grid(True, color="#374151", alpha=0.55, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color("#6b7280")

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
            text.set_color("#e5e7eb")

    fig.tight_layout(rect=(0, 0, 1, 0.82))
    return fig


def plot_rdf(results: list[RDFResult], element_a: str, element_b: str):
    """Plot one time-averaged RDF curve per selected simulation."""

    if not results:
        raise ValueError("No RDF data to plot.")

    fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor=THERMO_DARK_COLORS["figure"])
    color_cycle = cycle(THERMO_LINE_COLORS)
    for result in results:
        ax.plot(
            result.r,
            result.g_r,
            color=next(color_cycle),
            linewidth=2.0,
            label=f"Simulation {result.simulation_index}",
        )

    _style_dark_axes(ax, f"RDF {element_a}-{element_b}", "g(r)", x_label="r [A]")
    legend = ax.legend(frameon=False)
    for text in legend.get_texts():
        text.set_color(THERMO_DARK_COLORS["text"])
    fig.tight_layout()
    return fig


def plot_thermo(
    simulations: list[LoadedSimulation],
    parameter: str,
    legend_labels: dict[int, str] | None = None,
    step_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
):
    """Create one combined simulation figure and one averaged figure."""

    figures = []
    y_label = thermo_axis_label(parameter)
    plottable = [
        simulation
        for simulation in simulations
        if simulation.thermo_df is not None and parameter in simulation.thermo_df.columns
    ]
    if not plottable:
        raise ValueError(f"No thermo data found for parameter {parameter!r}.")

    fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor=THERMO_DARK_COLORS["figure"])
    color_cycle = cycle(THERMO_LINE_COLORS)
    for simulation in plottable:
        label = _thermo_legend_label(simulation, legend_labels)
        ax.plot(
            simulation.thermo_df["Step"],
            simulation.thermo_df[parameter],
            color=next(color_cycle),
            linewidth=2.0,
            label=label,
        )
    _style_dark_axes(ax, f"Selected simulations: {parameter}", y_label)
    _apply_step_range(ax, step_range)
    _apply_y_range(ax, y_range)
    legend = ax.legend(frameon=False)
    for text in legend.get_texts():
        text.set_color(THERMO_DARK_COLORS["text"])
    fig.tight_layout()
    figures.append(fig)

    averaged = aggregate_thermo(plottable, parameter)
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
    _apply_step_range(ax, step_range)
    _apply_y_range(ax, y_range)
    legend = ax.legend(frameon=False)
    for text in legend.get_texts():
        text.set_color(THERMO_DARK_COLORS["text"])
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


def _apply_step_range(ax, step_range: tuple[float, float] | None) -> None:
    """Apply an optional x-axis step range to a plot."""

    if step_range is None:
        return
    start, end = sorted(step_range)
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

    ax.set_facecolor(THERMO_DARK_COLORS["axes"])
    ax.set_xlabel(x_label, color=THERMO_DARK_COLORS["text"], fontsize=16, fontweight="bold")
    ax.set_ylabel(y_label, color=THERMO_DARK_COLORS["text"], fontsize=16, fontweight="bold")
    ax.set_title(title, color=THERMO_DARK_COLORS["title"], pad=12)
    ax.tick_params(axis="both", colors=THERMO_DARK_COLORS["tick"])
    ax.grid(True, color=THERMO_DARK_COLORS["grid"], alpha=0.55, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color(THERMO_DARK_COLORS["spine"])
