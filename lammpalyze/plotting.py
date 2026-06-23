"""Matplotlib plotting helpers."""

from __future__ import annotations

from itertools import cycle

import matplotlib.pyplot as plt

from lammpalyze.analysis import LoadedSimulation, aggregate_thermo
from lammpalyze.rdf import RDFResult

ReferenceLines = tuple[list[float], list[float]]

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
    "Temp": "K",
    "Press": "atm",
    "Vol": "A³",
    "Volume": "A³",
}


def plot_species(
    simulations: list[LoadedSimulation],
    species: list[str],
    reference_lines: ReferenceLines | None = None,
    step_range: tuple[float, float] | None = None,
    excluded_timesteps: list[int] | None = None,
    theme: str = "dark",
):
    """Plot selected species counts and total molecule count over time."""

    style = _theme_colors(theme)
    return [
        _plot_species_counts(simulations, species, reference_lines, step_range, excluded_timesteps, style),
        _plot_species_molecule_counts(simulations, reference_lines, step_range, excluded_timesteps, style),
    ]


def _plot_species_counts(
    simulations: list[LoadedSimulation],
    species: list[str],
    reference_lines: ReferenceLines | None = None,
    step_range: tuple[float, float] | None = None,
    excluded_timesteps: list[int] | None = None,
    style: dict[str, str] | None = None,
):
    """Plot selected species counts over time for each simulation."""

    style = style or THERMO_DARK_COLORS
    fig, ax = plt.subplots(facecolor=style["figure"])
    color_cycle = cycle(SPECIES_DARK_COLORS)
    plotted_lines = 0

    for simulation in simulations:
        species_df = _filtered_species_frame(simulation, step_range, excluded_timesteps)
        if species_df is None or species_df.empty:
            continue
        available_species = [name for name in species if name in species_df.columns]
        for name in available_species:
            ax.plot(
                species_df["Timestep"],
                species_df[name],
                label=f"R{simulation.index} {name}",
                color=next(color_cycle),
                linewidth=2.0,
            )
            plotted_lines += 1

    _style_axes(ax, "Species evolution", "Count", style, x_label="Timestep")
    _apply_step_range(ax, step_range)
    _add_reference_lines(ax, reference_lines, color=style["text"])

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
):
    """Plot the total number of molecules over time for each simulation."""

    style = style or THERMO_DARK_COLORS
    fig, ax = plt.subplots(facecolor=style["figure"])
    color_cycle = cycle(SPECIES_DARK_COLORS)
    plotted_lines = 0

    for simulation in simulations:
        species_df = _filtered_species_frame(simulation, step_range, excluded_timesteps)
        if species_df is None or species_df.empty or "No_Moles" not in species_df.columns:
            continue
        ax.plot(
            species_df["Timestep"],
            species_df["No_Moles"],
            label=f"R{simulation.index} No_Moles",
            color=next(color_cycle),
            linewidth=2.0,
        )
        plotted_lines += 1

    _style_axes(ax, "Total molecules", "Molecules", style, x_label="Timestep")
    _apply_step_range(ax, step_range)
    _add_reference_lines(ax, reference_lines, color=style["text"])

    if plotted_lines:
        legend = ax.legend(frameon=False)
        for text in legend.get_texts():
            text.set_color(style["text"])

    fig.tight_layout()
    return fig


def _filtered_species_frame(
    simulation: LoadedSimulation,
    step_range: tuple[float, float] | None,
    excluded_timesteps: list[int] | None,
):
    """Return species data filtered to the requested visible timesteps."""

    if simulation.species_df is None:
        return None
    frame = simulation.species_df
    if excluded_timesteps:
        frame = frame[~frame["Timestep"].isin(set(excluded_timesteps))]
    if step_range is not None:
        lower, upper = sorted(step_range)
        frame = frame[(frame["Timestep"] >= lower) & (frame["Timestep"] <= upper)]
    return frame


def plot_rdf(
    results: list[RDFResult],
    element_a: str,
    element_b: str,
    reference_lines: ReferenceLines | None = None,
    theme: str = "dark",
    gradient_colors: tuple[str, str] | None = None,
):
    """Plot one time-averaged RDF curve per selected simulation."""

    if not results:
        raise ValueError("No RDF data to plot.")

    style = _theme_colors(theme)
    fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor=style["figure"])
    line_colors = _line_colors(len(results), gradient_colors)
    for result, color in zip(results, line_colors, strict=False):
        ax.plot(
            result.r,
            result.g_r,
            color=color,
            linewidth=2.0,
            label=f"Simulation {result.simulation_index}",
        )

    _style_axes(ax, f"RDF {element_a}-{element_b}", "g(r)", style, x_label="r [A]")
    _add_reference_lines(ax, reference_lines, color=style["text"])
    legend = ax.legend(frameon=False)
    for text in legend.get_texts():
        text.set_color(style["text"])
    fig.tight_layout()
    return fig


def plot_charge_evolution(
    simulations: list[LoadedSimulation],
    elements: list[str],
    *,
    uncertainty: str = "band",
    step_range: tuple[float, float] | None = None,
    theme: str = "dark",
):
    """Plot mean atomic partial charge by element, with optional deviations."""

    if uncertainty not in {"band", "errorbar", "none"}:
        raise ValueError("uncertainty must be 'band', 'errorbar', or 'none'.")
    if not elements:
        raise ValueError("Select at least one element for charge plotting.")

    style = _theme_colors(theme)
    fig, ax = plt.subplots(figsize=(9.0, 5.2), facecolor=style["figure"])
    color_cycle = cycle(CHARGE_LINE_COLORS)
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
                    timesteps,
                    means,
                    yerr=deviations,
                    label=label,
                    color=color,
                    linewidth=1.8,
                    capsize=2,
                )
            else:
                ax.plot(timesteps, means, label=label, color=color, linewidth=2.0)
                if uncertainty == "band":
                    lower_values = [mean - std for mean, std in zip(means, deviations, strict=False)]
                    upper_values = [mean + std for mean, std in zip(means, deviations, strict=False)]
                    ax.fill_between(timesteps, lower_values, upper_values, color=color, alpha=0.18)
            plotted += 1

    if not plotted:
        raise ValueError("No charge observations match the selected simulations, elements, and range.")
    _style_axes(ax, "Atomic partial charges", "Mean partial charge [e]", style, x_label="Timestep")
    _apply_step_range(ax, step_range)
    legend = ax.legend(frameon=False, fontsize="small")
    for text in legend.get_texts():
        text.set_color(style["text"])
    fig.tight_layout()
    return fig


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
):
    """Create one combined simulation figure and one averaged figure."""

    figures = []
    running_average_points = _validated_running_average_points(running_average_points)
    average_groups = average_groups or None
    style = _theme_colors(theme)
    y_label = thermo_axis_label(parameter)
    plottable = [
        simulation
        for simulation in simulations
        if simulation.thermo_df is not None and parameter in simulation.thermo_df.columns
    ]
    if not plottable:
        raise ValueError(f"No thermo data found for parameter {parameter!r}.")

    fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor=style["figure"])
    line_colors = _line_colors(len(plottable), gradient_colors)
    for simulation, color in zip(plottable, line_colors, strict=False):
        label = _thermo_legend_label(simulation, legend_labels)
        ax.plot(
            simulation.thermo_df["Step"],
            simulation.thermo_df[parameter],
            color=color,
            linewidth=2.0,
            label=label,
        )
        if running_average_points is not None:
            ax.plot(
                simulation.thermo_df["Step"],
                simulation.thermo_df[parameter].rolling(window=running_average_points, min_periods=1).mean(),
                color=_inverse_hex_color(color),
                linestyle="-",
                linewidth=2.6,
                label=f"{label} Avg",
            )
    _style_axes(ax, f"Selected simulations: {parameter}", y_label, style)
    _apply_step_range(ax, step_range)
    _apply_y_range(ax, y_range)
    _add_reference_lines(ax, reference_lines, color=style["text"])
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
        ax.plot(averaged["Step"], averaged["mean"], color=color, linewidth=2.2, label=mean_label)
        ax.fill_between(
            averaged["Step"],
            averaged["mean"] - averaged["std"],
            averaged["mean"] + averaged["std"],
            alpha=0.22,
            color=std_color,
            label=std_label,
        )
    _style_axes(ax, f"Average {parameter}", y_label, style)
    _apply_step_range(ax, step_range)
    _apply_y_range(ax, y_range)
    _add_reference_lines(ax, reference_lines, color=style["text"])
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


def _validated_running_average_points(points: int | None) -> int | None:
    """Validate and return the optional running-average window size."""

    if points is None:
        return None
    if points < 1:
        raise ValueError("Running-average points must be at least 1.")
    return points


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
) -> None:
    """Apply a Matplotlib axis style from a color theme."""

    ax.set_facecolor(colors["axes"])
    ax.set_xlabel(x_label, color=colors["text"], fontsize=16, fontweight="bold")
    ax.set_ylabel(y_label, color=colors["text"], fontsize=16, fontweight="bold")
    ax.set_title(title, color=colors["title"], pad=12)
    ax.tick_params(axis="both", colors=colors["tick"])
    ax.grid(True, color=colors["grid"], alpha=0.55, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color(colors["spine"])
