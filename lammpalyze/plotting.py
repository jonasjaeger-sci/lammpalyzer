"""Matplotlib plotting helpers."""

from __future__ import annotations

from itertools import cycle

import matplotlib.pyplot as plt

from lammpalyze.analysis import LoadedSimulation, aggregate_thermo

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


def plot_thermo(simulations: list[LoadedSimulation], parameter: str):
    """Create one figure per simulation and one averaged figure."""

    figures = []
    for simulation in simulations:
        if simulation.thermo_df is None or parameter not in simulation.thermo_df.columns:
            continue
        fig, ax = plt.subplots()
        ax.plot(simulation.thermo_df["Step"], simulation.thermo_df[parameter])
        ax.set_xlabel("Step")
        ax.set_ylabel(parameter)
        ax.set_title(f"Simulation {simulation.index}: {parameter}")
        ax.grid(True)
        fig.tight_layout()
        figures.append(fig)

    averaged = aggregate_thermo(simulations, parameter)
    fig, ax = plt.subplots()
    ax.plot(averaged["Step"], averaged["mean"], label="Mean")
    ax.fill_between(
        averaged["Step"],
        averaged["mean"] - averaged["std"],
        averaged["mean"] + averaged["std"],
        alpha=0.25,
        label="Std. dev.",
    )
    ax.set_xlabel("Step")
    ax.set_ylabel(parameter)
    ax.set_title(f"Average {parameter}")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    figures.append(fig)
    return figures
