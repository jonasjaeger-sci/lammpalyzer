"""Matplotlib plotting helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt

from lammpalyze.analysis import LoadedSimulation, aggregate_thermo


def plot_species(simulations: list[LoadedSimulation], species: list[str]):
    """Plot selected species counts over time for each simulation."""

    fig, ax = plt.subplots()
    for simulation in simulations:
        if simulation.species_df is None:
            continue
        available_species = [name for name in species if name in simulation.species_df.columns]
        for name in available_species:
            ax.plot(
                simulation.species_df["Timestep"],
                simulation.species_df[name],
                label=f"R{simulation.index} {name}",
            )
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Count")
    ax.set_title("Species evolution")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
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
