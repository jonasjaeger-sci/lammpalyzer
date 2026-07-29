"""Plotting for trajectory-derived distances and angles."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from lammpalyze.analysis import LoadedSimulation
from lammpalyze.geometry import GeometrySeries
from lammpalyze.plotting import (
    PlotSettings,
    ReferenceLines,
    _plot_computed_series,
    add_atom_molecule_axis,
)


def plot_geometry(
    results: list[GeometrySeries],
    kind: str,
    *,
    simulations: list[LoadedSimulation] | None = None,
    molecule_atom_ids: list[int] | None = None,
    molecule_notation: str = "formula",
    step_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    running_average_points: int | None = None,
    reference_lines: ReferenceLines | None = None,
    legend_location: str = "best",
    theme: str = "dark",
    plot_settings: PlotSettings | None = None,
):
    """Plot trajectory-derived atom-pair distances or three-atom angles."""

    if kind not in {"distance", "angle"}:
        raise ValueError("Geometry kind must be 'distance' or 'angle'.")
    series = []
    for result in results:
        atom_label = "-".join(str(atom_id) for atom_id in result.atom_ids)
        series.append(
            (
                pd.Series(result.timesteps),
                pd.Series(result.values),
                f"Simulation {result.simulation_index} - {atom_label}",
            )
        )
    title = "Pair distances" if kind == "distance" else "Three-atom angles"
    y_label = "Distance (Å)" if kind == "distance" else "Angle (degrees)"
    figure = _plot_computed_series(
        series,
        title=title,
        y_label=y_label,
        step_range=step_range,
        y_range=y_range,
        running_average_points=running_average_points,
        reference_lines=reference_lines,
        legend_location=legend_location,
        theme=theme,
        plot_settings=plot_settings,
    )
    if molecule_atom_ids:
        if simulations is None:
            plt.close(figure)
            raise ValueError("Provide simulations when adding molecule-state tracks.")
        result_simulation_indices = {result.simulation_index for result in results}
        tracked_simulations = [
            simulation
            for simulation in simulations
            if simulation.index in result_simulation_indices
        ]
        try:
            add_atom_molecule_axis(
                figure.axes[0],
                tracked_simulations,
                molecule_atom_ids,
                notation=molecule_notation,
                legend_location=legend_location,
                theme=theme,
            )
        except Exception:
            plt.close(figure)
            raise
        figure.tight_layout()
    return figure
