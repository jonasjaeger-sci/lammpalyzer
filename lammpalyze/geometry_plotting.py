"""Plotting for trajectory-derived distances and angles."""

from __future__ import annotations

import pandas as pd

from lammpalyze.geometry import GeometrySeries
from lammpalyze.plotting import ReferenceLines, _plot_computed_series


def plot_geometry(
    results: list[GeometrySeries],
    kind: str,
    *,
    step_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    running_average_points: int | None = None,
    reference_lines: ReferenceLines | None = None,
    legend_location: str = "best",
    theme: str = "dark",
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
    return _plot_computed_series(
        series,
        title=title,
        y_label=y_label,
        step_range=step_range,
        y_range=y_range,
        running_average_points=running_average_points,
        reference_lines=reference_lines,
        legend_location=legend_location,
        theme=theme,
    )
