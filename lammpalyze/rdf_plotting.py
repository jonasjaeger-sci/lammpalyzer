"""Matplotlib rendering for radial-distribution results."""

from __future__ import annotations

from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd

from lammpalyze.plotting import (
    PlotSettings,
    ReferenceLines,
    _add_reference_lines,
    _inverse_hex_color,
    _line_colors,
    _style_axes,
    _theme_colors,
    _validated_legend_location,
    _validated_running_average_points,
)
from lammpalyze.rdf import RDFResult


def plot_rdf(
    results: list[RDFResult],
    element_a: str,
    element_b: str,
    reference_lines: ReferenceLines | None = None,
    running_average_points: int | None = None,
    legend_location: str = "best",
    theme: str = "dark",
    gradient_colors: tuple[str, str] | None = None,
    plot_settings: PlotSettings | None = None,
):
    """Plot selected RDF curves and their aligned mean/standard deviation."""

    if not results:
        raise ValueError("No RDF data to plot.")

    running_average_points = _validated_running_average_points(running_average_points)
    style = _theme_colors(theme)
    fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor=style["figure"])
    line_colors = _line_colors(len(results), gradient_colors)
    curve_labels = _rdf_curve_labels(results, f"{element_a} - {element_b}")
    for result, color, curve_label in zip(
        results,
        line_colors,
        curve_labels,
        strict=False,
    ):
        ax.plot(
            result.r,
            result.g_r,
            color=color,
            linewidth=2.0,
            label=curve_label,
        )
        if running_average_points is not None:
            running_average = pd.Series(result.g_r).rolling(
                window=running_average_points,
                min_periods=1,
            ).mean()
            ax.plot(
                result.r,
                running_average,
                color=_inverse_hex_color(color),
                linestyle="--",
                linewidth=1.8,
                label=f"{curve_label} Avg",
            )

    _style_axes(
        ax,
        "Normalized RDF",
        "g(r)",
        style,
        x_label="r [A]",
        plot_settings=plot_settings,
    )
    _add_reference_lines(ax, reference_lines, color=style["text"])
    legend = ax.legend(loc=_validated_legend_location(legend_location), frameon=False)
    for text in legend.get_texts():
        text.set_color(style["text"])
    fig.tight_layout()

    try:
        averaged = _plot_rdf_average(
            results,
            reference_lines=reference_lines,
            legend_location=legend_location,
            theme=theme,
            plot_settings=plot_settings,
        )
    except Exception:
        plt.close(fig)
        raise
    return [fig, averaged]


def _plot_rdf_average(
    results: list[RDFResult],
    *,
    reference_lines: ReferenceLines | None,
    legend_location: str,
    theme: str,
    plot_settings: PlotSettings | None = None,
):
    """Plot an RDF mean and deviation band at radii shared by all results."""

    style = _theme_colors(theme)
    frames = [
        pd.DataFrame({"r": result.r, f"simulation_{position}": result.g_r})
        for position, result in enumerate(results)
    ]
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="r", how="inner")
    if merged.empty:
        raise ValueError("Selected RDF curves have no common radius values for averaging.")
    value_columns = [column for column in merged.columns if column != "r"]
    mean = merged[value_columns].mean(axis=1)
    deviation = merged[value_columns].std(axis=1).fillna(0.0)

    fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor=style["figure"])
    ax.plot(
        merged["r"],
        mean,
        color=_inverse_hex_color(style["std"]),
        linewidth=2.2,
        label="Mean",
    )
    ax.fill_between(
        merged["r"],
        mean - deviation,
        mean + deviation,
        color=style["std"],
        alpha=0.22,
        label="Std. dev.",
    )
    _style_axes(
        ax,
        "Average normalized RDF",
        "g(r)",
        style,
        x_label="r [A]",
        plot_settings=plot_settings,
    )
    _add_reference_lines(ax, reference_lines, color=style["text"])
    legend = ax.legend(loc=_validated_legend_location(legend_location), frameon=False)
    for text in legend.get_texts():
        text.set_color(style["text"])
    fig.tight_layout()
    return fig


def _rdf_curve_labels(results: list[RDFResult], default_label: str) -> list[str]:
    """Return distinct legend labels while preserving user-provided pair names."""

    base_labels = [result.label or default_label for result in results]
    base_counts = Counter(base_labels)
    candidates = [
        (
            f"{label} (Simulation {result.simulation_index})"
            if base_counts[label] > 1
            else label
        )
        for result, label in zip(results, base_labels, strict=True)
    ]
    candidate_counts = Counter(candidates)
    occurrences = Counter()
    labels = []
    for candidate in candidates:
        occurrences[candidate] += 1
        if candidate_counts[candidate] > 1:
            labels.append(f"{candidate} [{occurrences[candidate]}]")
        else:
            labels.append(candidate)
    return labels
