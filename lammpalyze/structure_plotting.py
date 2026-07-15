"""Matplotlib rendering for structural-relaxation results."""

from __future__ import annotations

import matplotlib.pyplot as plt

from lammpalyze.plotting import (
    _line_colors,
    _style_axes,
    _theme_colors,
    _validated_legend_location,
)
from lammpalyze.structure import StructuralRelaxationResult


def plot_structural_relaxation(
    results: list[StructuralRelaxationResult],
    *,
    element_label: str = "All atoms",
    legend_location: str = "best",
    theme: str = "dark",
):
    """Plot S(q) and F_s(q,t) for one or more simulations."""

    if not results:
        raise ValueError("No structural-relaxation data to plot.")
    style = _theme_colors(theme)
    colors = _line_colors(len(results), None)

    static_figure, static_axis = plt.subplots(figsize=(8.5, 4.8), facecolor=style["figure"])
    for result, color in zip(results, colors, strict=False):
        static = result.static_structure_factor
        static_axis.errorbar(
            static.q,
            static.s_q,
            yerr=static.s_q_error,
            color=color,
            linewidth=1.8,
            elinewidth=1.0,
            capsize=2,
            marker="o",
            markersize=3.5,
            label=f"Simulation {result.simulation_index}",
        )
        static_axis.axvline(
            static.peak_q,
            color=color,
            linestyle=":",
            linewidth=1.2,
            alpha=0.7,
            label="_nolegend_",
        )
    _style_axes(
        static_axis,
        f"Static structure factor ({element_label})",
        "S(q)",
        style,
        x_label="|q| [1/A]",
    )
    _style_legend(static_axis, legend_location, style)
    static_figure.tight_layout()

    incoherent_figure, incoherent_axis = plt.subplots(figsize=(8.5, 4.8), facecolor=style["figure"])
    for result, color in zip(results, colors, strict=False):
        incoherent = result.incoherent_scattering
        lower = incoherent.f_s - incoherent.f_s_error
        upper = incoherent.f_s + incoherent.f_s_error
        label = f"Simulation {result.simulation_index}, q={incoherent.q:.4g}"
        incoherent_axis.plot(
            incoherent.time,
            incoherent.f_s,
            color=color,
            linewidth=2.0,
            label=label,
        )
        incoherent_axis.fill_between(
            incoherent.time,
            lower,
            upper,
            color=color,
            alpha=0.18,
            label="_nolegend_",
        )
    _style_axes(
        incoherent_axis,
        f"Incoherent scattering function ({element_label})",
        "F_s(q,t)",
        style,
        x_label="Time lag [timestep]",
    )
    _style_legend(incoherent_axis, legend_location, style)
    incoherent_figure.tight_layout()
    return [static_figure, incoherent_figure]


def _style_legend(axis, legend_location: str, style: dict[str, str]) -> None:
    """Apply the shared legend styling."""

    legend = axis.legend(loc=_validated_legend_location(legend_location), frameon=False)
    for text in legend.get_texts():
        text.set_color(style["text"])
