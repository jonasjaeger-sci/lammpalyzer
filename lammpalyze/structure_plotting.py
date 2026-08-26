"""Matplotlib rendering for structural-relaxation results."""

from __future__ import annotations

import matplotlib.pyplot as plt

from lammpalyze.plotting import (
    PlotSettings,
    _add_legend,
    _display_time_values,
    _line_colors,
    _style_axes,
    _time_axis_origin,
    _theme_colors,
    _time_axis_label,
)
from lammpalyze.structure import StructuralRelaxationResult


def plot_structural_relaxation(
    results: list[StructuralRelaxationResult],
    *,
    element_label: str = "All atoms",
    legend_location: str = "none",
    theme: str = "dark",
    plot_settings: PlotSettings | None = None,
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
        plot_settings=plot_settings,
    )
    _style_legend(static_axis, legend_location, style)
    static_figure.tight_layout()

    incoherent_figure, incoherent_axis = plt.subplots(figsize=(8.5, 4.8), facecolor=style["figure"])
    x_origin = _time_axis_origin([result.incoherent_scattering.time for result in results])
    for result, color in zip(results, colors, strict=False):
        incoherent = result.incoherent_scattering
        lower = incoherent.f_s - incoherent.f_s_error
        upper = incoherent.f_s + incoherent.f_s_error
        label = f"Simulation {result.simulation_index}, q={incoherent.q:.4g}"
        incoherent_axis.plot(
            _display_time_values(incoherent.time, plot_settings, x_origin),
            incoherent.f_s,
            color=color,
            linewidth=2.0,
            label=label,
        )
        incoherent_axis.fill_between(
            _display_time_values(incoherent.time, plot_settings, x_origin),
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
        x_label=_time_axis_label(plot_settings),
        plot_settings=plot_settings,
    )
    _style_legend(incoherent_axis, legend_location, style)
    incoherent_figure.tight_layout()
    return [static_figure, incoherent_figure]


def _style_legend(axis, legend_location: str, style: dict[str, str]) -> None:
    """Apply the shared legend styling."""

    _add_legend(axis, legend_location, style)
