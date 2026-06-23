"""Tests for plotting helpers."""
# pylint: disable=wrong-import-position

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lammpalyze.analysis import LoadedSimulation  # noqa: E402
from lammpalyze.parsers import ChargeStatistics  # noqa: E402
from lammpalyze.plotting import plot_charge_evolution, plot_rdf, plot_species, plot_thermo  # noqa: E402
from lammpalyze.rdf import RDFResult  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    """Close Matplotlib figures created by each plotting test."""

    yield
    plt.close("all")


def test_plot_thermo_returns_combined_and_average_figures():
    """Create combined and averaged thermo figures."""

    simulations = [
        LoadedSimulation(index=1, thermo_df=pd.DataFrame({"Step": [0, 1], "Temp": [300.0, 310.0]})),
        LoadedSimulation(index=2, thermo_df=pd.DataFrame({"Step": [0, 1], "Temp": [320.0, 330.0]})),
    ]

    figures = plot_thermo(simulations, "Temp")

    assert len(figures) == 2
    assert len(figures[0].axes[0].lines) == 2
    assert figures[0].axes[0].get_ylabel() == "Temp [K]"
    assert figures[1].axes[0].get_title() == "Average Temp"


def test_plot_thermo_applies_step_range_to_both_figures():
    """Apply a requested step range to both thermo figures."""

    simulations = [
        LoadedSimulation(index=1, thermo_df=pd.DataFrame({"Step": [0, 10, 20], "Temp": [300.0, 310.0, 320.0]})),
        LoadedSimulation(index=2, thermo_df=pd.DataFrame({"Step": [0, 10, 20], "Temp": [330.0, 340.0, 350.0]})),
    ]

    figures = plot_thermo(simulations, "Temp", step_range=(5, 15))

    assert figures[0].axes[0].get_xlim() == (5.0, 15.0)
    assert figures[1].axes[0].get_xlim() == (5.0, 15.0)


def test_plot_thermo_applies_y_range_to_both_figures():
    """Apply a requested y-axis range to both thermo figures."""

    simulations = [
        LoadedSimulation(index=1, thermo_df=pd.DataFrame({"Step": [0, 1], "Temp": [300.0, 310.0]})),
        LoadedSimulation(index=2, thermo_df=pd.DataFrame({"Step": [0, 1], "Temp": [320.0, 330.0]})),
    ]

    figures = plot_thermo(simulations, "Temp", y_range=(305, 325))

    assert figures[0].axes[0].get_ylim() == (305.0, 325.0)
    assert figures[1].axes[0].get_ylim() == (305.0, 325.0)


def test_plot_thermo_adds_running_average_to_first_figure_only():
    """Add a per-simulation running average to the selected-simulations plot."""

    simulations = [
        LoadedSimulation(index=1, thermo_df=pd.DataFrame({"Step": [0, 1, 2], "Temp": [1.0, 2.0, 4.0]})),
        LoadedSimulation(index=2, thermo_df=pd.DataFrame({"Step": [0, 1, 2], "Temp": [2.0, 4.0, 8.0]})),
    ]

    figures = plot_thermo(simulations, "Temp", running_average_points=2)

    assert len(figures[0].axes[0].lines) == 4
    np.testing.assert_allclose(figures[0].axes[0].lines[1].get_ydata(), [1.0, 1.5, 3.0])
    assert figures[0].axes[0].lines[0].get_color() == "#4cc9f0"
    assert figures[0].axes[0].lines[1].get_color() == "#b3360f"
    assert figures[0].axes[0].lines[1].get_linestyle() == "-"
    assert figures[0].axes[0].lines[1].get_label() == "Simulation 1 Avg"
    assert len(figures[1].axes[0].lines) == 1


def test_plot_thermo_adds_reference_lines_to_both_figures():
    """Draw configured vertical and horizontal reference lines on thermo plots."""

    simulations = [
        LoadedSimulation(index=1, thermo_df=pd.DataFrame({"Step": [0, 1], "Temp": [300.0, 310.0]})),
        LoadedSimulation(index=2, thermo_df=pd.DataFrame({"Step": [0, 1], "Temp": [320.0, 330.0]})),
    ]

    figures = plot_thermo(simulations, "Temp", reference_lines=([0.5], [315.0]))

    assert len(figures[0].axes[0].lines) == 4
    assert len(figures[1].axes[0].lines) == 3


def test_plot_thermo_compares_multiple_average_groups():
    """Plot independent average groups with their standard-deviation bands."""

    simulations = [
        LoadedSimulation(index=1, thermo_df=pd.DataFrame({"Step": [0, 1], "Temp": [300.0, 310.0]})),
        LoadedSimulation(index=2, thermo_df=pd.DataFrame({"Step": [0, 1], "Temp": [320.0, 330.0]})),
        LoadedSimulation(index=3, thermo_df=pd.DataFrame({"Step": [0, 1], "Temp": [340.0, 350.0]})),
        LoadedSimulation(index=4, thermo_df=pd.DataFrame({"Step": [0, 1], "Temp": [360.0, 370.0]})),
    ]

    figures = plot_thermo(
        simulations,
        "Temp",
        average_groups=[[1, 3], [2, 4]],
        average_group_labels=["Cool", "Warm"],
    )
    average_axis = figures[1].axes[0]

    assert [line.get_label() for line in average_axis.lines] == ["Cool mean", "Warm mean"]
    np.testing.assert_allclose(average_axis.lines[0].get_ydata(), [320.0, 330.0])
    np.testing.assert_allclose(average_axis.lines[1].get_ydata(), [340.0, 350.0])
    assert len(average_axis.collections) == 2


def test_plot_thermo_applies_bright_theme_and_gradient_colors():
    """Style thermo plots with a bright background and interpolated line colors."""

    simulations = [
        LoadedSimulation(index=1, thermo_df=pd.DataFrame({"Step": [0, 1], "Temp": [300.0, 310.0]})),
        LoadedSimulation(index=2, thermo_df=pd.DataFrame({"Step": [0, 1], "Temp": [320.0, 330.0]})),
        LoadedSimulation(index=3, thermo_df=pd.DataFrame({"Step": [0, 1], "Temp": [340.0, 350.0]})),
    ]

    figures = plot_thermo(
        simulations,
        "Temp",
        theme="Bright",
        gradient_colors=("#f9c74f", "#7209b7"),
    )

    assert figures[0].get_facecolor() == (0.9725490196078431, 0.9803921568627451, 0.9882352941176471, 1.0)
    assert [line.get_color() for line in figures[0].axes[0].lines] == ["#f9c74f", "#b66883", "#7209b7"]
    assert figures[0].axes[0].get_facecolor() == (1.0, 1.0, 1.0, 1.0)


def test_plot_species_adds_reference_lines():
    """Draw configured vertical and horizontal reference lines on species plots."""

    simulations = [
        LoadedSimulation(index=1, species_df=pd.DataFrame({"Timestep": [0, 1], "No_Moles": [4, 5], "Li": [1, 2]})),
    ]

    figures = plot_species(simulations, ["Li"], reference_lines=([0.5], [1.5]))

    assert len(figures) == 2
    assert len(figures[0].axes[0].lines) == 3
    assert figures[1].axes[0].lines[0].get_label() == "R1 No_Moles"
    np.testing.assert_allclose(figures[1].axes[0].lines[0].get_ydata(), [4, 5])


def test_plot_species_applies_bright_theme():
    """Style species plots with a bright background."""

    simulations = [
        LoadedSimulation(index=1, species_df=pd.DataFrame({"Timestep": [0, 1], "No_Moles": [4, 5], "Li": [1, 2]})),
    ]

    figures = plot_species(simulations, ["Li"], theme="Bright")

    assert figures[0].get_facecolor() == (0.9725490196078431, 0.9803921568627451, 0.9882352941176471, 1.0)
    assert figures[0].axes[0].get_facecolor() == (1.0, 1.0, 1.0, 1.0)
    assert figures[1].axes[0].get_facecolor() == (1.0, 1.0, 1.0, 1.0)


def test_plot_species_filters_timesteps_and_applies_step_range():
    """Filter excluded species timesteps before plotting a visible x-axis range."""

    simulations = [
        LoadedSimulation(
            index=1,
            species_df=pd.DataFrame(
                {
                    "Timestep": [0, 10, 20, 30],
                    "No_Moles": [100, 5, 50, 200],
                    "Li": [1, 100, 2, 200],
                }
            ),
        ),
    ]

    figures = plot_species(
        simulations,
        ["Li"],
        step_range=(5, 25),
        excluded_timesteps=[10],
    )

    assert figures[0].axes[0].get_xlim() == (5.0, 25.0)
    np.testing.assert_allclose(figures[0].axes[0].lines[0].get_xdata(), [20])
    np.testing.assert_allclose(figures[0].axes[0].lines[0].get_ydata(), [2])
    np.testing.assert_allclose(figures[1].axes[0].lines[0].get_xdata(), [20])
    np.testing.assert_allclose(figures[1].axes[0].lines[0].get_ydata(), [50])


def test_plot_charge_evolution_draws_element_means_and_standard_deviation_band():
    """Plot per-frame charge means and within-element population deviations."""

    simulation = LoadedSimulation(
        index=1,
        charge_statistics={
            0: {"Li": ChargeStatistics(mean=0.5, std=0.1, count=4)},
            10: {"Li": ChargeStatistics(mean=0.6, std=0.2, count=4)},
        },
    )

    figure = plot_charge_evolution([simulation], ["Li"], uncertainty="band")

    axis = figure.axes[0]
    assert axis.lines[0].get_label() == "Simulation 1 Li"
    np.testing.assert_allclose(axis.lines[0].get_ydata(), [0.5, 0.6])
    assert len(axis.collections) == 1
    assert axis.get_ylabel() == "Mean partial charge [e]"


def test_plot_rdf_does_not_add_cross_simulation_mean():
    """Plot RDF curves without adding a cross-simulation mean."""

    results = [
        RDFResult(simulation_index=1, r=np.array([0.5, 1.5]), g_r=np.array([1.0, 2.0]), timesteps=[0]),
        RDFResult(simulation_index=2, r=np.array([0.5, 1.5]), g_r=np.array([2.0, 3.0]), timesteps=[0]),
    ]

    figure = plot_rdf(results, "Li", "O")

    assert len(figure.axes[0].lines) == 2
    assert [line.get_label() for line in figure.axes[0].lines] == ["Simulation 1", "Simulation 2"]


def test_plot_rdf_adds_reference_lines():
    """Draw configured vertical and horizontal reference lines on RDF plots."""

    results = [
        RDFResult(simulation_index=1, r=np.array([0.5, 1.5]), g_r=np.array([1.0, 2.0]), timesteps=[0]),
    ]

    figure = plot_rdf(results, "Li", "O", reference_lines=([1.0], [1.5]))

    assert len(figure.axes[0].lines) == 3


def test_plot_rdf_applies_bright_theme_and_gradient_colors():
    """Style RDF plots with a bright background and interpolated line colors."""

    results = [
        RDFResult(simulation_index=1, r=np.array([0.5, 1.5]), g_r=np.array([1.0, 2.0]), timesteps=[0]),
        RDFResult(simulation_index=2, r=np.array([0.5, 1.5]), g_r=np.array([2.0, 3.0]), timesteps=[0]),
        RDFResult(simulation_index=3, r=np.array([0.5, 1.5]), g_r=np.array([3.0, 4.0]), timesteps=[0]),
    ]

    figure = plot_rdf(
        results,
        "Li",
        "O",
        theme="Bright",
        gradient_colors=("#f9c74f", "#7209b7"),
    )

    assert figure.get_facecolor() == (0.9725490196078431, 0.9803921568627451, 0.9882352941176471, 1.0)
    assert [line.get_color() for line in figure.axes[0].lines] == ["#f9c74f", "#b66883", "#7209b7"]
    assert figure.axes[0].get_facecolor() == (1.0, 1.0, 1.0, 1.0)
