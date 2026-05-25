import matplotlib
import pandas as pd
import numpy as np

matplotlib.use("Agg")

from lammpalyze.analysis import LoadedSimulation
from lammpalyze.plotting import plot_rdf, plot_thermo
from lammpalyze.rdf import RDFResult


def test_plot_thermo_returns_combined_and_average_figures():
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
    simulations = [
        LoadedSimulation(index=1, thermo_df=pd.DataFrame({"Step": [0, 10, 20], "Temp": [300.0, 310.0, 320.0]})),
        LoadedSimulation(index=2, thermo_df=pd.DataFrame({"Step": [0, 10, 20], "Temp": [330.0, 340.0, 350.0]})),
    ]

    figures = plot_thermo(simulations, "Temp", step_range=(5, 15))

    assert figures[0].axes[0].get_xlim() == (5.0, 15.0)
    assert figures[1].axes[0].get_xlim() == (5.0, 15.0)


def test_plot_thermo_applies_y_range_to_both_figures():
    simulations = [
        LoadedSimulation(index=1, thermo_df=pd.DataFrame({"Step": [0, 1], "Temp": [300.0, 310.0]})),
        LoadedSimulation(index=2, thermo_df=pd.DataFrame({"Step": [0, 1], "Temp": [320.0, 330.0]})),
    ]

    figures = plot_thermo(simulations, "Temp", y_range=(305, 325))

    assert figures[0].axes[0].get_ylim() == (305.0, 325.0)
    assert figures[1].axes[0].get_ylim() == (305.0, 325.0)


def test_plot_rdf_does_not_add_cross_simulation_mean():
    results = [
        RDFResult(simulation_index=1, r=np.array([0.5, 1.5]), g_r=np.array([1.0, 2.0]), timesteps=[0]),
        RDFResult(simulation_index=2, r=np.array([0.5, 1.5]), g_r=np.array([2.0, 3.0]), timesteps=[0]),
    ]

    figure = plot_rdf(results, "Li", "O")

    assert len(figure.axes[0].lines) == 2
    assert [line.get_label() for line in figure.axes[0].lines] == ["Simulation 1", "Simulation 2"]
