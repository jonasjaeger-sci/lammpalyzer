import matplotlib
import pandas as pd

matplotlib.use("Agg")

from lammpalyze.analysis import LoadedSimulation
from lammpalyze.plotting import plot_thermo


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
