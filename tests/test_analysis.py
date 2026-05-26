"""Tests for analysis aggregation helpers."""

import pandas as pd

from lammpalyze.analysis import LoadedSimulation, aggregate_thermo


def test_aggregate_thermo_mean_and_std():
    """Aggregate thermo values into timestep means and standard deviations."""

    simulations = [
        LoadedSimulation(index=1, thermo_df=pd.DataFrame({"Step": [0, 1], "Temp": [300.0, 310.0]})),
        LoadedSimulation(index=2, thermo_df=pd.DataFrame({"Step": [0, 1], "Temp": [320.0, 330.0]})),
    ]

    result = aggregate_thermo(simulations, "Temp")

    assert result["mean"].tolist() == [310.0, 320.0]
    assert result["std"].round(6).tolist() == [14.142136, 14.142136]
