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
from lammpalyze.plotting import (  # noqa: E402
    atom_molecule_membership,
    plot_charge_evolution,
    plot_msd,
    plot_pairwise,
    plot_rdf,
    plot_species,
    plot_thermo,
)
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


def test_plot_msd_uses_independent_file_column_selections():
    """Plot different computed MSD columns from different simulation files."""

    simulations = [
        LoadedSimulation(
            index=1,
            msd_df=pd.DataFrame({"Timestep": [0, 100], "c_msd_C[1]": [0.0, 0.2]}),
        ),
        LoadedSimulation(
            index=2,
            msd_df=pd.DataFrame({"Timestep": [0, 100], "c_msd_Li[4]": [0.0, 1.5]}),
        ),
    ]

    figures = plot_msd(
        simulations,
        [(1, "c_msd_C[1]"), (2, "c_msd_Li[4]")],
        legend_location="lower right",
    )

    assert len(figures) == 2
    axis = figures[0].axes[0]
    assert [line.get_label() for line in axis.lines] == [
        "MSD1 - c_msd_C[1]",
        "MSD2 - c_msd_Li[4]",
    ]
    np.testing.assert_allclose(axis.lines[1].get_ydata(), [0.0, 1.5])
    assert axis.get_xlabel() == "Timestep"
    assert axis.get_legend()._loc == 4
    assert figures[1].axes[0].get_legend()._loc == 4
    np.testing.assert_allclose(figures[1].axes[0].lines[0].get_ydata(), [0.0, 0.85])


def test_plot_msd_compares_selected_average_groups():
    """Create independent MSD mean and deviation curves for simulation groups."""

    simulations = [
        LoadedSimulation(
            index=index,
            msd_df=pd.DataFrame({"Timestep": [0, 100], "total": [0.0, float(index)]}),
        )
        for index in range(1, 5)
    ]

    figures = plot_msd(
        simulations,
        [(index, "total") for index in range(1, 5)],
        average_groups=[[1, 3], [2, 4]],
        average_group_labels=["Odd", "Even"],
    )

    average_axis = figures[1].axes[0]
    assert [line.get_label() for line in average_axis.lines] == ["Odd mean", "Even mean"]
    np.testing.assert_allclose(average_axis.lines[0].get_ydata(), [0.0, 2.0])
    np.testing.assert_allclose(average_axis.lines[1].get_ydata(), [0.0, 3.0])
    assert len(average_axis.collections) == 2


def test_plot_pairwise_draws_selected_pair_trajectories():
    """Plot only requested pair descriptors for the selected local-dump value."""

    simulation = LoadedSimulation(
        index=1,
        pairwise_df=pd.DataFrame(
            {
                "Timestep": [0, 0, 100, 100],
                "Pair": ["1-2", "1-3", "1-2", "1-3"],
                "c_pdist": [1.0, 2.0, 1.2, 2.2],
            }
        ),
    )

    figure = plot_pairwise([simulation], "c_pdist", [(1, "1-3")])

    axis = figure.axes[0]
    assert axis.lines[0].get_label() == "Dump1 - 1-3"
    np.testing.assert_allclose(axis.lines[0].get_xdata(), [0, 100])
    np.testing.assert_allclose(axis.lines[0].get_ydata(), [2.0, 2.2])
    assert axis.get_ylabel() == "c_pdist"


def test_plot_pairwise_adds_numerized_atom_molecule_axis():
    """Track the formula containing a selected atom on a secondary y-axis."""

    simulation = LoadedSimulation(
        index=1,
        pairwise_df=pd.DataFrame(
            {
                "Timestep": [0, 100],
                "Pair": ["1-2", "1-2"],
                "c_pdist": [1.0, 1.2],
            }
        ),
        smiles_id={0: [["1", "2"]], 100: [["1", "2", "3"]], 200: [["1", "2"]]},
        smiles={0: ["CC"], 100: ["[Li]CC"], 200: ["CC"]},
        chem_formulas={0: ["C2H6"], 100: ["C2H6Li"], 200: ["C2H6"]},
    )

    figure = plot_pairwise(
        [simulation],
        "c_pdist",
        [(1, "1-2")],
        molecule_atom=(1, 1),
        molecule_notation="formula",
        legend_location="upper left",
    )

    assert len(figure.axes) == 2
    molecule_axis = figure.axes[1]
    np.testing.assert_allclose(molecule_axis.lines[0].get_ydata(), [1, 2, 1])
    assert [tick.get_text() for tick in molecule_axis.get_yticklabels()] == [
        "1: C2H6",
        "2: C2H6Li",
    ]
    assert molecule_axis.get_ylabel() == "Atom 1 molecule (formula)"
    assert figure.axes[0].get_legend()._loc == 2
    assert [text.get_text() for text in figure.axes[0].get_legend().get_texts()] == [
        "Dump1 - 1-2",
        "Atom 1 molecule",
    ]


def test_atom_molecule_membership_supports_smiles_notation():
    """Numerize recurring SMILES labels consistently across bond timesteps."""

    simulation = LoadedSimulation(
        index=3,
        smiles_id={0: [["7"]], 10: [["7", "8"]], 20: [["7"]]},
        smiles={0: ["[Li]"], 10: ["[Li]O"], 20: ["[Li]"]},
        chem_formulas={0: ["Li"], 10: ["LiO"], 20: ["Li"]},
    )

    timesteps, values, labels = atom_molecule_membership(simulation, 7, "smiles")

    assert timesteps == [0, 10, 20]
    assert values == [1, 2, 1]
    assert labels == {1: "[Li]", 2: "[Li]O"}


def test_plot_rdf_returns_selected_and_cross_simulation_average_figures():
    """Plot selected RDF curves plus their aligned mean and deviation band."""

    results = [
        RDFResult(simulation_index=1, r=np.array([0.5, 1.5]), g_r=np.array([1.0, 2.0]), timesteps=[0]),
        RDFResult(simulation_index=2, r=np.array([0.5, 1.5]), g_r=np.array([2.0, 3.0]), timesteps=[0]),
    ]

    figures = plot_rdf(results, "Li", "O")

    assert len(figures) == 2
    assert len(figures[0].axes[0].lines) == 2
    assert [line.get_label() for line in figures[0].axes[0].lines] == [
        "Simulation 1",
        "Simulation 2",
    ]
    np.testing.assert_allclose(figures[1].axes[0].lines[0].get_ydata(), [1.5, 2.5])
    assert len(figures[1].axes[0].collections) == 1
    assert figures[1].axes[0].get_title() == "Average RDF Li-O"


def test_plot_rdf_running_average_smooths_each_selected_curve():
    """Add point-window running averages only to the selected-simulations plot."""

    results = [
        RDFResult(
            simulation_index=1,
            r=np.array([0.5, 1.5, 2.5]),
            g_r=np.array([1.0, 3.0, 5.0]),
            timesteps=[0],
        )
    ]

    figures = plot_rdf(
        results,
        "Li",
        "O",
        running_average_points=2,
        legend_location="center left",
    )

    selected_axis = figures[0].axes[0]
    assert [line.get_label() for line in selected_axis.lines] == [
        "Simulation 1",
        "Simulation 1 Avg",
    ]
    np.testing.assert_allclose(selected_axis.lines[1].get_ydata(), [1.0, 2.0, 4.0])
    assert selected_axis.lines[1].get_linestyle() == "--"
    assert selected_axis.get_legend()._loc == 6
    assert len(figures[1].axes[0].lines) == 1
    assert figures[1].axes[0].get_legend()._loc == 6


def test_plot_rdf_average_uses_only_shared_radius_values():
    """Avoid extrapolating shorter RDF curves during cross-simulation averaging."""

    results = [
        RDFResult(
            simulation_index=1,
            r=np.array([0.5, 1.5, 2.5]),
            g_r=np.array([1.0, 2.0, 3.0]),
            timesteps=[0],
        ),
        RDFResult(
            simulation_index=2,
            r=np.array([0.5, 1.5]),
            g_r=np.array([3.0, 4.0]),
            timesteps=[0],
        ),
    ]

    figures = plot_rdf(results, "Li", "O")

    average_line = figures[1].axes[0].lines[0]
    np.testing.assert_allclose(average_line.get_xdata(), [0.5, 1.5])
    np.testing.assert_allclose(average_line.get_ydata(), [2.0, 3.0])


def test_plot_rdf_adds_reference_lines():
    """Draw configured vertical and horizontal reference lines on RDF plots."""

    results = [
        RDFResult(simulation_index=1, r=np.array([0.5, 1.5]), g_r=np.array([1.0, 2.0]), timesteps=[0]),
    ]

    figures = plot_rdf(results, "Li", "O", reference_lines=([1.0], [1.5]))

    assert len(figures[0].axes[0].lines) == 3
    assert len(figures[1].axes[0].lines) == 3


def test_plot_rdf_applies_bright_theme_and_gradient_colors():
    """Style RDF plots with a bright background and interpolated line colors."""

    results = [
        RDFResult(simulation_index=1, r=np.array([0.5, 1.5]), g_r=np.array([1.0, 2.0]), timesteps=[0]),
        RDFResult(simulation_index=2, r=np.array([0.5, 1.5]), g_r=np.array([2.0, 3.0]), timesteps=[0]),
        RDFResult(simulation_index=3, r=np.array([0.5, 1.5]), g_r=np.array([3.0, 4.0]), timesteps=[0]),
    ]

    figures = plot_rdf(
        results,
        "Li",
        "O",
        theme="Bright",
        gradient_colors=("#f9c74f", "#7209b7"),
    )

    assert figures[0].get_facecolor() == (0.9725490196078431, 0.9803921568627451, 0.9882352941176471, 1.0)
    assert [line.get_color() for line in figures[0].axes[0].lines] == [
        "#f9c74f",
        "#b66883",
        "#7209b7",
    ]
    assert figures[0].axes[0].get_facecolor() == (1.0, 1.0, 1.0, 1.0)
    assert figures[1].axes[0].get_facecolor() == (1.0, 1.0, 1.0, 1.0)
