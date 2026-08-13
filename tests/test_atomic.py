"""Tests for trajectory-backed atomic data analysis."""

from pathlib import Path
import threading

import numpy as np
import pytest

import lammpalyze.atomic as atomic_module
from lammpalyze.analysis import LoadedSimulation
from lammpalyze.atomic import (
    AtomicCollectionCancelled,
    atomic_property_label,
    collect_atomic_series,
    collect_element_atomic_series,
    parse_atom_ids,
    trajectory_atomic_properties,
)
from lammpalyze.plotting import plot_atomic_data, plot_atomic_data_figures


def test_atomic_data_supports_flexible_columns_and_derived_vectors(tmp_path: Path):
    """Read reordered optional fields and calculate vector magnitudes."""

    trajectory = tmp_path / "atomic.lammpstrj"
    trajectory.write_text(_atomic_trajectory_text(), encoding="utf-8")
    simulation = LoadedSimulation(
        index=1,
        trajectory_path=trajectory,
        type_to_element={1: "Li", 2: "O"},
    )

    assert trajectory_atomic_properties(trajectory) == [
        "q",
        "vx",
        "vy",
        "vz",
        "v",
        "fx",
        "fy",
        "fz",
        "f",
        "custom",
    ]

    charge = collect_atomic_series([simulation], "q", elements=["Li"])[0]
    assert charge.timesteps == [0, 10]
    np.testing.assert_allclose(charge.means, [0.6, 0.7])
    np.testing.assert_allclose(charge.deviations, [0.1, 0.1])
    assert charge.counts == [2, 2]

    force = collect_atomic_series([simulation], "f", elements=["Li"])[0]
    np.testing.assert_allclose(force.means, [8.5, 10.0])
    np.testing.assert_allclose(force.deviations, [3.5, 0.0])

    velocity = collect_atomic_series([simulation], "v", atom_ids=[1, 3])
    assert [series.label for series in velocity] == [
        "Simulation 1 atom 1",
        "Simulation 1 atom 3",
    ]
    np.testing.assert_allclose(velocity[0].means, [1.0, 2.0])
    np.testing.assert_allclose(velocity[1].means, [2.0, 3.0])
    assert velocity[0].deviations == [0.0, 0.0]


def test_plot_atomic_data_labels_force_units_and_element_spread(tmp_path: Path):
    """Plot element means with the force unit and uncertainty band."""

    trajectory = tmp_path / "atomic.lammpstrj"
    trajectory.write_text(_atomic_trajectory_text(), encoding="utf-8")
    simulation = LoadedSimulation(index=4, trajectory_path=trajectory)

    figure = plot_atomic_data([simulation], "fx", elements=["Li"])

    axis = figure.axes[0]
    assert axis.lines[0].get_label() == "Simulation 4 Li"
    assert axis.get_ylabel() == "Force x [(kcal/mol)/Angstrom]"
    assert len(axis.collections) == 1


def test_element_plot_can_add_individual_atom_figure_in_one_pass(tmp_path: Path, monkeypatch):
    """Plot element statistics and each matching atom without rereading the dump."""

    trajectory = tmp_path / "atomic.lammpstrj"
    trajectory.write_text(_atomic_trajectory_text(), encoding="utf-8")
    simulation = LoadedSimulation(index=2, trajectory_path=trajectory)
    original_iterator = atomic_module.iter_lammpstrj_frames
    iterator_calls = 0

    def counted_iterator(*args, **kwargs):
        nonlocal iterator_calls
        iterator_calls += 1
        yield from original_iterator(*args, **kwargs)

    monkeypatch.setattr(atomic_module, "iter_lammpstrj_frames", counted_iterator)

    figures = plot_atomic_data_figures(
        [simulation],
        "q",
        elements=["Li"],
        include_individual_element_atoms=True,
    )

    assert iterator_calls == 1
    assert len(figures) == 2
    assert len(figures[0].axes[0].lines) == 1
    assert [line.get_label() for line in figures[1].axes[0].lines] == [
        "Simulation 2 Li atom 1",
        "Simulation 2 Li atom 2",
    ]
    np.testing.assert_allclose(figures[1].axes[0].lines[0].get_ydata(), [0.5, 0.6])
    np.testing.assert_allclose(figures[1].axes[0].lines[1].get_ydata(), [0.7, 0.8])


def test_collect_element_atomic_series_returns_group_and_atoms(tmp_path: Path):
    """Expose aggregate and per-atom values to non-GUI callers."""

    trajectory = tmp_path / "atomic.lammpstrj"
    trajectory.write_text(_atomic_trajectory_text(), encoding="utf-8")
    simulation = LoadedSimulation(index=3, trajectory_path=trajectory)

    series = collect_element_atomic_series([simulation], "v", ["O"])

    assert len(series.aggregate) == 1
    assert len(series.individual) == 1
    assert series.individual[0].label == "Simulation 3 O atom 3"
    np.testing.assert_allclose(series.individual[0].means, [2.0, 3.0])


def test_individual_element_atom_collection_can_fail_fast_for_broad_selections(tmp_path: Path):
    """Avoid building a giant individual-atom plot from a broad element selection."""

    trajectory = tmp_path / "atomic.lammpstrj"
    trajectory.write_text(_atomic_trajectory_text(), encoding="utf-8")
    simulation = LoadedSimulation(index=3, trajectory_path=trajectory)

    with pytest.raises(ValueError, match="more than 1 series"):
        collect_element_atomic_series(
            [simulation],
            "q",
            ["Li"],
            max_individual_series=1,
        )


def test_atomic_collection_honors_cancellation_before_reading(tmp_path: Path):
    """Let the GUI stop a large trajectory scan before Tk is destroyed."""

    trajectory = tmp_path / "atomic.lammpstrj"
    trajectory.write_text(_atomic_trajectory_text(), encoding="utf-8")
    simulation = LoadedSimulation(index=3, trajectory_path=trajectory)
    cancel_event = threading.Event()
    cancel_event.set()

    with pytest.raises(AtomicCollectionCancelled):
        collect_atomic_series(
            [simulation],
            "q",
            elements=["Li"],
            cancelled=cancel_event.is_set,
        )


def test_parse_atom_ids_accepts_lists_and_ranges():
    """Expand atom-ID lists while sorting and removing duplicates."""

    assert parse_atom_ids("7, 2-4 3") == [2, 3, 4, 7]


def test_atomic_property_labels_include_requested_units():
    """Expose the requested trajectory-data units in plot labels."""

    assert atomic_property_label("q") == "Atomic charge [e]"
    assert atomic_property_label("v") == "Velocity magnitude [Angstroms/femtosecond]"
    assert atomic_property_label("f") == "Force magnitude [(kcal/mol)/Angstrom]"


def _atomic_trajectory_text() -> str:
    """Return two frames containing charge, velocity, force, and custom data."""

    return """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
3
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS element id type z q y x fx fy fz vx vy vz custom
Li 1 1 3 0.5 2 1 3 4 0 1 0 0 11
Li 2 1 4 0.7 3 2 0 0 12 0 1 0 12
O 3 2 5 -1.2 4 3 1 2 2 0 0 2 13
ITEM: TIMESTEP
10
ITEM: NUMBER OF ATOMS
3
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS element id type z q y x fx fy fz vx vy vz custom
Li 1 1 3 0.6 2 1 6 8 0 2 0 0 14
Li 2 1 4 0.8 3 2 0 6 8 0 2 0 15
O 3 2 5 -1.1 4 3 2 3 6 0 0 3 16
"""
