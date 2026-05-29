"""Tests for GUI data helpers."""

from lammpalyze.analysis import LoadedSimulation
from lammpalyze.gui import molecule_render_size, reaction_path_table_data
from lammpalyze.gui.helpers import image_output_path, suffixed_image_output_path


def test_molecule_render_size_follows_available_area():
    """Scale molecule render sizes to the available GUI area."""

    assert molecule_render_size(900, 700) == (876, 676)
    assert molecule_render_size(1, 1) == (720, 520)
    assert molecule_render_size(3000, 2200) == (1800, 1400)


def test_reaction_path_table_data_counts_paths_per_simulation():
    """Build reaction table totals and per-simulation counts."""

    simulations = [
        LoadedSimulation(
            index=1,
            smiles={0: ["AB"], 1: ["A", "B"]},
            smiles_id={0: [["1", "2"]], 1: [["1"], ["2"]]},
        ),
        LoadedSimulation(
            index=2,
            smiles={0: ["AB"], 1: ["A", "B"], 2: ["A", "B"]},
            smiles_id={0: [["1", "2"]], 1: [["1"], ["2"]], 2: [["1"], ["2"]]},
        ),
    ]

    simulation_indices, paths, counts = reaction_path_table_data(simulations)

    assert simulation_indices == [1, 2]
    assert [(path.reaction, path.count) for path in paths] == [("['AB'] -> ['A', 'B']", 2)]
    assert counts["['AB'] -> ['A', 'B']"] == {1: 1, 2: 1}


def test_image_output_path_defaults_to_png():
    """Default image saves to PNG when the user omits a suffix."""

    assert str(image_output_path("species")) == "species.png"
    assert str(image_output_path("species.svg")) == "species.svg"


def test_suffixed_image_output_path_preserves_extension():
    """Add plot suffixes without discarding the selected image format."""

    assert str(suffixed_image_output_path("thermo.pdf", "average")) == "thermo_average.pdf"
    assert str(suffixed_image_output_path("thermo", "selected")) == "thermo_selected.png"
