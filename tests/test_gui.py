"""Tests for GUI data helpers."""

from lammpalyze.analysis import LoadedSimulation
from lammpalyze.gui import connected_reaction_pathway_data, molecule_render_size, reaction_path_table_data
from lammpalyze.gui.helpers import (
    image_output_path,
    parse_reference_lines,
    parse_simulation_groups,
    parse_timestep_values,
    suffixed_image_output_path,
)


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


def test_connected_reaction_pathway_data_uses_formula_notation():
    """Build connected pathway rows for the GUI helper."""

    simulations = [
        LoadedSimulation(
            index=1,
            smiles={0: ["raw"], 1: ["[Li]", "O=C1"], 2: ["[Li]O=C1"]},
            smiles_id={0: [["1", "2"]], 1: [["1"], ["2"]], 2: [["1", "2"]]},
            chem_formulas={0: ["raw"], 1: ["Li", "C3H4O3"], 2: ["LiC3H4O3"]},
        )
    ]

    pathways = connected_reaction_pathway_data(simulations, notation="formula")

    assert len(pathways) == 1
    assert pathways[0].steps[0].source == "C3H4O3 + Li"
    assert pathways[0].steps[0].target == "LiC3H4O3"


def test_image_output_path_defaults_to_png():
    """Default image saves to PNG when the user omits a suffix."""

    assert str(image_output_path("species")) == "species.png"
    assert str(image_output_path("species.svg")) == "species.svg"


def test_suffixed_image_output_path_preserves_extension():
    """Add plot suffixes without discarding the selected image format."""

    assert str(suffixed_image_output_path("thermo.pdf", "average")) == "thermo_average.pdf"
    assert str(suffixed_image_output_path("thermo", "selected")) == "thermo_selected.png"


def test_parse_reference_lines_accepts_common_separators():
    """Parse GUI reference-line entries split by commas, spaces, or semicolons."""

    assert parse_reference_lines("1, 2; 3\n4") == [1.0, 2.0, 3.0, 4.0]
    assert parse_reference_lines("  ") == []


def test_parse_simulation_groups_accepts_semicolon_separated_groups():
    """Parse thermo average groups while preserving group order."""

    assert parse_simulation_groups("1, 3; 2 4; 4,4") == [[1, 3], [2, 4], [4]]
    assert parse_simulation_groups("  ") == []


def test_parse_timestep_values_accepts_common_list_syntax():
    """Parse species timestep exclusions from common list formats."""

    assert parse_timestep_values("(200, 3400); 4200") == [200, 3400, 4200]
    assert parse_timestep_values("  ") == []
