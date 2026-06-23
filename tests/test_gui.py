"""Tests for GUI data helpers."""

from lammpalyze.analysis import LoadedSimulation
from lammpalyze.gui import (
    connected_reaction_pathway_data,
    molecule_render_size,
    reaction_path_display_order,
    reaction_path_ids,
    reaction_path_table_data,
)
from lammpalyze.gui.helpers import (
    image_output_path,
    molecule_observation_summary,
    parse_reference_lines,
    parse_simulation_groups,
    parse_timestep_values,
    suffixed_image_output_path,
)
from lammpalyze.parsers import ComponentProperties
from lammpalyze.reactions import ReactionPath


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


def test_reaction_path_display_order_groups_reverse_reactions():
    """Place reverse reactions next to their original path by default."""

    paths = [
        ReactionPath("['AB'] -> ['A', 'B']", 8),
        ReactionPath("['C'] -> ['D']", 4),
        ReactionPath("['A', 'B'] -> ['AB']", 3),
    ]
    ordered_paths, identifiers = reaction_path_display_order(paths)

    assert [path.reaction for path in ordered_paths] == [
        "['AB'] -> ['A', 'B']",
        "['A', 'B'] -> ['AB']",
        "['C'] -> ['D']",
    ]
    assert identifiers == {
        "['AB'] -> ['A', 'B']": "1",
        "['A', 'B'] -> ['AB']": "1*",
        "['C'] -> ['D']": "2",
    }
    assert reaction_path_ids(paths) == identifiers


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
    assert pathways[0].steps[0].counts_by_simulation == ((1, 1),)

    assert not connected_reaction_pathway_data(simulations, notation="formula", min_count=2)


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
    assert not parse_reference_lines("  ")


def test_parse_simulation_groups_accepts_semicolon_separated_groups():
    """Parse thermo average groups while preserving group order."""

    assert parse_simulation_groups("1, 3; 2 4; 4,4") == [[1, 3], [2, 4], [4]]
    assert not parse_simulation_groups("  ")


def test_parse_timestep_values_accepts_common_list_syntax():
    """Parse species timestep exclusions from common list formats."""

    assert parse_timestep_values("(200, 3400); 4200") == [200, 3400, 4200]
    assert not parse_timestep_values("  ")


def test_molecule_observation_summary_reports_charge_ions_and_flags():
    """Summarize component metadata across repeated SMILES observations."""

    simulation = LoadedSimulation(
        index=1,
        smiles={0: ["[Li]"], 10: ["[Li]"]},
        component_properties={
            0: [ComponentProperties(0.6, "cation candidate", ())],
            10: [ComponentProperties(0.8, "cation candidate", ("valence warning",))],
        },
    )

    summary = molecule_observation_summary(simulation, "[Li]")

    assert "charge mean +0.700 e" in summary
    assert "cation candidate: 2" in summary
    assert "Suspicious observations: 1" in summary
