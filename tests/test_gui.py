"""Tests for GUI data helpers."""

import matplotlib.pyplot as plt

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
    ordered_thermo_parameters,
    parse_reference_lines,
    parse_simulation_groups,
    parse_timestep_values,
    suffixed_image_output_path,
)
from lammpalyze.gui.pathway_graph import build_pathway_graph, pathway_graph_choices, pathway_graph_image_extent
from lammpalyze.gui.canvas import _format_hover_value, _nearest_line_point
from lammpalyze.gui.thermo_tab import apply_thermo_axis_ranges
from lammpalyze.parsers import ComponentProperties
from lammpalyze.reactions import ReactionPath


def test_molecule_render_size_follows_available_area():
    """Scale molecule render sizes to the available GUI area."""

    assert molecule_render_size(900, 700) == (876, 676)
    assert molecule_render_size(1, 1) == (720, 520)
    assert molecule_render_size(3000, 2200) == (1800, 1400)


def test_ordered_thermo_parameters_keeps_nondefault_columns():
    """Keep every thermo field while placing common choices first."""

    columns = ["Step", "Atoms", "E_coul", "Temp", "E_vdwl", "TotEng"]

    assert ordered_thermo_parameters(columns) == [
        "Temp",
        "Atoms",
        "E_coul",
        "E_vdwl",
        "TotEng",
    ]


def test_apply_thermo_axis_ranges_redraws_existing_figures():
    """Update both axes and request a redraw without rebuilding thermo plots."""

    figure, axis = plt.subplots()
    axis.plot([0, 10], [200, 400])
    redraws = []
    figure.canvas.draw_idle = lambda: redraws.append(True)

    apply_thermo_axis_ranges([figure.canvas], (2, 8), (250, 350))

    assert axis.get_xlim() == (2.0, 8.0)
    assert axis.get_ylim() == (250.0, 350.0)
    assert redraws == [True]
    plt.close(figure)


def test_apply_thermo_axis_ranges_restores_automatic_limits():
    """Restore data-driven limits when optional ranges are cleared."""

    figure, axis = plt.subplots()
    axis.plot([0, 10], [200, 400])
    axis.set_xlim(2, 8)
    axis.set_ylim(250, 350)

    apply_thermo_axis_ranges([figure.canvas], None, None)

    assert axis.get_xlim()[0] < 0
    assert axis.get_xlim()[1] > 10
    assert axis.get_ylim()[0] < 200
    assert axis.get_ylim()[1] > 400
    plt.close(figure)


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


def test_pathway_graph_builds_selected_branch_top_down():
    """Represent a selected connected pathway as state nodes and reaction arrows."""

    simulations = [
        LoadedSimulation(
            index=1,
            smiles={
                0: ["raw"],
                1: ["A", "D"],
                2: ["B", "C", "D"],
                3: ["B", "E"],
            },
            smiles_id={
                0: [["1", "2", "3"]],
                1: [["1", "2"], ["3"]],
                2: [["1"], ["2"], ["3"]],
                3: [["1"], ["2", "3"]],
            },
            chem_formulas={
                0: ["raw"],
                1: ["A", "D"],
                2: ["B", "C", "D"],
                3: ["B", "E"],
            },
        )
    ]
    pathway = connected_reaction_pathway_data(simulations, notation="formula")[0]

    assert pathway_graph_choices(pathway) == [
        ("A", "A [depth 1]: A -> B + C (n=1)"),
        ("B", "B [depth 2]: C + D -> E (n=1)"),
    ]

    graph = build_pathway_graph(pathway, root_label="A")

    assert [(node.label, node.depth) for node in graph.nodes] == [
        ("A", 0),
        ("B + C", 1),
        ("C + D", 1),
        ("E", 2),
    ]
    assert [(edge.source_key, edge.target_key, edge.arrow) for edge in graph.edges] == [
        ("0:A", "1:B + C", "->"),
        ("1:C + D", "2:E", "->"),
    ]

    child_graph = build_pathway_graph(pathway, root_label="B")

    assert [(node.label, node.depth) for node in child_graph.nodes] == [
        ("C + D", 0),
        ("E", 1),
    ]
    assert [(edge.source_key, edge.target_key, edge.arrow) for edge in child_graph.edges] == [
        ("0:C + D", "1:E", "->"),
    ]


def test_pathway_graph_image_extent_preserves_snapshot_aspect_ratio():
    """Fit pathway graph snapshots inside nodes without stretching them."""

    extent = pathway_graph_image_extent(0.0, 0.0, 6.2, 4.0, (700, 980, 4))
    draw_width = extent[1] - extent[0]
    draw_height = extent[3] - extent[2]

    assert round(draw_width / draw_height, 6) == round(980 / 700, 6)
    assert draw_width <= 6.2 * 0.94
    assert draw_height <= 4.0 * 0.59


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


def test_nearest_line_point_finds_labelled_data_in_display_coordinates():
    """Find a plotted point near the mouse while ignoring reference lines."""

    figure, axis = plt.subplots()
    line = axis.plot([10, 20], [1.5, 3.5], label="Simulation 1")[0]
    axis.axvline(20)
    figure.canvas.draw()
    x_pixel, y_pixel = line.get_transform().transform((20, 3.5))

    nearest = _nearest_line_point(figure, x_pixel + 2, y_pixel - 2)

    assert nearest is not None
    assert nearest[0] is line
    assert nearest[1:] == (1, 20.0, 3.5)
    assert _format_hover_value(1.23456789) == "1.234568"
    plt.close(figure)
