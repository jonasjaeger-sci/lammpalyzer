"""Tests for reaction path counting and occurrence lookup."""

from pathlib import Path

from lammpalyze.analysis import LammpalyzeProject, LoadedSimulation, load_project
from lammpalyze.config import parse_input_file
from lammpalyze.reactions import (
    build_connected_reaction_pathways,
    count_reaction_paths,
    find_connected_reaction_occurrence,
    find_reaction_occurrences,
    format_connected_reaction_pathways,
)


def test_count_reaction_paths_counts_split_reaction():
    """Count split and recombination reactions across adjacent timesteps."""

    smiles = {
        0: ["AB"],
        1: ["A", "B"],
        2: ["A", "B"],
        3: ["AB"],
    }
    smiles_id = {
        0: [["1", "2"]],
        1: [["1"], ["2"]],
        2: [["1"], ["2"]],
        3: [["1", "2"]],
    }

    paths = count_reaction_paths(smiles, smiles_id)

    assert [(path.reaction, path.count) for path in paths] == [
        ("['AB'] -> ['A', 'B']", 1),
        ("['A', 'B'] -> ['AB']", 1),
    ]


def test_count_reaction_paths_skips_events_touching_excluded_components():
    """Do not turn excluded suspicious structures into disappearance events."""

    smiles = {0: ["AB"], 1: ["A", "B"]}
    smiles_id = {0: [["1", "2"]], 1: [["1"], ["2"]]}

    paths = count_reaction_paths(smiles, smiles_id, excluded_components={1: {0}})

    assert not paths


def test_skip_quality_mode_bridges_suspicious_intermediate():
    """Collapse A to suspicious B to C into one clean A to C event."""

    smiles = {0: ["A"], 1: ["B"], 2: ["C"]}
    smiles_id = {0: [["1"]], 1: [["1"]], 2: [["1"]]}

    paths = count_reaction_paths(
        smiles,
        smiles_id,
        excluded_components={1: {0}},
        quality_mode="skip",
    )
    occurrences = find_reaction_occurrences(
        smiles,
        smiles_id,
        excluded_components={1: {0}},
        quality_mode="skip",
    )

    assert [(path.reaction, path.count) for path in paths] == [("['A'] -> ['C']", 1)]
    assert len(occurrences) == 1
    assert occurrences[0].timestep_reactants == 0
    assert occurrences[0].timestep_products == 2


def test_skip_quality_mode_bridges_cluster_without_hiding_unrelated_reaction():
    """Bridge a partially suspicious split while retaining a separate clean event."""

    smiles = {
        0: ["A", "X"],
        1: ["B", "D", "Y"],
        2: ["C", "D", "Y"],
    }
    smiles_id = {
        0: [["1", "2"], ["3"]],
        1: [["1"], ["2"], ["3"]],
        2: [["1"], ["2"], ["3"]],
    }

    paths = count_reaction_paths(
        smiles,
        smiles_id,
        excluded_components={1: {0}},
        quality_mode="skip",
    )

    assert {(path.reaction, path.count) for path in paths} == {
        ("['A'] -> ['C', 'D']", 1),
        ("['X'] -> ['Y']", 1),
    }


def test_project_tables_use_skip_quality_mode_for_bridged_pathways():
    """Propagate skip-mode bridges into project and connected-pathway tables."""

    simulation = LoadedSimulation(
        index=1,
        smiles={0: ["raw"], 1: ["A"], 2: ["B"], 3: ["C"]},
        smiles_id={0: [["1"]], 1: [["1"]], 2: [["1"]], 3: [["1"]]},
        chem_formulas={0: ["raw"], 1: ["A"], 2: ["B"], 3: ["C"]},
        excluded_components={2: {0}},
        structure_quality_mode="skip",
    )
    project = LammpalyzeProject(config=None, simulations=[simulation])

    assert [(path.reaction, path.count) for path in project.reaction_paths()] == [
        ("['raw'] -> ['A']", 1),
        ("['A'] -> ['C']", 1),
    ]
    pathways = project.connected_reaction_pathways(notation="smiles")
    assert [(step.source, step.target) for step in pathways[0].steps] == [("A", "C")]


def test_project_first_reaction_occurrences_report_simulation_and_timesteps():
    """Collect first occurrence metadata for every observed reaction path."""

    project = LammpalyzeProject(
        config=None,
        simulations=[
            LoadedSimulation(
                index=4,
                smiles={0: ["AB"], 10: ["A", "B"], 20: ["AB"]},
                smiles_id={0: [["1", "2"]], 10: [["1"], ["2"]], 20: [["1", "2"]]},
            )
        ],
    )

    occurrences = project.first_reaction_occurrences()

    simulation, occurrence = occurrences["['AB'] -> ['A', 'B']"]
    assert simulation.index == 4
    assert occurrence.simulation_index == 4
    assert occurrence.timestep_reactants == 0
    assert occurrence.timestep_products == 10


def test_count_reaction_paths_respects_duplicate_species_stoichiometry():
    """Treat A + A -> A as a reaction instead of collapsing duplicates by set."""

    smiles = {0: ["A", "A"], 1: ["A"]}
    smiles_id = {0: [["1"], ["2"]], 1: [["1", "2"]]}

    paths = count_reaction_paths(smiles, smiles_id)

    assert [(path.reaction, path.count) for path in paths] == [("['A', 'A'] -> ['A']", 1)]


def test_find_reaction_occurrences_returns_first_atom_metadata():
    """Return atom ids and simulation metadata for the first matching reaction."""

    smiles = {0: ["AB"], 1: ["A", "B"]}
    smiles_id = {0: [["1", "2"]], 1: [["1"], ["2"]]}

    occurrences = find_reaction_occurrences(
        smiles,
        smiles_id,
        reaction_filter="['AB'] -> ['A', 'B']",
        first_only=True,
        simulation_index=7,
    )

    assert len(occurrences) == 1
    assert occurrences[0].simulation_index == 7
    assert occurrences[0].timestep_reactants == 0
    assert occurrences[0].timestep_products == 1
    assert occurrences[0].reactant_atom_ids == ["1", "2"]
    assert occurrences[0].product_atom_ids == ["1", "2"]


def test_build_connected_reaction_pathways_groups_branching_formula_paths():
    """Connect reaction states across simulations and assign product depths."""

    simulations = [
        LoadedSimulation(
            index=1,
            smiles={0: ["raw"], 1: ["A"], 2: ["B"], 3: ["C1"]},
            smiles_id={0: [["1"]], 1: [["1"]], 2: [["1"]], 3: [["1"]]},
            chem_formulas={0: ["raw"], 1: ["LiC3H4O3"], 2: ["B"], 3: ["C1"]},
        ),
        LoadedSimulation(
            index=2,
            smiles={0: ["raw"], 1: ["A"], 2: ["B"], 3: ["C2"]},
            smiles_id={0: [["1"]], 1: [["1"]], 2: [["1"]], 3: [["1"]]},
            chem_formulas={0: ["raw"], 1: ["LiC3H4O3"], 2: ["B"], 3: ["C2"]},
        ),
    ]

    pathways = build_connected_reaction_pathways(simulations, notation="formula")

    assert len(pathways) == 1
    assert pathways[0].root_states == ("LiC3H4O3",)
    observed_steps = [
        (step.depth, step.source, step.arrow, step.target, step.simulations, step.counts_by_simulation)
        for step in pathways[0].steps
    ]
    assert observed_steps == [
        (1, "LiC3H4O3", "->", "B", (1, 2), ((1, 1), (2, 1))),
        (2, "B", "->", "C1", (1,), ((1, 1),)),
        (2, "B", "->", "C2", (2,), ((2, 1),)),
    ]


def test_build_connected_reaction_pathways_merges_reversible_smiles_edges():
    """Show reciprocal reactions as one bidirectional pathway step."""

    simulations = [
        LoadedSimulation(
            index=1,
            smiles={0: ["raw"], 1: ["AB"], 2: ["A", "B"], 3: ["AB"]},
            smiles_id={0: [["1", "2"]], 1: [["1", "2"]], 2: [["1"], ["2"]], 3: [["1", "2"]]},
            chem_formulas={0: ["raw"], 1: ["AB"], 2: ["A", "B"], 3: ["AB"]},
        )
    ]

    pathways = build_connected_reaction_pathways(simulations, notation="smiles")

    assert len(pathways) == 1
    observed_steps = [
        (step.label, step.depth, step.source, step.arrow, step.target, step.count)
        for step in pathways[0].steps
    ]
    assert observed_steps == [
        ("A", 1, "AB", "<->", "A + B", 2)
    ]
    assert pathways[0].steps[0].counts_by_simulation == ((1, 2),)
    assert "Pathway A [depth 1]: AB <-> A + B" in format_connected_reaction_pathways(pathways)


def test_connected_pathway_threshold_filters_low_count_steps():
    """Only retain connected pathway steps that meet the minimum count."""

    simulations = [
        LoadedSimulation(
            index=1,
            smiles={0: ["raw"], 1: ["A"], 2: ["B"], 3: ["C1"]},
            smiles_id={0: [["1"]], 1: [["1"]], 2: [["1"]], 3: [["1"]]},
            chem_formulas={0: ["raw"], 1: ["A"], 2: ["B"], 3: ["C1"]},
        ),
        LoadedSimulation(
            index=2,
            smiles={0: ["raw"], 1: ["A"], 2: ["B"], 3: ["C2"]},
            smiles_id={0: [["1"]], 1: [["1"]], 2: [["1"]], 3: [["1"]]},
            chem_formulas={0: ["raw"], 1: ["A"], 2: ["B"], 3: ["C2"]},
        ),
    ]

    pathways = build_connected_reaction_pathways(simulations, notation="formula", min_count=2)

    assert len(pathways) == 1
    assert [
        (step.source, step.arrow, step.target, step.count, step.counts_by_simulation)
        for step in pathways[0].steps
    ] == [
        ("A", "->", "B", 2, ((1, 1), (2, 1))),
    ]


def test_connected_pathway_threshold_uses_reversible_display_count():
    """Apply the threshold to the displayed bidirectional count."""

    simulation = LoadedSimulation(
        index=1,
        smiles={0: ["raw"], 1: ["AB"], 2: ["A", "B"], 3: ["AB"]},
        smiles_id={0: [["1", "2"]], 1: [["1", "2"]], 2: [["1"], ["2"]], 3: [["1", "2"]]},
        chem_formulas={0: ["raw"], 1: ["AB"], 2: ["A", "B"], 3: ["AB"]},
    )

    pathways = build_connected_reaction_pathways([simulation], notation="smiles", min_count=2)

    assert len(pathways) == 1
    assert [(step.source, step.arrow, step.target, step.count) for step in pathways[0].steps] == [
        ("AB", "<->", "A + B", 2)
    ]


def test_connected_formula_pathways_skip_formula_equivalent_smiles_changes():
    """Avoid displaying projected formula non-reactions such as Li3 -> Li3."""

    simulations = [
        LoadedSimulation(
            index=1,
            smiles={0: ["raw"], 1: ["[Li][Li][Li]"], 2: ["[Li]1[Li][Li]1"]},
            smiles_id={0: [["1", "2", "3"]], 1: [["1", "2", "3"]], 2: [["1", "2", "3"]]},
            chem_formulas={0: ["raw"], 1: ["Li3"], 2: ["Li3"]},
        )
    ]

    formula_pathways = build_connected_reaction_pathways(simulations, notation="formula")
    smiles_pathways = build_connected_reaction_pathways(simulations, notation="smiles")

    assert not formula_pathways
    assert len(smiles_pathways) == 1
    assert smiles_pathways[0].steps[0].source == "[Li][Li][Li]"
    assert smiles_pathways[0].steps[0].target == "[Li]1[Li][Li]1"


def test_connected_pathways_track_product_lineage_with_new_reactants():
    """Connect later reactions to product ancestry even when whole sides differ."""

    simulation = LoadedSimulation(
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

    pathways = build_connected_reaction_pathways([simulation], notation="formula")

    assert len(pathways) == 1
    assert pathways[0].root_states == ("A", "D")
    observed_steps = [
        (step.label, step.parents, step.depth, step.source, step.arrow, step.target)
        for step in pathways[0].steps
    ]
    assert observed_steps == [
        ("A", (), 1, "A", "->", "B + C"),
        ("B", ("A",), 2, "C + D", "->", "E"),
    ]


def test_connected_pathway_parents_match_displayed_reactants_in_example():
    """Do not link a parent when its products are absent from child reactants."""

    repo_root = Path(__file__).resolve().parents[1]
    config = parse_input_file(repo_root / "examples" / "example_NVT_vs_NPT" / "lmplyz.inp")
    project = load_project(config)

    pathways = build_connected_reaction_pathways(project.simulations, notation="formula")
    steps_by_reaction = {
        (step.source, step.arrow, step.target): step
        for pathway in pathways
        for step in pathway.steps
    }

    step = steps_by_reaction[("CLiO3 + Li", "<->", "CLi2O3")]

    assert "A" not in step.parents
    assert step.parents == ("C",)


def test_find_connected_reaction_occurrence_matches_formula_step():
    """Map a formula pathway step back to the concrete SMILES event."""

    simulation = LoadedSimulation(
        index=3,
        smiles={0: ["raw"], 1: ["[Li]", "O=C1"], 2: ["[Li]O=C1"]},
        smiles_id={0: [["1", "2"]], 1: [["1"], ["2"]], 2: [["1", "2"]]},
        chem_formulas={0: ["raw"], 1: ["Li", "C3H4O3"], 2: ["LiC3H4O3"]},
    )
    pathway = build_connected_reaction_pathways([simulation], notation="formula")[0]

    connected_occurrence = find_connected_reaction_occurrence(
        simulation,
        pathway.steps[0],
        notation="formula",
    )

    assert connected_occurrence is not None
    assert connected_occurrence.matched_direction == "forward"
    assert connected_occurrence.occurrence.reaction == "['O=C1', '[Li]'] -> ['[Li]O=C1']"
    assert connected_occurrence.occurrence.reactant_atom_ids == ["1", "2"]
