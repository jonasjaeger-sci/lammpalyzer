from lammpalyze.reactions import count_reaction_paths, find_reaction_occurrences


def test_count_reaction_paths_counts_split_reaction():
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


def test_find_reaction_occurrences_returns_first_atom_metadata():
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
