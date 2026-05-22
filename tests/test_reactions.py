from lammpalyze.reactions import count_reaction_paths


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
