import pytest

from lammpalyze.smiles import canonicalize_smiles, smiles_for_formula


def test_smiles_for_formula_returns_observed_values():
    formulas = {0: ["H2", "O"], 1: ["H2"]}
    smiles = {0: ["[H][H]", "[O]"], 1: ["[H][H]"]}

    assert smiles_for_formula(formulas, smiles, "H2") == ["[H][H]"]


def test_canonicalize_smiles_rejects_invalid_input():
    pytest.importorskip("rdkit")

    with pytest.raises(ValueError):
        canonicalize_smiles("not a smiles")
