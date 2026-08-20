"""Tests for SMILES helpers."""

import pytest

from lammpalyze.smiles import (
    canonicalize_smiles,
    extend_molecule_descriptor_list,
    format_molecule_descriptor_list,
    reaction_smiles_groups,
    reaction_smiles_path,
    smiles_for_formula,
)


def test_smiles_for_formula_returns_observed_values():
    """Return unique SMILES values observed for a formula."""

    formulas = {0: ["H2", "O"], 1: ["H2"]}
    smiles = {0: ["[H][H]", "[O]"], 1: ["[H][H]"]}

    assert smiles_for_formula(formulas, smiles, "H2") == ["[H][H]"]


def test_extend_molecule_descriptor_list_preserves_order_and_omits_duplicates():
    """Grow a descriptor list across gallery selections without repeated values."""

    assert extend_molecule_descriptor_list(
        ["[Li+]"],
        [" CCO ", "[Li+]", "", "[O-]C=O"],
    ) == ["[Li+]", "CCO", "[O-]C=O"]


def test_format_molecule_descriptor_list_produces_geometry_ready_strings():
    """Produce a quoted list that can be pasted into the Geometry filter."""

    assert format_molecule_descriptor_list(["C3H4LiO3", "[Li+]"]) == (
        '["C3H4LiO3", "[Li+]"]'
    )


def test_reaction_smiles_groups_parses_formatted_path():
    """Split reaction-path strings into reactant and product SMILES lists."""

    assert reaction_smiles_groups("['CCO', '[Li+]'] -> ['CC[O-]', '[Li+]']") == (
        ["CCO", "[Li+]"],
        ["CC[O-]", "[Li+]"],
    )


def test_reaction_smiles_groups_rejects_invalid_paths():
    """Reject strings that are not formatted reaction paths."""

    with pytest.raises(ValueError, match="must contain"):
        reaction_smiles_groups("CCO")


def test_reaction_smiles_path_sorts_groups():
    """Format reaction groups in the same order used by path counting."""

    assert reaction_smiles_path(["[Li+]", "CCO"], ["O", "C"]) == "['CCO', '[Li+]'] -> ['C', 'O']"


def test_canonicalize_smiles_rejects_invalid_input():
    """Reject invalid SMILES strings during canonicalization."""

    pytest.importorskip("rdkit")

    with pytest.raises(ValueError):
        canonicalize_smiles("not a smiles")
