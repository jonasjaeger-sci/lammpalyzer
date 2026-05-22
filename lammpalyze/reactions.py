"""Reaction-path extraction and occurrence counting."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from lammpalyze.parsers import map_atoms_to_mols


@dataclass(frozen=True)
class ReactionPath:
    """A counted reaction path in SMILES notation."""

    reaction: str
    count: int


@dataclass(frozen=True)
class ReactionOccurrence:
    """One concrete occurrence of a reaction path."""

    reaction: str
    timestep_reactants: int
    timestep_products: int
    reactants: list[str]
    products: list[str]
    reactant_atom_ids: list[str]
    product_atom_ids: list[str]
    simulation_index: int | None = None


class UnionFindReax:
    """Union-find helper for grouping reactants and products across timesteps."""

    def __init__(self) -> None:
        self.root: dict[tuple[str, int], tuple[str, int]] = {}

    def find(self, value: tuple[str, int]) -> tuple[str, int]:
        """Return the representative root for ``value``."""

        if value not in self.root:
            self.root[value] = value
        if self.root[value] != value:
            self.root[value] = self.find(self.root[value])
        return self.root[value]

    def union(self, value1: tuple[str, int], value2: tuple[str, int]) -> None:
        """Join two values into the same reaction cluster."""

        root1 = self.find(value1)
        root2 = self.find(value2)
        if root1 != root2:
            self.root[root1] = root2


def reaction_clusters(
    mol_list_t1: list[list[int]],
    mol_list_t2: list[list[int]],
) -> list[dict[str, list[int]]]:
    """Build connected reaction clusters between two consecutive timesteps."""

    union_find = UnionFindReax()
    for reactant_index, products in enumerate(mol_list_t1):
        for product_index in products:
            union_find.union(("reactant", reactant_index), ("product", product_index))

    reactions: dict[tuple[str, int], dict[str, list[int]]] = defaultdict(
        lambda: {"reactants": [], "products": []}
    )

    for reactant_index in range(len(mol_list_t1)):
        root = union_find.find(("reactant", reactant_index))
        reactions[root]["reactants"].append(reactant_index)

    for product_index in range(len(mol_list_t2)):
        root = union_find.find(("product", product_index))
        reactions[root]["products"].append(product_index)

    return list(reactions.values())


def count_reaction_paths(
    smiles: dict[int, list[str]],
    smiles_id: dict[int, list[list[str]]],
) -> list[ReactionPath]:
    """Count all reaction paths between consecutive timesteps.

    This refactors the historical ``execute.py`` logic: atom ids are mapped to
    molecule indexes at ``t1`` and ``t2``, connected reaction clusters are found,
    unchanged molecule sets are ignored, and identical paths are counted.
    """

    timesteps = sorted(smiles.keys())
    reaction_paths: Counter[str] = Counter()

    for t1, t2 in zip(timesteps, timesteps[1:], strict=False):
        atom_mapping_t1 = map_atoms_to_mols(smiles[t1], smiles_id[t1])
        atom_mapping_t2 = map_atoms_to_mols(smiles[t2], smiles_id[t2])

        pointer_t1_t2: list[list[int]] = []
        pointer_t2_t1: list[list[int]] = []

        for molecule in smiles_id[t1]:
            products = {atom_mapping_t2[atom_id][1] for atom_id in molecule if atom_id in atom_mapping_t2}
            pointer_t1_t2.append(sorted(products))

        for molecule in smiles_id[t2]:
            reactants = {atom_mapping_t1[atom_id][1] for atom_id in molecule if atom_id in atom_mapping_t1}
            pointer_t2_t1.append(sorted(reactants))

        for reaction in reaction_clusters(pointer_t1_t2, pointer_t2_t1):
            reactants = sorted(smiles[t1][index] for index in reaction["reactants"])
            products = sorted(smiles[t2][index] for index in reaction["products"])
            if set(reactants) != set(products):
                reaction_paths[_format_reaction(reactants, products)] += 1

    return [
        ReactionPath(reaction, count)
        for reaction, count in sorted(reaction_paths.items(), key=lambda item: item[1], reverse=True)
    ]


def find_reaction_occurrences(
    smiles: dict[int, list[str]],
    smiles_id: dict[int, list[list[str]]],
    reaction_filter: str | None = None,
    *,
    first_only: bool = False,
    simulation_index: int | None = None,
) -> list[ReactionOccurrence]:
    """Return concrete reaction occurrences with timestep and atom-id metadata."""

    occurrences: list[ReactionOccurrence] = []
    for t1, t2, reaction in _iter_reactions(smiles, smiles_id):
        reactants = sorted(smiles[t1][index] for index in reaction["reactants"])
        products = sorted(smiles[t2][index] for index in reaction["products"])
        if set(reactants) == set(products):
            continue

        reaction_path = _format_reaction(reactants, products)
        if reaction_filter is not None and reaction_path != reaction_filter:
            continue

        reactant_atom_ids = sorted(
            {atom_id for index in reaction["reactants"] for atom_id in smiles_id[t1][index]},
            key=_atom_sort_key,
        )
        product_atom_ids = sorted(
            {atom_id for index in reaction["products"] for atom_id in smiles_id[t2][index]},
            key=_atom_sort_key,
        )
        occurrences.append(
            ReactionOccurrence(
                reaction=reaction_path,
                timestep_reactants=t1,
                timestep_products=t2,
                reactants=reactants,
                products=products,
                reactant_atom_ids=reactant_atom_ids,
                product_atom_ids=product_atom_ids,
                simulation_index=simulation_index,
            )
        )
        if first_only:
            return occurrences

    return occurrences


def write_reaction_paths(paths: list[ReactionPath], output_file: str | Path = "paths.out") -> Path:
    """Write counted reaction paths to a tab-separated table."""

    output_path = Path(output_file)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["Reaction", "Count"])
        for path in paths:
            writer.writerow([path.reaction, path.count])
    return output_path


def _format_reaction(reactants: list[str], products: list[str]) -> str:
    return f"{reactants} -> {products}"


def _iter_reactions(smiles: dict[int, list[str]], smiles_id: dict[int, list[list[str]]]):
    timesteps = sorted(smiles.keys())
    for t1, t2 in zip(timesteps, timesteps[1:], strict=False):
        atom_mapping_t1 = map_atoms_to_mols(smiles[t1], smiles_id[t1])
        atom_mapping_t2 = map_atoms_to_mols(smiles[t2], smiles_id[t2])

        pointer_t1_t2: list[list[int]] = []
        pointer_t2_t1: list[list[int]] = []

        for molecule in smiles_id[t1]:
            products = {atom_mapping_t2[atom_id][1] for atom_id in molecule if atom_id in atom_mapping_t2}
            pointer_t1_t2.append(sorted(products))

        for molecule in smiles_id[t2]:
            reactants = {atom_mapping_t1[atom_id][1] for atom_id in molecule if atom_id in atom_mapping_t1}
            pointer_t2_t1.append(sorted(reactants))

        for reaction in reaction_clusters(pointer_t1_t2, pointer_t2_t1):
            yield t1, t2, reaction


def _atom_sort_key(atom_id: str) -> tuple[int, str]:
    try:
        return int(atom_id), atom_id
    except ValueError:
        return 0, atom_id
