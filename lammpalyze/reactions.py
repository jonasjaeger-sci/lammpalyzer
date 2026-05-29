"""Reaction path extraction, counting, and export helpers."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from lammpalyze.parsers import map_atoms_to_mols


@dataclass(frozen=True)
class ReactionPath:
    """Counted reaction path, stored in the SMILES notation used by the parser."""

    reaction: str
    count: int


@dataclass(frozen=True)
class ReactionOccurrence:
    """Concrete reaction event with enough atom metadata for visualization."""

    reaction: str
    timestep_reactants: int
    timestep_products: int
    reactants: list[str]
    products: list[str]
    reactant_atom_ids: list[str]
    product_atom_ids: list[str]
    simulation_index: int | None = None


class UnionFindReax:
    """Small disjoint-set helper for matching reactants to products."""

    def __init__(self) -> None:
        """Initialize an empty disjoint-set forest."""

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

    Atom ids are mapped to molecule indexes at ``t1`` and ``t2``, connected
    reaction clusters are found, unchanged molecule sets are ignored, and
    identical paths are counted.
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


def write_reaction_paths(paths: list[ReactionPath], output_file: str | Path = "paths.csv") -> Path:
    """Compatibility wrapper for writing the total reaction-path CSV."""

    return write_reaction_paths_csv(paths, output_file)


def write_reaction_paths_csv(
    paths: list[ReactionPath],
    output_file: str | Path = "paths.csv",
    *,
    simulation_indices: list[int] | None = None,
    counts_by_reaction: Mapping[str, Mapping[int, int]] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> Path:
    """Write reaction paths as CSV with optional metadata and per-run counts."""

    output_path = Path(output_file)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        if metadata:
            writer.writerow(["Metadata", "Value"])
            for key, value in metadata.items():
                writer.writerow([key, _metadata_value(value)])
            writer.writerow([])

        simulation_indices = simulation_indices or []
        counts_by_reaction = counts_by_reaction or {}
        simulation_columns = [f"Simulation {index}" for index in simulation_indices]
        writer.writerow(["Reaction", *simulation_columns, "Sum"])
        for path in paths:
            per_simulation = counts_by_reaction.get(path.reaction, {})
            writer.writerow(
                [
                    path.reaction,
                    *(per_simulation.get(index, 0) for index in simulation_indices),
                    path.count,
                ]
            )
    return output_path


def _format_reaction(reactants: list[str], products: list[str]) -> str:
    """Format sorted reactant and product SMILES lists as a path string."""

    return f"{reactants} -> {products}"


def _metadata_value(value: object) -> str:
    """Render metadata values without leaking Python container syntax into CSV."""

    if isinstance(value, (list, tuple, set)):
        return ";".join(str(item) for item in value)
    return str(value)


def _iter_reactions(smiles: dict[int, list[str]], smiles_id: dict[int, list[list[str]]]):
    """Yield reaction clusters for each adjacent timestep pair."""

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
    """Return a stable numeric-first sort key for atom identifiers."""

    try:
        return int(atom_id), atom_id
    except ValueError:
        return 0, atom_id
