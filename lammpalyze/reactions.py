"""Reaction path clustering, counting, and CSV export utilities."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from lammpalyze.parsers import map_atoms_to_mols


@dataclass(frozen=True)
class ReactionPath:
    """One reaction string plus the number of times it was observed."""

    reaction: str
    count: int


@dataclass(frozen=True)
class ReactionOccurrence:
    """A specific reaction event, including atom ids for later visualization."""

    reaction: str
    timestep_reactants: int
    timestep_products: int
    reactants: list[str]
    products: list[str]
    reactant_atom_ids: list[str]
    product_atom_ids: list[str]
    simulation_index: int | None = None


@dataclass(frozen=True)
class ConnectedReactionStep:
    """One displayed edge in a connected reaction pathway."""

    label: str
    parents: tuple[str, ...]
    depth: int
    source: str
    arrow: str
    target: str
    count: int
    simulations: tuple[int, ...]


@dataclass(frozen=True)
class ConnectedReactionPathway:
    """A connected component of reaction states and transitions."""

    index: int
    root_states: tuple[str, ...]
    steps: tuple[ConnectedReactionStep, ...]


class UnionFindReax:
    """Disjoint-set structure used while linking molecules across timesteps."""

    def __init__(self) -> None:
        """Start with no known reactant/product nodes."""

        self.root: dict[tuple[str, int], tuple[str, int]] = {}

    def find(self, value: tuple[str, int]) -> tuple[str, int]:
        """Locate the representative node, creating it if needed."""

        if value not in self.root:
            self.root[value] = value
        if self.root[value] != value:
            self.root[value] = self.find(self.root[value])
        return self.root[value]

    def union(self, value1: tuple[str, int], value2: tuple[str, int]) -> None:
        """Treat two molecule indexes as part of the same reaction cluster."""

        root1 = self.find(value1)
        root2 = self.find(value2)
        if root1 != root2:
            self.root[root1] = root2


def reaction_clusters(
    mol_list_t1: list[list[int]],
    mol_list_t2: list[list[int]],
) -> list[dict[str, list[int]]]:
    """Connect reactant and product molecule indexes for adjacent timesteps."""

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
    """Count reaction signatures over a sequence of parsed bond frames.

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
            if Counter(reactants) != Counter(products):
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
    """List concrete events, optionally narrowed to one reaction string."""

    occurrences: list[ReactionOccurrence] = []
    for t1, t2, reaction in _iter_reactions(smiles, smiles_id):
        reactants = sorted(smiles[t1][index] for index in reaction["reactants"])
        products = sorted(smiles[t2][index] for index in reaction["products"])
        if Counter(reactants) == Counter(products):
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


def build_reaction_path_table(simulations) -> tuple[list[int], list[ReactionPath], dict[str, dict[int, int]]]:
    """Build total and per-simulation reaction counts from loaded simulations."""

    simulation_indices = []
    counts_by_reaction: dict[str, dict[int, int]] = {}
    all_paths: dict[str, int] = {}
    for simulation in simulations:
        if simulation.smiles is None or simulation.smiles_id is None:
            continue
        simulation_indices.append(simulation.index)
        for path in count_reaction_paths(simulation.smiles, simulation.smiles_id):
            counts_by_reaction.setdefault(path.reaction, {})[simulation.index] = path.count
            all_paths[path.reaction] = all_paths.get(path.reaction, 0) + path.count
    paths = [
        ReactionPath(reaction, count)
        for reaction, count in sorted(all_paths.items(), key=lambda item: item[1], reverse=True)
    ]
    return simulation_indices, paths, counts_by_reaction


def build_connected_reaction_pathways(
    simulations,
    notation: str = "formula",
) -> list[ConnectedReactionPathway]:
    """Build possible reaction hierarchies from initially present species."""

    if notation not in {"formula", "smiles"}:
        raise ValueError("notation must be 'formula' or 'smiles'.")

    initial_species: set[str] = set()
    edge_counts: Counter[tuple[tuple[str, ...], tuple[str, ...]]] = Counter()
    edge_simulations: dict[tuple[tuple[str, ...], tuple[str, ...]], set[int]] = defaultdict(set)
    edge_orders: dict[tuple[tuple[str, ...], tuple[str, ...]], int] = {}
    order = 0

    for simulation in simulations:
        if simulation.smiles is None or simulation.smiles_id is None:
            continue
        if notation == "formula":
            if simulation.chem_formulas is None:
                continue
            values_by_time = simulation.chem_formulas
        else:
            values_by_time = simulation.smiles

        timesteps = sorted(simulation.smiles)
        if not timesteps:
            continue
        initial_timestep = timesteps[1] if len(timesteps) > 1 else timesteps[0]
        initial_species.update(values_by_time[initial_timestep])
        timestep_positions = {timestep: index for index, timestep in enumerate(timesteps)}
        initial_position = timestep_positions[initial_timestep]

        for t1, t2, reaction in _iter_reactions(simulation.smiles, simulation.smiles_id):
            if timestep_positions[t1] < initial_position:
                continue
            source = tuple(sorted(values_by_time[t1][index] for index in reaction["reactants"]))
            target = tuple(sorted(values_by_time[t2][index] for index in reaction["products"]))
            if source == target:
                continue

            edge = (source, target)
            edge_counts[edge] += 1
            edge_simulations[edge].add(simulation.index)
            edge_orders[edge] = min(edge_orders.get(edge, order), order)
            order += 1

    if not edge_counts:
        return []

    groups = _reachable_pathway_groups(edge_counts, edge_simulations, edge_orders, initial_species)
    if not groups:
        return []
    steps = _pathway_steps_from_groups(groups)
    return [
        ConnectedReactionPathway(
            index=1,
            root_states=tuple(sorted(initial_species)),
            steps=tuple(steps),
        )
    ]


def _reachable_pathway_groups(
    edge_counts: Counter[tuple[tuple[str, ...], tuple[str, ...]]],
    edge_simulations: dict[tuple[tuple[str, ...], tuple[str, ...]], set[int]],
    edge_orders: dict[tuple[tuple[str, ...], tuple[str, ...]], int],
    initial_species: set[str],
) -> list[dict[str, object]]:
    """Orient and depth reaction groups by possible species descent."""

    species_depth = {species: 0 for species in initial_species}
    reachable_edges: dict[tuple[tuple[str, ...], tuple[str, ...]], int] = {}
    changed = True
    while changed:
        changed = False
        for source, target in sorted(edge_counts, key=lambda edge: edge_orders[edge]):
            source_depth = _source_ready_depth(source, species_depth)
            if source_depth is None:
                continue
            reaction_depth = source_depth + 1
            edge = (source, target)
            if edge not in reachable_edges or reaction_depth < reachable_edges[edge]:
                reachable_edges[edge] = reaction_depth
                changed = True
            for product in target:
                if product not in species_depth or reaction_depth < species_depth[product]:
                    species_depth[product] = reaction_depth
                    changed = True

    grouped: dict[tuple[tuple[str, ...], tuple[str, ...]], dict[str, object]] = {}
    for source, target in reachable_edges:
        key = _undirected_edge_key(source, target)
        group = grouped.setdefault(
            key,
            {
                "source": source,
                "target": target,
                "depth": reachable_edges[(source, target)],
                "order": edge_orders[(source, target)],
                "arrow": "<->" if (target, source) in edge_counts else "->",
            },
        )
        current_key = (
            reachable_edges[(source, target)],
            edge_orders[(source, target)],
        )
        best_key = (
            group["depth"],
            group["order"],
        )
        if current_key < best_key:
            group["source"] = source
            group["target"] = target
            group["depth"] = reachable_edges[(source, target)]
            group["order"] = edge_orders[(source, target)]

    groups = []
    for key, group in grouped.items():
        source, target = key
        forward = (source, target)
        reverse = (target, source)
        simulations = set()
        count = 0
        for edge in (forward, reverse):
            if edge in edge_counts:
                count += edge_counts[edge]
                simulations.update(edge_simulations[edge])
        group["count"] = count
        group["simulations"] = tuple(sorted(simulations))
        group["products"] = set(group["target"])
        groups.append(group)

    groups.sort(key=lambda group: (group["depth"], group["order"], _format_state(group["source"])))
    return groups


def _pathway_steps_from_groups(groups: list[dict[str, object]]) -> list[ConnectedReactionStep]:
    """Convert reachable reaction groups to labelled display steps."""

    labels = {id(group): _pathway_step_label(index) for index, group in enumerate(groups)}
    steps = []
    for group in groups:
        parents = _possible_parent_labels(group, groups, labels)
        steps.append(
            ConnectedReactionStep(
                label=labels[id(group)],
                parents=parents,
                depth=group["depth"],
                source=_format_state(group["source"]),
                arrow=group["arrow"],
                target=_format_state(group["target"]),
                count=group["count"],
                simulations=group["simulations"],
            )
        )
    return steps


def _source_ready_depth(source: tuple[str, ...], species_depth: dict[str, int]) -> int | None:
    """Return the maximum known species depth for a reactant side."""

    depths = [species_depth[value] for value in source if value in species_depth]
    if len(depths) != len(source):
        return None
    return max(depths, default=0)


def _possible_parent_labels(
    group: dict[str, object],
    groups: list[dict[str, object]],
    labels: dict[int, str],
) -> tuple[str, ...]:
    """Return parent labels whose products feed this group's reactants."""

    depth = group["depth"]
    if depth <= 1:
        return ()
    reactants = set(group["source"])
    parent_labels = [
        labels[id(parent)]
        for parent in groups
        if parent["depth"] == depth - 1 and reactants & parent["products"]
    ]
    return tuple(parent_labels)


def format_connected_reaction_pathways(pathways: list[ConnectedReactionPathway]) -> str:
    """Format connected pathway components as primary-path trees."""

    if not pathways:
        return "No connected reaction pathways available."

    lines = []
    for pathway in pathways:
        roots = ", ".join(pathway.root_states) if pathway.root_states else "(unknown)"
        lines.append(f"Initial species: {roots}")
        lines.extend(_format_pathway_tree(pathway.steps))
    return "\n".join(lines)


def _format_pathway_tree(steps: tuple[ConnectedReactionStep, ...]) -> list[str]:
    """Format depth-1 pathways with indented descendant pathways."""

    children: dict[str, list[ConnectedReactionStep]] = defaultdict(list)
    roots = []
    for step in steps:
        if not step.parents:
            roots.append(step)
        for parent in step.parents:
            children[parent].append(step)

    lines = []
    for root in roots:
        _append_pathway_branch(root, children, lines, indent=0, active=())
    return lines


def _append_pathway_branch(
    step: ConnectedReactionStep,
    children: dict[str, list[ConnectedReactionStep]],
    lines: list[str],
    *,
    indent: int,
    active: tuple[str, ...],
) -> None:
    """Append one pathway branch, guarding against accidental cycles."""

    simulations = ", ".join(str(index) for index in step.simulations)
    lines.append(
        f"{'  ' * indent}Pathway {step.label} [depth {step.depth}]: "
        f"{step.source} {step.arrow} {step.target} "
        f"(n={step.count}; simulations {simulations})"
    )
    if step.label in active:
        return
    for child in children.get(step.label, []):
        _append_pathway_branch(child, children, lines, indent=indent + 1, active=(*active, step.label))


def write_reaction_paths_csv(
    paths: list[ReactionPath],
    output_file: str | Path = "paths.csv",
    *,
    simulation_indices: list[int] | None = None,
    counts_by_reaction: Mapping[str, Mapping[int, int]] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> Path:
    """Write a reaction table CSV, including metadata when the CLI provides it."""

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
    """Represent sorted reactants and products as the historical path string."""

    return f"{reactants} -> {products}"


def _metadata_value(value: object) -> str:
    """Convert metadata values to compact strings for the header block."""

    if isinstance(value, (list, tuple, set)):
        return ";".join(str(item) for item in value)
    return str(value)


def _format_state(values: tuple[str, ...]) -> str:
    """Render one side of a reaction as a compact state label."""

    if not values:
        return "(none)"
    counts = Counter(values)
    return " + ".join(_format_state_part(value, count) for value, count in sorted(counts.items()))


def _format_state_part(value: str, count: int) -> str:
    """Render one molecule label, compacting repeated molecules."""

    if count == 1:
        return value
    return f"{count}{value}"


def _pathway_step_label(index: int) -> str:
    """Return spreadsheet-style step labels: A, B, ..., Z, AA."""

    label = ""
    value = index
    while True:
        value, remainder = divmod(value, 26)
        label = chr(ord("A") + remainder) + label
        if value == 0:
            return label
        value -= 1


def _undirected_edge_key(
    source: tuple[str, ...],
    target: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return a stable key for two states regardless of direction."""

    if source <= target:
        return source, target
    return target, source


def _iter_reactions(smiles: dict[int, list[str]], smiles_id: dict[int, list[list[str]]]):
    """Walk adjacent frames and yield the raw cluster maps used by counters."""

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
    """Sort atom ids numerically when possible, with a string fallback."""

    try:
        return int(atom_id), atom_id
    except ValueError:
        return 0, atom_id
