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
    counts_by_simulation: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class ConnectedReactionPathway:
    """A connected component of reaction states and transitions."""

    index: int
    root_states: tuple[str, ...]
    steps: tuple[ConnectedReactionStep, ...]


@dataclass(frozen=True)
class ConnectedReactionOccurrence:
    """A concrete occurrence matching a displayed connected pathway step."""

    step: ConnectedReactionStep
    occurrence: ReactionOccurrence
    matched_direction: str = "forward"


@dataclass
class PathwayEdgeData:
    """Aggregated reaction edges for connected pathway construction."""

    initial_species: set[str]
    edge_counts: Counter[tuple[tuple[str, ...], tuple[str, ...]]]
    edge_simulations: dict[tuple[tuple[str, ...], tuple[str, ...]], set[int]]
    edge_simulation_counts: dict[tuple[tuple[str, ...], tuple[str, ...]], Counter[int]]
    edge_orders: dict[tuple[tuple[str, ...], tuple[str, ...]], int]


@dataclass
class _PendingSkipLineage:
    """Atom lineage waiting to emerge from suspicious component observations."""

    baseline_timestep: int | None
    atom_ids: set[str]


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
    excluded_components: Mapping[int, set[int]] | None = None,
    quality_mode: str = "exclude",
) -> list[ReactionPath]:
    """Count reaction signatures over a sequence of parsed bond frames.

    Atom ids are mapped to molecule indexes at ``t1`` and ``t2``, connected
    reaction clusters are found, unchanged molecule sets are ignored, and
    identical paths are counted.
    """

    reaction_paths: Counter[str] = Counter()

    for t1, t2, reaction in _iter_reactions(
        smiles,
        smiles_id,
        excluded_components,
        quality_mode,
    ):
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
    excluded_components: Mapping[int, set[int]] | None = None,
    quality_mode: str = "exclude",
) -> list[ReactionOccurrence]:
    """List concrete events, optionally narrowed to one reaction string."""

    occurrences: list[ReactionOccurrence] = []
    for t1, t2, reaction in _iter_reactions(
        smiles,
        smiles_id,
        excluded_components,
        quality_mode,
    ):
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


def find_connected_reaction_occurrence(
    simulation,
    step: ConnectedReactionStep,
    notation: str = "formula",
) -> ConnectedReactionOccurrence | None:
    """Return the first concrete event in ``simulation`` matching ``step``.

    Connected pathway rows may be displayed as formulas, while concrete
    reaction occurrences are stored as SMILES.  This matcher compares the
    projected source and target states in the selected notation and returns a
    normal :class:`ReactionOccurrence` carrying the atom ids needed by
    visualization code.
    """

    _validate_connected_pathway_options(notation, min_count=1)
    values_by_time = _pathway_values_by_time(simulation, notation)
    if values_by_time is None:
        return None
    if simulation.smiles is None or simulation.smiles_id is None:
        return None

    for t1, t2, reaction in _iter_reactions(
        simulation.smiles,
        simulation.smiles_id,
        getattr(simulation, "excluded_components", None),
        getattr(simulation, "structure_quality_mode", "exclude"),
    ):
        source_values = sorted(values_by_time[t1][index] for index in reaction["reactants"])
        target_values = sorted(values_by_time[t2][index] for index in reaction["products"])
        if Counter(source_values) == Counter(target_values):
            continue

        source = _format_state(tuple(source_values))
        target = _format_state(tuple(target_values))
        direction = ""
        if source == step.source and target == step.target:
            direction = "forward"
        elif step.arrow == "<->" and source == step.target and target == step.source:
            direction = "reverse"
        if not direction:
            continue

        reactants = sorted(simulation.smiles[t1][index] for index in reaction["reactants"])
        products = sorted(simulation.smiles[t2][index] for index in reaction["products"])
        reactant_atom_ids = sorted(
            {atom_id for index in reaction["reactants"] for atom_id in simulation.smiles_id[t1][index]},
            key=_atom_sort_key,
        )
        product_atom_ids = sorted(
            {atom_id for index in reaction["products"] for atom_id in simulation.smiles_id[t2][index]},
            key=_atom_sort_key,
        )
        return ConnectedReactionOccurrence(
            step=step,
            occurrence=ReactionOccurrence(
                reaction=_format_reaction(reactants, products),
                timestep_reactants=t1,
                timestep_products=t2,
                reactants=reactants,
                products=products,
                reactant_atom_ids=reactant_atom_ids,
                product_atom_ids=product_atom_ids,
                simulation_index=getattr(simulation, "index", None),
            ),
            matched_direction=direction,
        )
    return None


def build_reaction_path_table(simulations) -> tuple[list[int], list[ReactionPath], dict[str, dict[int, int]]]:
    """Build total and per-simulation reaction counts from loaded simulations."""

    simulation_indices = []
    counts_by_reaction: dict[str, dict[int, int]] = {}
    all_paths: dict[str, int] = {}
    for simulation in simulations:
        if simulation.smiles is None or simulation.smiles_id is None:
            continue
        simulation_indices.append(simulation.index)
        for path in count_reaction_paths(
            simulation.smiles,
            simulation.smiles_id,
            getattr(simulation, "excluded_components", None),
            getattr(simulation, "structure_quality_mode", "exclude"),
        ):
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
    min_count: int = 1,
) -> list[ConnectedReactionPathway]:
    """Build possible reaction hierarchies from initially present species."""

    _validate_connected_pathway_options(notation, min_count)
    edge_data = _collect_pathway_edge_data(simulations, notation)
    if not edge_data.edge_counts:
        return []

    edge_data = _filter_pathway_edge_data(edge_data, min_count)
    if not edge_data.edge_counts:
        return []

    groups = _reachable_pathway_groups(
        edge_data.edge_counts,
        edge_data.edge_simulations,
        edge_data.edge_simulation_counts,
        edge_data.edge_orders,
        edge_data.initial_species,
    )
    if not groups:
        return []
    return _connected_pathways_from_groups(groups, edge_data.initial_species)


def _validate_connected_pathway_options(notation: str, min_count: int) -> None:
    """Validate connected pathway display options."""

    if notation not in {"formula", "smiles"}:
        raise ValueError("notation must be 'formula' or 'smiles'.")
    if min_count < 1:
        raise ValueError("min_count must be at least 1.")


def _collect_pathway_edge_data(simulations, notation: str) -> PathwayEdgeData:
    """Collect reaction-state edges from loaded simulations."""

    edge_data = PathwayEdgeData(
        initial_species=set(),
        edge_counts=Counter(),
        edge_simulations=defaultdict(set),
        edge_simulation_counts=defaultdict(Counter),
        edge_orders={},
    )
    order = 0
    for simulation in simulations:
        values_by_time = _pathway_values_by_time(simulation, notation)
        if values_by_time is None:
            continue
        timesteps = sorted(simulation.smiles)
        if not timesteps:
            continue

        initial_timestep = timesteps[1] if len(timesteps) > 1 else timesteps[0]
        excluded_initial = (getattr(simulation, "excluded_components", None) or {}).get(
            initial_timestep,
            set(),
        )
        edge_data.initial_species.update(
            value
            for index, value in enumerate(values_by_time[initial_timestep])
            if index not in excluded_initial
        )
        order = _collect_simulation_pathway_edges(
            simulation,
            values_by_time,
            timesteps,
            edge_data,
            order,
        )
    return edge_data


def _pathway_values_by_time(simulation, notation: str):
    """Return molecule labels by timestep for one simulation and notation."""

    if simulation.smiles is None or simulation.smiles_id is None:
        return None
    if notation == "formula":
        if simulation.chem_formulas is None:
            return None
        return simulation.chem_formulas
    return simulation.smiles


def _collect_simulation_pathway_edges(
    simulation,
    values_by_time,
    timesteps: list[int],
    edge_data: PathwayEdgeData,
    order: int,
) -> int:
    """Add connected-pathway edges for one simulation and return next order."""

    timestep_positions = {timestep: index for index, timestep in enumerate(timesteps)}
    initial_timestep = timesteps[1] if len(timesteps) > 1 else timesteps[0]
    initial_position = timestep_positions[initial_timestep]

    for t1, t2, reaction in _iter_reactions(
        simulation.smiles,
        simulation.smiles_id,
        getattr(simulation, "excluded_components", None),
        getattr(simulation, "structure_quality_mode", "exclude"),
    ):
        if timestep_positions[t1] < initial_position:
            continue
        source = tuple(sorted(values_by_time[t1][index] for index in reaction["reactants"]))
        target = tuple(sorted(values_by_time[t2][index] for index in reaction["products"]))
        if source == target:
            continue

        edge = (source, target)
        edge_data.edge_counts[edge] += 1
        edge_data.edge_simulations[edge].add(simulation.index)
        edge_data.edge_simulation_counts[edge][simulation.index] += 1
        edge_data.edge_orders[edge] = min(edge_data.edge_orders.get(edge, order), order)
        order += 1
    return order


def _filter_pathway_edge_data(edge_data: PathwayEdgeData, min_count: int) -> PathwayEdgeData:
    """Filter edge data by minimum total count."""

    edge_counts, edge_simulations, edge_simulation_counts, edge_orders = _filter_pathway_edges_by_count(
        edge_data.edge_counts,
        edge_data.edge_simulations,
        edge_data.edge_simulation_counts,
        edge_data.edge_orders,
        min_count,
    )
    return PathwayEdgeData(
        initial_species=edge_data.initial_species,
        edge_counts=edge_counts,
        edge_simulations=edge_simulations,
        edge_simulation_counts=edge_simulation_counts,
        edge_orders=edge_orders,
    )


def _connected_pathways_from_groups(
    groups: list[dict[str, object]],
    initial_species: set[str],
) -> list[ConnectedReactionPathway]:
    """Build connected pathway data objects from oriented reaction groups."""

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
    edge_simulation_counts: dict[tuple[tuple[str, ...], tuple[str, ...]], Counter[int]],
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
        counts_by_simulation: Counter[int] = Counter()
        count = 0
        for edge in (forward, reverse):
            if edge in edge_counts:
                count += edge_counts[edge]
                simulations.update(edge_simulations[edge])
                counts_by_simulation.update(edge_simulation_counts[edge])
        group["count"] = count
        group["simulations"] = tuple(sorted(simulations))
        group["counts_by_simulation"] = tuple(sorted(counts_by_simulation.items()))
        group["products"] = set(group["target"])
        groups.append(group)

    groups.sort(key=lambda group: (group["depth"], group["order"], _format_state(group["source"])))
    return groups


def _filter_pathway_edges_by_count(
    edge_counts: Counter[tuple[tuple[str, ...], tuple[str, ...]]],
    edge_simulations: dict[tuple[tuple[str, ...], tuple[str, ...]], set[int]],
    edge_simulation_counts: dict[tuple[tuple[str, ...], tuple[str, ...]], Counter[int]],
    edge_orders: dict[tuple[tuple[str, ...], tuple[str, ...]], int],
    min_count: int,
) -> tuple[
    Counter[tuple[tuple[str, ...], tuple[str, ...]]],
    dict[tuple[tuple[str, ...], tuple[str, ...]], set[int]],
    dict[tuple[tuple[str, ...], tuple[str, ...]], Counter[int]],
    dict[tuple[tuple[str, ...], tuple[str, ...]], int],
]:
    """Remove pathway edges whose displayed total would be below ``min_count``."""

    if min_count <= 1:
        return edge_counts, edge_simulations, edge_simulation_counts, edge_orders

    group_totals: Counter[tuple[tuple[str, ...], tuple[str, ...]]] = Counter()
    for edge, count in edge_counts.items():
        group_totals[_undirected_edge_key(*edge)] += count

    filtered_counts: Counter[tuple[tuple[str, ...], tuple[str, ...]]] = Counter()
    filtered_simulations: dict[tuple[tuple[str, ...], tuple[str, ...]], set[int]] = defaultdict(set)
    filtered_simulation_counts: dict[tuple[tuple[str, ...], tuple[str, ...]], Counter[int]] = defaultdict(Counter)
    filtered_orders: dict[tuple[tuple[str, ...], tuple[str, ...]], int] = {}

    for edge, count in edge_counts.items():
        if group_totals[_undirected_edge_key(*edge)] < min_count:
            continue
        filtered_counts[edge] = count
        filtered_simulations[edge] = set(edge_simulations[edge])
        filtered_simulation_counts[edge] = Counter(edge_simulation_counts[edge])
        filtered_orders[edge] = edge_orders[edge]

    return filtered_counts, filtered_simulations, filtered_simulation_counts, filtered_orders


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
                counts_by_simulation=group["counts_by_simulation"],
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


def _iter_reactions(
    smiles: dict[int, list[str]],
    smiles_id: dict[int, list[list[str]]],
    excluded_components: Mapping[int, set[int]] | None = None,
    quality_mode: str = "exclude",
):
    """Yield reaction clusters using the selected suspicious-structure policy."""

    if quality_mode == "skip" and excluded_components:
        yield from _iter_reactions_skipping_suspicious(
            smiles,
            smiles_id,
            excluded_components,
        )
        return

    for t1, t2, reaction in _iter_adjacent_reaction_clusters(smiles, smiles_id):
        excluded_t1 = (excluded_components or {}).get(t1, set())
        excluded_t2 = (excluded_components or {}).get(t2, set())
        if any(index in excluded_t1 for index in reaction["reactants"]):
            continue
        if any(index in excluded_t2 for index in reaction["products"]):
            continue
        yield t1, t2, reaction


def _iter_adjacent_reaction_clusters(
    smiles: dict[int, list[str]],
    smiles_id: dict[int, list[list[str]]],
):
    """Yield unfiltered atom-mapped reaction clusters between adjacent frames."""

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


def _iter_reactions_skipping_suspicious(
    smiles: dict[int, list[str]],
    smiles_id: dict[int, list[list[str]]],
    suspicious_components: Mapping[int, set[int]],
):
    """Bridge suspicious component lineages between their nearest clean states."""

    timesteps = sorted(smiles)
    timestep_positions = {timestep: index for index, timestep in enumerate(timesteps)}
    pending: list[_PendingSkipLineage] = []

    for t1, t2, reaction in _iter_adjacent_reaction_clusters(smiles, smiles_id):
        atom_ids = _reaction_cluster_atom_ids(reaction, t1, t2, smiles_id)
        overlapping = [lineage for lineage in pending if lineage.atom_ids & atom_ids]
        touches_suspicious = _reaction_touches_components(
            reaction,
            t1,
            t2,
            suspicious_components,
        )
        if not overlapping and not touches_suspicious:
            yield t1, t2, reaction
            continue

        merged_atoms = set(atom_ids)
        existing_baselines = []
        for lineage in overlapping:
            merged_atoms.update(lineage.atom_ids)
            if lineage.baseline_timestep is not None:
                existing_baselines.append(lineage.baseline_timestep)
            pending.remove(lineage)

        baseline = min(existing_baselines) if existing_baselines else _latest_clean_timestep(
            timesteps,
            timestep_positions[t1],
            merged_atoms,
            smiles_id,
            suspicious_components,
        )
        pending.append(_PendingSkipLineage(baseline, merged_atoms))
        pending = _merge_pending_lineages(pending, t2, smiles_id)

        resolved = []
        for lineage in pending:
            baseline_indexes, product_indexes, closed_atoms = _lineage_component_closure(
                lineage.baseline_timestep,
                t2,
                lineage.atom_ids,
                smiles_id,
            )
            lineage.atom_ids = closed_atoms
            if any(index in suspicious_components.get(t2, set()) for index in product_indexes):
                continue
            resolved.append(lineage)
            if lineage.baseline_timestep is None:
                continue
            if any(
                index in suspicious_components.get(lineage.baseline_timestep, set())
                for index in baseline_indexes
            ):
                continue
            yield lineage.baseline_timestep, t2, {
                "reactants": baseline_indexes,
                "products": product_indexes,
            }
        for lineage in resolved:
            pending.remove(lineage)


def _reaction_cluster_atom_ids(
    reaction: dict[str, list[int]],
    t1: int,
    t2: int,
    smiles_id: dict[int, list[list[str]]],
) -> set[str]:
    """Return all atoms participating in one adjacent reaction cluster."""

    return {
        atom_id
        for timestep, side in ((t1, "reactants"), (t2, "products"))
        for index in reaction[side]
        for atom_id in smiles_id[timestep][index]
    }


def _reaction_touches_components(
    reaction: dict[str, list[int]],
    t1: int,
    t2: int,
    components: Mapping[int, set[int]],
) -> bool:
    """Return whether a cluster contains a selected component on either side."""

    return any(index in components.get(t1, set()) for index in reaction["reactants"]) or any(
        index in components.get(t2, set()) for index in reaction["products"]
    )


def _latest_clean_timestep(
    timesteps: list[int],
    end_position: int,
    atom_ids: set[str],
    smiles_id: dict[int, list[list[str]]],
    suspicious_components: Mapping[int, set[int]],
) -> int | None:
    """Find the nearest earlier frame where the selected atom lineage was clean."""

    for timestep in reversed(timesteps[:end_position + 1]):
        indexes = _component_indexes_overlapping(smiles_id[timestep], atom_ids)
        observed_atoms = {
            atom_id for index in indexes for atom_id in smiles_id[timestep][index]
        }
        if atom_ids <= observed_atoms and not any(
            index in suspicious_components.get(timestep, set()) for index in indexes
        ):
            return timestep
    return None


def _merge_pending_lineages(
    pending: list[_PendingSkipLineage],
    current_timestep: int,
    smiles_id: dict[int, list[list[str]]],
) -> list[_PendingSkipLineage]:
    """Expand pending groups through current components and merge overlaps."""

    changed = True
    while changed:
        changed = False
        for lineage in pending:
            _, _, closed_atoms = _lineage_component_closure(
                lineage.baseline_timestep,
                current_timestep,
                lineage.atom_ids,
                smiles_id,
            )
            if closed_atoms != lineage.atom_ids:
                lineage.atom_ids = closed_atoms
                changed = True

        for index, first in enumerate(pending):
            overlapping_index = next(
                (
                    other_index
                    for other_index in range(index + 1, len(pending))
                    if first.atom_ids & pending[other_index].atom_ids
                ),
                None,
            )
            if overlapping_index is None:
                continue
            second = pending.pop(overlapping_index)
            first.atom_ids.update(second.atom_ids)
            baselines = [
                value
                for value in (first.baseline_timestep, second.baseline_timestep)
                if value is not None
            ]
            first.baseline_timestep = min(baselines) if baselines else None
            changed = True
            break
    return pending


def _lineage_component_closure(
    baseline_timestep: int | None,
    current_timestep: int,
    seed_atoms: set[str],
    smiles_id: dict[int, list[list[str]]],
) -> tuple[list[int], list[int], set[str]]:
    """Close an atom set over its baseline and current component partitions."""

    atom_ids = set(seed_atoms)
    baseline_indexes: list[int] = []
    current_indexes: list[int] = []
    while True:
        previous_atoms = set(atom_ids)
        baseline_indexes = (
            _component_indexes_overlapping(smiles_id[baseline_timestep], atom_ids)
            if baseline_timestep is not None
            else []
        )
        current_indexes = _component_indexes_overlapping(smiles_id[current_timestep], atom_ids)
        for timestep, indexes in (
            (baseline_timestep, baseline_indexes),
            (current_timestep, current_indexes),
        ):
            if timestep is None:
                continue
            for index in indexes:
                atom_ids.update(smiles_id[timestep][index])
        if atom_ids == previous_atoms:
            return baseline_indexes, current_indexes, atom_ids


def _component_indexes_overlapping(
    components: list[list[str]],
    atom_ids: set[str],
) -> list[int]:
    """Return component indexes containing at least one selected atom."""

    return [index for index, component in enumerate(components) if atom_ids.intersection(component)]


def _atom_sort_key(atom_id: str) -> tuple[int, str]:
    """Sort atom ids numerically when possible, with a string fallback."""

    try:
        return int(atom_id), atom_id
    except ValueError:
        return 0, atom_id
