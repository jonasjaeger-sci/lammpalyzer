"""Helpers for drawing connected reaction pathways as directed graphs."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from lammpalyze.reactions import ConnectedReactionPathway, ConnectedReactionStep


@dataclass(frozen=True)
class PathwayGraphNode:
    """One rendered state in a selected connected-pathway branch."""

    key: str
    label: str
    depth: int
    source_step_label: str | None = None
    source_side: str | None = None


@dataclass(frozen=True)
class PathwayGraphEdge:
    """One rendered reaction arrow between two state nodes."""

    source_key: str
    target_key: str
    step_label: str
    arrow: str
    count: int


@dataclass(frozen=True)
class PathwayGraph:
    """Nodes and arrows for a selected connected-pathway branch."""

    nodes: tuple[PathwayGraphNode, ...]
    edges: tuple[PathwayGraphEdge, ...]


def pathway_graph_choices(pathway: ConnectedReactionPathway) -> list[tuple[str, str]]:
    """Return selectable pathway labels for graph rendering."""

    return [
        (
            step.label,
            f"{step.label} [depth {step.depth}]: {step.source} {step.arrow} {step.target} (n={step.count})",
        )
        for step in pathway.steps
    ]


def build_pathway_graph(
    pathway: ConnectedReactionPathway,
    root_label: str | None = None,
) -> PathwayGraph:
    """Build a top-down graph for one selected root pathway branch."""

    if not pathway.steps:
        return PathwayGraph(nodes=(), edges=())

    steps_by_label = {step.label: step for step in pathway.steps}
    children_by_parent: dict[str, list[ConnectedReactionStep]] = defaultdict(list)
    roots = []
    for step in pathway.steps:
        if step.parents:
            for parent in step.parents:
                children_by_parent[parent].append(step)
        else:
            roots.append(step)

    root = steps_by_label.get(root_label or "")
    if root is None:
        root = roots[0] if roots else pathway.steps[0]

    branch_steps = _collect_branch_steps(root, children_by_parent)
    root_source_depth = max(0, root.depth - 1)
    node_by_key: dict[str, PathwayGraphNode] = {}
    edges = []
    for step in branch_steps:
        source_depth = max(0, step.depth - 1 - root_source_depth)
        target_depth = max(source_depth + 1, step.depth - root_source_depth)
        source_key = _node_key(step.source, source_depth)
        target_key = _node_key(step.target, target_depth)
        node_by_key.setdefault(
            source_key,
            PathwayGraphNode(
                key=source_key,
                label=step.source,
                depth=source_depth,
                source_step_label=step.label,
                source_side="reactants",
            ),
        )
        node_by_key.setdefault(
            target_key,
            PathwayGraphNode(
                key=target_key,
                label=step.target,
                depth=target_depth,
                source_step_label=step.label,
                source_side="products",
            ),
        )
        edges.append(
            PathwayGraphEdge(
                source_key=source_key,
                target_key=target_key,
                step_label=step.label,
                arrow=step.arrow,
                count=step.count,
            )
        )

    nodes = sorted(node_by_key.values(), key=lambda node: (node.depth, node.label, node.key))
    return PathwayGraph(nodes=tuple(nodes), edges=tuple(edges))


def _collect_branch_steps(
    root: ConnectedReactionStep,
    children_by_parent: dict[str, list[ConnectedReactionStep]],
) -> list[ConnectedReactionStep]:
    """Return ``root`` and descendants, guarding against accidental cycles."""

    collected = []
    stack = [(root, ())]
    while stack:
        step, active = stack.pop()
        if step.label in active:
            continue
        collected.append(step)
        children = children_by_parent.get(step.label, [])
        for child in reversed(children):
            stack.append((child, (*active, step.label)))
    return collected


def _node_key(label: str, depth: int) -> str:
    """Return a stable node key for one state at one depth."""

    return f"{depth}:{label}"


def pathway_graph_image_extent(
    x: float,
    y: float,
    node_width: float,
    node_height: float,
    image_shape,
) -> tuple[float, float, float, float]:
    """Return a node-image extent that preserves the source image aspect ratio."""

    image_height, image_width = image_shape[:2]
    if image_height <= 0 or image_width <= 0:
        return (
            x - node_width * 0.47,
            x + node_width * 0.47,
            y - node_height * 0.12,
            y + node_height * 0.47,
        )

    max_width = node_width * 0.94
    max_height = node_height * 0.59
    image_aspect = image_width / image_height
    target_aspect = max_width / max_height
    if target_aspect > image_aspect:
        draw_height = max_height
        draw_width = draw_height * image_aspect
    else:
        draw_width = max_width
        draw_height = draw_width / image_aspect

    center_y = y + node_height * 0.175
    return (
        x - draw_width / 2,
        x + draw_width / 2,
        center_y - draw_height / 2,
        center_y + draw_height / 2,
    )
