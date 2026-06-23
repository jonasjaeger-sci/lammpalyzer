"""Small data and sizing helpers for the Tkinter GUI."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from statistics import fmean

from lammpalyze.reactions import (
    ConnectedReactionPathway,
    ReactionPath,
    build_connected_reaction_pathways,
    build_reaction_path_table,
)
from lammpalyze.smiles import reaction_smiles_groups, reaction_smiles_path

DEFAULT_IMAGE_EXTENSION = ".png"
IMAGE_FILETYPES = (
    ("PNG image", "*.png"),
    ("PDF document", "*.pdf"),
    ("SVG image", "*.svg"),
    ("JPEG image", "*.jpg"),
    ("All files", "*.*"),
)
PNG_FILETYPES = (("PNG image", "*.png"),)
RASTER_IMAGE_FILETYPES = (
    ("PNG image", "*.png"),
    ("JPEG image", "*.jpg"),
    ("All files", "*.*"),
)
MOLECULE_IMAGE_PADDING = 24
MOLECULE_IMAGE_FALLBACK_SIZE = (720, 520)
MOLECULE_IMAGE_MAX_SIZE = (1800, 1400)
MOLECULE_RESIZE_DEBOUNCE_MS = 150
THERMO_DEFAULTS = ["Temp", "PotEng", "KinEng", "Press", "Volume", "Density"]
REFERENCE_LINE_SPLIT_PATTERN = re.compile(r"[\s,;]+")


def molecule_render_size(container_width: int, container_height: int) -> tuple[int, int]:
    """Return a molecule image size that follows the available GUI area."""

    if container_width <= MOLECULE_IMAGE_PADDING or container_height <= MOLECULE_IMAGE_PADDING:
        return MOLECULE_IMAGE_FALLBACK_SIZE

    image_width = min(container_width - MOLECULE_IMAGE_PADDING, MOLECULE_IMAGE_MAX_SIZE[0])
    image_height = min(container_height - MOLECULE_IMAGE_PADDING, MOLECULE_IMAGE_MAX_SIZE[1])
    return max(1, image_width), max(1, image_height)


def reaction_path_table_data(simulations) -> tuple[list[int], list[ReactionPath], dict[str, dict[int, int]]]:
    """Return simulation indexes, total paths, and per-simulation counts."""

    return build_reaction_path_table(simulations)


def reaction_path_ids(paths: list[ReactionPath]) -> dict[str, str]:
    """Assign stable display IDs, marking reverse reactions with ``*``."""

    _, identifiers = reaction_path_display_order(paths)
    return identifiers


def reaction_path_display_order(paths: list[ReactionPath]) -> tuple[list[ReactionPath], dict[str, str]]:
    """Return reaction paths grouped with their reverse paths, plus display IDs."""

    path_by_reaction = {path.reaction: path for path in paths}
    identifiers = {}
    ordered_paths = []
    seen = set()
    next_identifier = 1

    for path in paths:
        if path.reaction in seen:
            continue

        identifiers[path.reaction] = str(next_identifier)
        ordered_paths.append(path)
        seen.add(path.reaction)

        reverse_reaction = _reverse_reaction_path(path.reaction)
        if reverse_reaction in path_by_reaction and reverse_reaction not in seen:
            identifiers[reverse_reaction] = f"{next_identifier}*"
            ordered_paths.append(path_by_reaction[reverse_reaction])
            seen.add(reverse_reaction)

        next_identifier += 1

    return ordered_paths, identifiers


def _reverse_reaction_path(reaction: str) -> str:
    """Return the formatted reverse reaction path, or an empty string."""

    try:
        reactants, products = reaction_smiles_groups(reaction)
    except ValueError:
        return ""
    return reaction_smiles_path(reactants=products, products=reactants)


def connected_reaction_pathway_data(
    simulations,
    notation: str = "formula",
    min_count: int = 1,
) -> list[ConnectedReactionPathway]:
    """Return connected reaction pathways in formula or SMILES notation."""

    return build_connected_reaction_pathways(simulations, notation=notation, min_count=min_count)


def molecule_observation_summary(simulation, target_smiles: str) -> str:
    """Summarize component charge, ion candidates, and flags for one SMILES."""

    if not simulation.smiles or not simulation.component_properties or not target_smiles:
        return ""
    observations = [
        simulation.component_properties[timestep][index]
        for timestep, smiles_values in simulation.smiles.items()
        for index, smiles in enumerate(smiles_values)
        if smiles == target_smiles
        and timestep in simulation.component_properties
        and index < len(simulation.component_properties[timestep])
    ]
    if not observations:
        return ""
    charges = [properties.charge for properties in observations]
    ion_counts = Counter(
        properties.ion_candidate for properties in observations if properties.ion_candidate
    )
    suspicious_count = sum(properties.suspicious for properties in observations)
    summary = (
        f"Observed {len(observations)} time(s); component charge mean {fmean(charges):+.3f} e "
        f"(range {min(charges):+.3f} to {max(charges):+.3f} e)."
    )
    if ion_counts:
        labels = ", ".join(f"{label}: {count}" for label, count in sorted(ion_counts.items()))
        summary += f" Ion candidates: {labels}."
    if suspicious_count:
        summary += f" Suspicious observations: {suspicious_count}."
    return summary


def image_output_path(filename: str) -> Path:
    """Return an image path, defaulting to PNG when no suffix is provided."""

    path = Path(filename)
    if not path.suffix:
        path = path.with_suffix(DEFAULT_IMAGE_EXTENSION)
    return path


def suffixed_image_output_path(filename: str, suffix: str) -> Path:
    """Return an image path with ``suffix`` inserted before the extension."""

    path = image_output_path(filename)
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def parse_reference_lines(value: str) -> list[float]:
    """Parse comma-, semicolon-, or whitespace-separated reference-line values."""

    stripped = value.strip()
    if not stripped:
        return []
    lines = []
    for token in REFERENCE_LINE_SPLIT_PATTERN.split(stripped):
        if token:
            lines.append(float(token))
    return lines


def parse_timestep_values(value: str) -> list[int]:
    """Parse comma-, semicolon-, or whitespace-separated timestep values."""

    stripped = re.sub(r"[()\[\]{}]", " ", value).strip()
    if not stripped:
        return []
    timesteps = []
    for token in REFERENCE_LINE_SPLIT_PATTERN.split(stripped):
        if token:
            timesteps.append(int(token))
    return timesteps


def parse_simulation_groups(value: str) -> list[list[int]]:
    """Parse semicolon-separated simulation-index groups."""

    groups = []
    for raw_group in value.split(";"):
        stripped = raw_group.strip()
        if not stripped:
            continue
        group = []
        seen = set()
        for token in REFERENCE_LINE_SPLIT_PATTERN.split(stripped):
            if not token:
                continue
            index = int(token)
            if index not in seen:
                group.append(index)
                seen.add(index)
        if group:
            groups.append(group)
    return groups
