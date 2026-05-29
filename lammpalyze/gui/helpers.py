"""Small data and sizing helpers for the Tkinter GUI."""

from __future__ import annotations

from pathlib import Path

from lammpalyze.reactions import ReactionPath, build_reaction_path_table

DEFAULT_IMAGE_EXTENSION = ".png"
IMAGE_FILETYPES = (
    ("PNG image", "*.png"),
    ("PDF document", "*.pdf"),
    ("SVG image", "*.svg"),
    ("JPEG image", "*.jpg"),
    ("All files", "*.*"),
)
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
