"""OVITO scene generation for reaction visualization."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from math import sqrt
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from lammpalyze.analysis import LoadedSimulation
from lammpalyze.parsers import ReaxBond, TrajectoryAtom, TrajectoryFrame
from lammpalyze.reactions import ReactionOccurrence


ELEMENT_MASSES = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "Li": 6.94,
    "Na": 22.990,
    "Mg": 24.305,
    "P": 30.974,
    "S": 32.06,
    "Cl": 35.45,
    "Br": 79.904,
    "I": 126.904,
}

CPK_RADII = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "Li": 1.82,
    "Na": 2.27,
    "Mg": 1.73,
    "P": 1.80,
    "S": 1.80,
    "Cl": 1.75,
    "Br": 1.85,
    "I": 1.98,
}

SPHERE_DIAMETER_SCALE = 0.50
OTHER_SPHERE_DIAMETER = 0.55
DEFAULT_SPHERE_DIAMETER = 0.65
SPHERE_DENSITY = 1.0


class OvitoNotAvailableError(FileNotFoundError):
    """Raised when the OVITO executable cannot be found."""


@dataclass(frozen=True)
class OvitoScene:
    """Files needed to open a generated OVITO reaction scene."""

    directory: Path
    data_file: Path
    info_file: Path

    @property
    def dump_file(self) -> Path:
        """Backward-compatible alias for the OVITO-opened scene file."""

        return self.data_file


@dataclass(frozen=True)
class ReactionStateSnapshot:
    """Files for one rendered reaction state thumbnail."""

    directory: Path
    data_file: Path
    image_file: Path
    info_file: Path
    renderer: str


def normalize_reaction_path(text: str) -> str:
    """Normalize editable/dropdown text copied from ``paths.out``."""

    stripped = text.strip()
    if "\t" in stripped:
        return stripped.split("\t", maxsplit=1)[0].strip()
    return stripped


def create_reaction_scene(
    simulation: LoadedSimulation,
    occurrence: ReactionOccurrence,
    *,
    output_dir: str | Path | None = None,
) -> OvitoScene:
    """Create a side-by-side reactant/product ball-and-stick scene for OVITO."""

    if simulation.trajectory_path is None:
        raise ValueError(f"Simulation {simulation.index} has no trajectory file.")
    if simulation.bond_path is None:
        raise ValueError(f"Simulation {simulation.index} has no ReaxFF bond file.")
    if simulation.type_to_element is None:
        raise ValueError(f"Simulation {simulation.index} has no atom type to element mapping.")

    directory = Path(output_dir) if output_dir is not None else Path(tempfile.mkdtemp(prefix="lammpalyze_ovito_"))
    directory.mkdir(parents=True, exist_ok=True)

    reactant_frame = simulation.read_trajectory_frame(occurrence.timestep_reactants)
    product_frame = simulation.read_trajectory_frame(occurrence.timestep_products)
    reactant_bonds = simulation.read_bond_frame(occurrence.timestep_reactants)
    product_bonds = simulation.read_bond_frame(occurrence.timestep_products)

    data_file = directory / "reaction_side_by_side.data"
    info_file = directory / "reaction_scene.txt"
    _write_side_by_side_data(
        data_file,
        reactant_frame,
        product_frame,
        reactant_bonds,
        product_bonds,
        occurrence,
        simulation.type_to_element,
    )
    _write_scene_info(info_file, occurrence)
    return OvitoScene(directory=directory, data_file=data_file, info_file=info_file)


def create_reaction_state_snapshot(
    simulation: LoadedSimulation,
    occurrence: ReactionOccurrence,
    side: str,
    *,
    output_dir: str | Path | None = None,
    image_size: tuple[int, int] = (320, 220),
) -> ReactionStateSnapshot:
    """Create an image for one reactant or product state.

    A LAMMPS data file is written alongside the PNG so the exact highlighted
    state can be opened in OVITO later.  Inline thumbnails prefer OVITO's
    Python renderer when available and otherwise use a small Matplotlib
    ball-and-stick fallback.
    """

    if side not in {"reactants", "products"}:
        raise ValueError("side must be 'reactants' or 'products'.")
    if simulation.trajectory_path is None:
        raise ValueError(f"Simulation {simulation.index} has no trajectory file.")
    if simulation.bond_path is None:
        raise ValueError(f"Simulation {simulation.index} has no ReaxFF bond file.")
    if simulation.type_to_element is None:
        raise ValueError(f"Simulation {simulation.index} has no atom type to element mapping.")

    directory = Path(output_dir) if output_dir is not None else Path(tempfile.mkdtemp(prefix="lammpalyze_state_"))
    directory.mkdir(parents=True, exist_ok=True)
    timestep = occurrence.timestep_reactants if side == "reactants" else occurrence.timestep_products
    atom_ids = occurrence.reactant_atom_ids if side == "reactants" else occurrence.product_atom_ids
    frame = simulation.read_trajectory_frame(timestep)
    bonds = simulation.read_bond_frame(timestep)

    data_file = directory / f"{side}_{timestep}.data"
    image_file = directory / f"{side}_{timestep}.png"
    info_file = directory / f"{side}_{timestep}.txt"
    reaction_atom_ids = {int(atom_id) for atom_id in atom_ids}

    _write_single_state_data(
        data_file,
        frame,
        bonds,
        reaction_atom_ids,
        simulation.type_to_element,
    )
    focus_atoms = _focus_atoms(frame, reaction_atom_ids)
    renderer = _render_state_snapshot_with_ovito(data_file, image_file, image_size, focus_atoms)
    if renderer is None:
        _render_state_snapshot_with_matplotlib(
            frame,
            bonds,
            reaction_atom_ids,
            simulation.type_to_element,
            image_file,
            image_size,
        )
        renderer = "matplotlib"
    _write_state_snapshot_info(info_file, occurrence, side, timestep, renderer)
    return ReactionStateSnapshot(
        directory=directory,
        data_file=data_file,
        image_file=image_file,
        info_file=info_file,
        renderer=renderer,
    )


def launch_ovito_scene(scene: OvitoScene, ovito_executable: str | None = None) -> subprocess.Popen:
    """Launch OVITO with the generated LAMMPS data file."""

    executable = ovito_executable or _find_ovito_executable()
    if executable is not None:
        return subprocess.Popen([executable, str(scene.data_file)])

    raise OvitoNotAvailableError(
        "OVITO is not installed or could not be found. Install OVITO, add it to PATH, "
        "or set OVITO_BIN to the executable path to use reaction visualization."
    )


def _find_ovito_executable() -> str | None:
    """Return an OVITO executable path from environment, PATH, or defaults."""

    for env_name in ("OVITO_BIN", "OVITO_bin", "ovito_bin"):
        executable = os.environ.get(env_name)
        if executable:
            return executable

    for executable_name in ("ovito", "OVITO", "Ovito"):
        executable = shutil.which(executable_name)
        if executable is not None:
            return executable

    ovito_appimage = Path.home() / "bin" / "ovito"
    if ovito_appimage.exists():
        return str(ovito_appimage)

    return None


def _write_side_by_side_data(
    output_file: Path,
    reactant_frame: TrajectoryFrame,
    product_frame: TrajectoryFrame,
    reactant_bonds: list[ReaxBond],
    product_bonds: list[ReaxBond],
    occurrence: ReactionOccurrence,
    type_to_element: dict[int, str],
) -> None:
    """Write reactant and product frames as one OVITO-readable data file."""

    max_atom_id = max(atom.atom_id for atom in reactant_frame.atoms + product_frame.atoms)
    x_length = reactant_frame.bounds[0, 1] - reactant_frame.bounds[0, 0]
    x_gap = x_length * 0.35
    left_shift = -0.5 * (x_length + x_gap)
    right_shift = 0.5 * (x_length + x_gap)

    reactant_atom_ids = {int(atom_id) for atom_id in occurrence.reactant_atom_ids}
    product_atom_ids = {int(atom_id) for atom_id in occurrence.product_atom_ids}
    visual_type_map = _visual_type_map(type_to_element)
    reactant_visual_atoms = [
        _visual_atom(
            atom,
            atom.atom_id,
            left_shift,
            atom.atom_id in reactant_atom_ids,
            type_to_element,
            visual_type_map,
        )
        for atom in reactant_frame.atoms
    ]
    product_visual_atoms = [
        _visual_atom(
            atom,
            atom.atom_id + max_atom_id,
            right_shift,
            atom.atom_id in product_atom_ids,
            type_to_element,
            visual_type_map,
        )
        for atom in product_frame.atoms
    ]
    visual_atoms = reactant_visual_atoms + product_visual_atoms
    x_bounds, y_bounds, z_bounds = _bounds_from_visual_atoms(visual_atoms)
    reactant_visual_bonds = _visual_bonds(reactant_bonds, atom_id_offset=0, start_bond_id=1)
    product_visual_bonds = _visual_bonds(
        product_bonds,
        atom_id_offset=max_atom_id,
        start_bond_id=len(reactant_visual_bonds) + 1,
    )
    visual_bonds = reactant_visual_bonds + product_visual_bonds

    with output_file.open("w", encoding="utf-8") as handle:
        handle.write("LAMMPS data file generated by lammpalyze for OVITO ball-and-stick reaction view\n\n")
        handle.write(f"{len(visual_atoms)} atoms\n")
        handle.write(f"{len(visual_bonds)} bonds\n")
        handle.write(f"{len(visual_type_map)} atom types\n")
        handle.write("1 bond types\n\n")
        handle.write(f"{x_bounds[0]:.8f} {x_bounds[1]:.8f} xlo xhi\n")
        handle.write(f"{y_bounds[0]:.8f} {y_bounds[1]:.8f} ylo yhi\n")
        handle.write(f"{z_bounds[0]:.8f} {z_bounds[1]:.8f} zlo zhi\n\n")
        handle.write("Masses\n\n")
        for visual_type, element in sorted(visual_type_map.items()):
            mass = ELEMENT_MASSES.get(element, 1.0)
            label = "Other" if element == "Other" else element
            handle.write(f"{visual_type} {mass:.6f} # {label}\n")
        handle.write("\nAtoms # sphere\n\n")
        for atom_id, visual_type, diameter, x, y, z in visual_atoms:
            handle.write(f"{atom_id} {visual_type} {diameter:.4f} {SPHERE_DENSITY:.4f} {x:.8f} {y:.8f} {z:.8f}\n")
        handle.write("\nBonds\n\n")
        for bond_id, atom_i, atom_j in visual_bonds:
            handle.write(f"{bond_id} 1 {atom_i} {atom_j}\n")


def _write_single_state_data(
    output_file: Path,
    frame: TrajectoryFrame,
    bonds: list[ReaxBond],
    reaction_atom_ids: set[int],
    type_to_element: dict[int, str],
) -> None:
    """Write one highlighted frame as an OVITO-readable LAMMPS data file."""

    visual_type_map = _visual_type_map(type_to_element)
    visual_atoms = [
        _visual_atom(
            atom,
            atom.atom_id,
            0.0,
            atom.atom_id in reaction_atom_ids,
            type_to_element,
            visual_type_map,
        )
        for atom in frame.atoms
    ]
    visual_bonds = _visual_bonds(bonds, atom_id_offset=0, start_bond_id=1)
    x_bounds, y_bounds, z_bounds = _bounds_from_visual_atoms(visual_atoms)

    with output_file.open("w", encoding="utf-8") as handle:
        handle.write("LAMMPS data file generated by lammpalyze for an OVITO reaction-state view\n\n")
        handle.write(f"{len(visual_atoms)} atoms\n")
        handle.write(f"{len(visual_bonds)} bonds\n")
        handle.write(f"{len(visual_type_map)} atom types\n")
        handle.write("1 bond types\n\n")
        handle.write(f"{x_bounds[0]:.8f} {x_bounds[1]:.8f} xlo xhi\n")
        handle.write(f"{y_bounds[0]:.8f} {y_bounds[1]:.8f} ylo yhi\n")
        handle.write(f"{z_bounds[0]:.8f} {z_bounds[1]:.8f} zlo zhi\n\n")
        handle.write("Masses\n\n")
        for visual_type, element in sorted(visual_type_map.items()):
            mass = ELEMENT_MASSES.get(element, 1.0)
            label = "Other" if element == "Other" else element
            handle.write(f"{visual_type} {mass:.6f} # {label}\n")
        handle.write("\nAtoms # sphere\n\n")
        for atom_id, visual_type, diameter, x, y, z in visual_atoms:
            handle.write(
                f"{atom_id} {visual_type} {diameter:.4f} {SPHERE_DENSITY:.4f} "
                f"{x:.8f} {y:.8f} {z:.8f}\n"
            )
        handle.write("\nBonds\n\n")
        for bond_id, atom_i, atom_j in visual_bonds:
            handle.write(f"{bond_id} 1 {atom_i} {atom_j}\n")


def _render_state_snapshot_with_ovito(
    data_file: Path,
    image_file: Path,
    image_size: tuple[int, int],
    focus_atoms: list[TrajectoryAtom],
) -> str | None:
    """Render ``data_file`` with OVITO's Python API, if it is installed."""

    try:
        from ovito.io import import_file  # type: ignore[import-not-found]
        from ovito.vis import TachyonRenderer, Viewport  # type: ignore[import-not-found]
    except Exception:
        return None

    pipeline = None
    try:
        pipeline = import_file(str(data_file))
        pipeline.add_to_scene()
        center, radius = _camera_focus(focus_atoms)
        camera_direction = (0.55, -0.70, -0.46)
        distance = max(radius * 5.5, 6.0)
        viewport = Viewport(type=Viewport.Type.Ortho)
        viewport.camera_dir = camera_direction
        viewport.camera_pos = tuple(
            center[index] - camera_direction[index] * distance
            for index in range(3)
        )
        viewport.fov = max(radius * 1.8, 2.2)
        viewport.render_image(
            filename=str(image_file),
            size=image_size,
            renderer=TachyonRenderer(),
        )
    except Exception:
        return None
    finally:
        if pipeline is not None:
            try:
                pipeline.remove_from_scene()
            except Exception:
                pass
    return "ovito-python"


def _render_state_snapshot_with_matplotlib(
    frame: TrajectoryFrame,
    bonds: list[ReaxBond],
    reaction_atom_ids: set[int],
    type_to_element: dict[int, str],
    image_file: Path,
    image_size: tuple[int, int],
) -> None:
    """Render a compact ball-and-stick PNG without requiring OVITO."""

    width, height = image_size
    figure = Figure(figsize=(width / 100, height / 100), dpi=100)
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(111, projection="3d")
    atom_by_id = {atom.atom_id: atom for atom in frame.atoms}
    context_atoms = [atom for atom in frame.atoms if atom.atom_id not in reaction_atom_ids]
    reaction_atoms = [atom for atom in frame.atoms if atom.atom_id in reaction_atom_ids]

    _plot_atoms(axis, context_atoms, type_to_element, alpha=0.16, size=18)
    _plot_bonds(axis, bonds, atom_by_id, color="#9a9a9a", alpha=0.16, linewidth=0.7)
    _plot_bonds(
        axis,
        [
            bond
            for bond in bonds
            if bond.atom_i in reaction_atom_ids and bond.atom_j in reaction_atom_ids
        ],
        atom_by_id,
        color="#2f2f2f",
        alpha=0.75,
        linewidth=1.4,
    )
    _plot_atoms(axis, reaction_atoms, type_to_element, alpha=0.95, size=62)
    _set_equal_3d_bounds(axis, _focus_atoms(frame, reaction_atom_ids))
    axis.view_init(elev=24, azim=-58)
    axis.set_axis_off()
    figure.patch.set_alpha(0.0)
    axis.set_facecolor((1, 1, 1, 0))
    figure.savefig(image_file, bbox_inches="tight", pad_inches=0.02, transparent=True)
    figure.clear()


def _plot_atoms(axis, atoms: list[TrajectoryAtom], type_to_element: dict[int, str], *, alpha: float, size: int) -> None:
    """Plot atoms grouped by element color."""

    if not atoms:
        return
    grouped: dict[str, list[TrajectoryAtom]] = {}
    for atom in atoms:
        grouped.setdefault(type_to_element.get(atom.atom_type, "Other"), []).append(atom)
    for element, element_atoms in grouped.items():
        axis.scatter(
            [atom.x for atom in element_atoms],
            [atom.y for atom in element_atoms],
            [atom.z for atom in element_atoms],
            s=size,
            c=_element_color(element),
            edgecolors="#2a2a2a" if alpha > 0.5 else "none",
            linewidths=0.35,
            alpha=alpha,
            depthshade=True,
        )


def _plot_bonds(
    axis,
    bonds: list[ReaxBond],
    atom_by_id: dict[int, TrajectoryAtom],
    *,
    color: str,
    alpha: float,
    linewidth: float,
) -> None:
    """Plot bond segments for atoms present in ``atom_by_id``."""

    for bond in bonds:
        atom_i = atom_by_id.get(bond.atom_i)
        atom_j = atom_by_id.get(bond.atom_j)
        if atom_i is None or atom_j is None:
            continue
        axis.plot(
            [atom_i.x, atom_j.x],
            [atom_i.y, atom_j.y],
            [atom_i.z, atom_j.z],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
        )


def _focus_atoms(frame: TrajectoryFrame, reaction_atom_ids: set[int]) -> list[TrajectoryAtom]:
    """Return atoms used for snapshot camera framing."""

    reaction_atoms = [atom for atom in frame.atoms if atom.atom_id in reaction_atom_ids]
    return reaction_atoms or frame.atoms


def _camera_focus(atoms: list[TrajectoryAtom]) -> tuple[tuple[float, float, float], float]:
    """Return center and radius for a focused snapshot camera."""

    if not atoms:
        return (0.0, 0.0, 0.0), 1.0
    x_values = [atom.x for atom in atoms]
    y_values = [atom.y for atom in atoms]
    z_values = [atom.z for atom in atoms]
    center = (
        0.5 * (min(x_values) + max(x_values)),
        0.5 * (min(y_values) + max(y_values)),
        0.5 * (min(z_values) + max(z_values)),
    )
    span = max(
        max(x_values) - min(x_values),
        max(y_values) - min(y_values),
        max(z_values) - min(z_values),
        1.0,
    )
    diagonal = sqrt(
        (max(x_values) - min(x_values)) ** 2
        + (max(y_values) - min(y_values)) ** 2
        + (max(z_values) - min(z_values)) ** 2
    )
    radius = max(0.34 * max(span, diagonal) + 0.25, 0.85)
    return center, radius


def _set_equal_3d_bounds(axis, atoms: list[TrajectoryAtom]) -> None:
    """Keep snapshots from appearing distorted by unequal axis spans."""

    centers, radius = _camera_focus(atoms)
    axis.set_xlim(centers[0] - radius, centers[0] + radius)
    axis.set_ylim(centers[1] - radius, centers[1] + radius)
    axis.set_zlim(centers[2] - radius, centers[2] + radius)


def _visual_type_map(type_to_element: dict[int, str]) -> dict[int, str]:
    """Map visual atom types to element labels, reserving type one for context."""

    elements = sorted(set(type_to_element.values()))
    return {1: "X"} | {index + 2: element for index, element in enumerate(elements)}


def _visual_atom(
    atom: TrajectoryAtom,
    atom_id: int,
    x_shift: float,
    is_reaction_atom: bool,
    type_to_element: dict[int, str],
    visual_type_map: dict[int, str],
) -> tuple[int, int, float, float, float, float]:
    """Return one shifted atom row for the OVITO sphere data file."""

    element = type_to_element.get(atom.atom_type)
    visual_type = 1
    diameter = OTHER_SPHERE_DIAMETER
    if is_reaction_atom and element is not None:
        visual_type = next(key for key, value in visual_type_map.items() if value == element)
        diameter = _element_diameter(element)
    return atom_id, visual_type, diameter, atom.x + x_shift, atom.y, atom.z


def _visual_bonds(bonds: list[ReaxBond], atom_id_offset: int, start_bond_id: int) -> list[tuple[int, int, int]]:
    """Return OVITO bond rows with adjusted atom ids and sequential bond ids."""

    return [
        (bond_index, bond.atom_i + atom_id_offset, bond.atom_j + atom_id_offset)
        for bond_index, bond in enumerate(bonds, start=start_bond_id)
    ]


def _bounds_from_visual_atoms(
    visual_atoms: list[tuple[int, int, float, float, float, float]],
    padding: float = 4.0,
) -> tuple[list[float], list[float], list[float]]:
    """Return padded x, y, and z bounds around visual atom coordinates."""

    x_values = [atom[3] for atom in visual_atoms]
    y_values = [atom[4] for atom in visual_atoms]
    z_values = [atom[5] for atom in visual_atoms]
    return (
        [min(x_values) - padding, max(x_values) + padding],
        [min(y_values) - padding, max(y_values) + padding],
        [min(z_values) - padding, max(z_values) + padding],
    )


def _element_diameter(element: str) -> float:
    """Return the rendered sphere diameter for a chemical element."""

    radius = CPK_RADII.get(element)
    if radius is None:
        return DEFAULT_SPHERE_DIAMETER
    return radius * SPHERE_DIAMETER_SCALE


def _element_color(element: str) -> str:
    """Return a conventional compact color for atom snapshots."""

    colors = {
        "H": "#f4f4f4",
        "C": "#343434",
        "N": "#2f5dff",
        "O": "#e23b30",
        "F": "#47b45a",
        "Li": "#8a5fd1",
        "Na": "#6d8fe8",
        "Mg": "#25a24a",
        "P": "#ef8a24",
        "S": "#e7c72f",
        "Cl": "#2fa84f",
        "Br": "#8b3f2f",
        "I": "#6f3b99",
        "Other": "#888888",
    }
    return colors.get(element, "#888888")


def _write_scene_info(info_file: Path, occurrence: ReactionOccurrence) -> None:
    """Write a companion text summary for a generated OVITO scene."""

    info_file.write_text(
        "\n".join(
            [
                "lammpalyze OVITO reaction scene",
                f"Reaction: {occurrence.reaction}",
                f"Left/reactants timestep: {occurrence.timestep_reactants}",
                f"Right/products timestep: {occurrence.timestep_products}",
                "The opened LAMMPS data file contains atom positions and ReaxFF bonds.",
                "The Atoms section uses LAMMPS sphere style with reduced diameters.",
                "Bonds are exported in the LAMMPS data Bonds section for OVITO rendering.",
                "Coordinates are exported from unwrapped trajectory columns when available.",
                "Atom type 1 / X: non-reaction atoms",
                "Atom types 2+: reaction atoms grouped by element.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_state_snapshot_info(
    info_file: Path,
    occurrence: ReactionOccurrence,
    side: str,
    timestep: int,
    renderer: str,
) -> None:
    """Write a companion text summary for one state snapshot."""

    info_file.write_text(
        "\n".join(
            [
                "lammpalyze reaction-state snapshot",
                f"Reaction: {occurrence.reaction}",
                f"Side: {side}",
                f"Timestep: {timestep}",
                f"Renderer: {renderer}",
                "The LAMMPS data file is OVITO-readable and highlights reaction atoms by element.",
                "Atom type 1 / X: non-reaction atoms.",
                "Atom types 2+: reaction atoms grouped by element.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
