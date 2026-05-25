"""OVITO scene generation for reaction visualization."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from lammpalyze.analysis import LoadedSimulation
from lammpalyze.parsers import ReaxBond, TrajectoryAtom, TrajectoryFrame, read_lammpstrj_frame, read_reax_bonds_frame
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

    reactant_frame = read_lammpstrj_frame(simulation.trajectory_path, occurrence.timestep_reactants)
    product_frame = read_lammpstrj_frame(simulation.trajectory_path, occurrence.timestep_products)
    reactant_bonds = read_reax_bonds_frame(simulation.bond_path, occurrence.timestep_reactants)
    product_bonds = read_reax_bonds_frame(simulation.bond_path, occurrence.timestep_products)

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


def launch_ovito_scene(scene: OvitoScene, ovito_executable: str | None = None) -> subprocess.Popen:
    """Launch OVITO with the generated LAMMPS data file."""

    executable = ovito_executable or _find_ovito_executable()
    if executable is not None:
        return subprocess.Popen([executable, str(scene.data_file)])

    shell = os.environ.get("SHELL") or "/bin/bash"
    if Path(shell).exists():
        command = f"ovito {shlex.quote(str(scene.data_file))}"
        return subprocess.Popen([shell, "-ic", command])

    if os.name == "nt":  # pragma: no cover - Windows fallback.
        command = f"ovito {shlex.quote(str(scene.data_file))}"
        return subprocess.Popen(command, shell=True)

    raise FileNotFoundError(
        "Could not find the OVITO executable. Install OVITO, add it to PATH, "
        "or set OVITO_BIN to the executable path."
    )


def _find_ovito_executable() -> str | None:
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


def _visual_type_map(type_to_element: dict[int, str]) -> dict[int, str]:
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
    element = type_to_element.get(atom.atom_type)
    visual_type = 1
    diameter = OTHER_SPHERE_DIAMETER
    if is_reaction_atom and element is not None:
        visual_type = next(key for key, value in visual_type_map.items() if value == element)
        diameter = _element_diameter(element)
    return atom_id, visual_type, diameter, atom.x + x_shift, atom.y, atom.z


def _visual_bonds(bonds: list[ReaxBond], atom_id_offset: int, start_bond_id: int) -> list[tuple[int, int, int]]:
    return [
        (bond_index, bond.atom_i + atom_id_offset, bond.atom_j + atom_id_offset)
        for bond_index, bond in enumerate(bonds, start=start_bond_id)
    ]


def _bounds_from_visual_atoms(
    visual_atoms: list[tuple[int, int, float, float, float, float]],
    padding: float = 4.0,
) -> tuple[list[float], list[float], list[float]]:
    x_values = [atom[3] for atom in visual_atoms]
    y_values = [atom[4] for atom in visual_atoms]
    z_values = [atom[5] for atom in visual_atoms]
    return (
        [min(x_values) - padding, max(x_values) + padding],
        [min(y_values) - padding, max(y_values) + padding],
        [min(z_values) - padding, max(z_values) + padding],
    )


def _element_diameter(element: str) -> float:
    radius = CPK_RADII.get(element)
    if radius is None:
        return DEFAULT_SPHERE_DIAMETER
    return radius * SPHERE_DIAMETER_SCALE


def _write_scene_info(info_file: Path, occurrence: ReactionOccurrence) -> None:
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
