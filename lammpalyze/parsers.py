"""Parsers for LAMMPS/ReaxFF output files."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

try:  # RDKit is only needed for bond/SMILES parsing.
    from rdkit import Chem
    from rdkit.Chem import Descriptors
except ImportError:  # pragma: no cover - depends on optional external package.
    Chem = None
    Descriptors = None


@dataclass(frozen=True)
class TrajectoryAtom:
    """One atom entry from a LAMMPS trajectory frame."""

    atom_id: int
    atom_type: int
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class TrajectoryFrame:
    """A LAMMPS trajectory frame with bounds and atom positions."""

    timestep: int
    bounds: np.ndarray
    atoms: list[TrajectoryAtom]


@dataclass(frozen=True)
class ReaxBond:
    """One bond from a ReaxFF bonds frame."""

    atom_i: int
    atom_j: int
    order: float


def eval_species(species_file: str | Path) -> tuple[list[str], dict[str, list[int]], pd.DataFrame]:
    """Parse a LAMMPS ``reaxff/species`` output file."""

    species_path = Path(species_file)
    species_set: set[str] = set()
    static_cols = {"Timestep", "No_Moles", "No_Specs"}

    with species_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith("#"):
                cols = line.lstrip("#").split()
                species_set.update(col for col in cols if col not in static_cols)

    species = sorted(species_set)
    species_dict: dict[str, list[int]] = {"Timestep": []}
    species_dict.update({name: [] for name in species})

    current_cols: list[str] | None = None
    with species_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                current_cols = line.lstrip("#").split()
                continue
            if current_cols is None:
                raise ValueError(f"Data row before header in species file: {species_path}")
            current_row = dict(zip(current_cols, line.split(), strict=False))
            species_dict["Timestep"].append(int(current_row["Timestep"]))
            for name in species:
                species_dict[name].append(int(current_row.get(name, 0)))

    return species, species_dict, pd.DataFrame(species_dict)


def eval_thermo(
    thermo_file: str | Path,
    indicator1: str = "Step",
    indicator2: str = "Loop",
) -> tuple[dict[str, list[float]], pd.DataFrame]:
    """Parse the thermo table from a LAMMPS log file."""

    thermo_path = Path(thermo_file)
    thermo_dict: dict[str, list[float]] = {}
    thermo_cols: list[str] = []
    in_table = False

    with thermo_path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue

            if stripped.startswith(indicator1):
                thermo_cols = stripped.split()
                thermo_dict = {col: [] for col in thermo_cols}
                in_table = True
                continue

            if stripped.startswith(indicator2):
                break

            if in_table:
                values = stripped.split()
                if len(values) < len(thermo_cols):
                    continue
                for value, col in zip(values, thermo_cols, strict=False):
                    thermo_dict[col].append(float(value))

    if not thermo_dict:
        raise ValueError(f"No thermo table starting with {indicator1!r} found in {thermo_path}")
    return thermo_dict, pd.DataFrame(thermo_dict)


def parse_bonds(
    bond_file: str | Path,
    type_to_element: dict[int, str],
) -> tuple[dict[str, list[str]], dict[int, list[str]], dict[int, list[list[str]]], dict[int, list[str]]]:
    """Parse a ReaxFF bonds file into SMILES and chemical formula data.

    Returns ``atom_evolution, smiles, smiles_atoms, chem_formulas``. The atom
    identifiers are kept as strings to match the historical script behavior.
    """

    _require_rdkit()
    bond_path = Path(bond_file)

    atoms: dict[str, str] = {}
    bonds: list[tuple[int, int, object]] = []
    atom_evolution: dict[str, list[str]] = defaultdict(list)
    smiles: dict[int, list[str]] = {}
    smiles_atoms: dict[int, list[list[str]]] = {}
    chem_formulas: dict[int, list[str]] = {}
    molecule_cache: dict[tuple[tuple[str, ...], tuple[tuple[int, int, str], ...]], tuple[str, str]] = {}
    counter = 0
    timestep: int | None = None
    n_atoms: int | None = None

    with bond_path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("#"):
                if "Timestep" in line:
                    timestep = int(line.split()[-1])
                elif "Number of particles" in line:
                    n_atoms = int(line.split()[-1])
                continue

            if timestep is None or n_atoms is None:
                raise ValueError(f"Bond data before timestep header in {bond_path}")

            parts = line.split()
            if len(parts) < 3:
                continue

            atom_id = int(parts[0])
            atom_type = int(parts[1])
            n_bonds = int(parts[2])
            atoms[parts[0]] = type_to_element[atom_type]

            bonded_atoms = [int(value) for value in parts[3 : 3 + n_bonds]]
            bond_orders = [float(value) for value in parts[4 + n_bonds : 4 + 2 * n_bonds]]
            for bonded_id, bond_order in zip(bonded_atoms, bond_orders, strict=False):
                if atom_id < bonded_id:
                    bonds.append((atom_id, bonded_id, bo_to_rdkit_bond(bond_order)))

            counter += 1
            if counter == n_atoms:
                _store_bond_frame(
                    timestep,
                    atoms,
                    bonds,
                    molecule_cache,
                    atom_evolution,
                    smiles,
                    smiles_atoms,
                    chem_formulas,
                )
                atoms = {}
                bonds = []
                counter = 0

    return atom_evolution, smiles, smiles_atoms, chem_formulas


def bo_to_rdkit_bond(bond_order: float):
    """Map a continuous ReaxFF bond order to an RDKit bond type."""

    _require_rdkit()
    if bond_order >= 2.5:
        return Chem.BondType.TRIPLE
    if bond_order >= 1.5:
        return Chem.BondType.DOUBLE
    return Chem.BondType.SINGLE


def first_appearance(values_by_time: dict[int, list[str]]) -> tuple[list[str], dict[str, list[int]]]:
    """Return unique values and the first ``[timestep, index]`` where each appears."""

    unique = set(chain.from_iterable(values_by_time.values()))
    remains = unique.copy()
    first: dict[str, list[int]] = {}

    for time, molecules in values_by_time.items():
        if not remains:
            break
        for index, molecule in enumerate(molecules):
            if molecule in remains:
                first[molecule] = [time, index]
                remains.discard(molecule)

    return sorted(unique), first


def parse_traj(filename: str | Path) -> Iterator[np.ndarray]:
    """Yield wrapped ``[q, x, y, z]`` atom arrays from a LAMMPS trajectory."""

    with Path(filename).open(encoding="utf-8") as handle:
        n_atoms = 0
        min_coords = np.zeros(3)
        box_lengths = np.ones(3)
        while True:
            line = handle.readline()
            if not line:
                break

            if "NUMBER" in line:
                n_atoms = int(handle.readline())

            if "ITEM: BOX BOUNDS" in line:
                bounds = np.array([[float(x) for x in handle.readline().split()] for _ in range(3)])
                min_coords = bounds[:, 0]
                box_lengths = bounds[:, 1] - bounds[:, 0]

            if "ITEM: ATOMS" in line:
                cols = line.split()[2:]
                frame_data = []
                for _ in range(n_atoms):
                    atom_line = handle.readline().split()
                    frame_data.append(
                        [
                            float(atom_line[cols.index("q")]),
                            float(atom_line[cols.index("xu")]),
                            float(atom_line[cols.index("yu")]),
                            float(atom_line[cols.index("zu")]),
                        ]
                    )
                unwrapped = np.array(frame_data)
                wrapped = unwrapped.copy()
                wrapped[:, 1:] = min_coords + (unwrapped[:, 1:] - min_coords) % box_lengths
                yield wrapped


def read_lammpstrj_frame(filename: str | Path, target_timestep: int) -> TrajectoryFrame:
    """Read one trajectory frame by timestep for external visualization."""

    with Path(filename).open(encoding="utf-8") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue

            timestep = int(handle.readline().strip())
            number_header = handle.readline()
            if not number_header.startswith("ITEM: NUMBER OF ATOMS"):
                raise ValueError(f"Malformed trajectory frame at timestep {timestep} in {filename}")
            n_atoms = int(handle.readline().strip())

            bounds_header = handle.readline()
            if not bounds_header.startswith("ITEM: BOX BOUNDS"):
                raise ValueError(f"Missing box bounds at timestep {timestep} in {filename}")
            bounds = np.array([[float(value) for value in handle.readline().split()[:2]] for _ in range(3)])

            atoms_header = handle.readline()
            if not atoms_header.startswith("ITEM: ATOMS"):
                raise ValueError(f"Missing atom table at timestep {timestep} in {filename}")
            columns = atoms_header.split()[2:]

            atoms = []
            for _ in range(n_atoms):
                values = handle.readline().split()
                if timestep == target_timestep:
                    atoms.append(_trajectory_atom_from_values(columns, values))

            if timestep == target_timestep:
                return TrajectoryFrame(timestep=timestep, bounds=bounds, atoms=atoms)

    raise ValueError(f"Timestep {target_timestep} not found in trajectory file {filename}")


def read_reax_bonds_frame(filename: str | Path, target_timestep: int) -> list[ReaxBond]:
    """Read bonds from one ReaxFF bond-file frame."""

    bonds: list[ReaxBond] = []
    in_target = False
    n_atoms: int | None = None
    rows_read = 0

    with Path(filename).open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("#"):
                if "Timestep" in line:
                    if in_target:
                        return bonds
                    timestep = int(line.split()[-1])
                    in_target = timestep == target_timestep
                    n_atoms = None
                    rows_read = 0
                    bonds = []
                elif in_target and "Number of particles" in line:
                    n_atoms = int(line.split()[-1])
                continue

            if not in_target:
                continue

            parts = line.split()
            if len(parts) < 3:
                continue

            atom_i = int(parts[0])
            n_bonds = int(parts[2])
            bonded_atoms = [int(value) for value in parts[3 : 3 + n_bonds]]
            bond_orders = [float(value) for value in parts[4 + n_bonds : 4 + 2 * n_bonds]]
            for atom_j, bond_order in zip(bonded_atoms, bond_orders, strict=False):
                if atom_i < atom_j:
                    bonds.append(ReaxBond(atom_i=atom_i, atom_j=atom_j, order=bond_order))

            rows_read += 1
            if n_atoms is not None and rows_read >= n_atoms:
                return bonds

    if in_target:
        return bonds
    raise ValueError(f"Timestep {target_timestep} not found in ReaxFF bond file {filename}")


def map_atoms_to_mols(smiles_list: list[str], ids_list: list[list[str]]) -> dict[str, tuple[str, int]]:
    """Map each atom id to its molecule SMILES and molecule index."""

    atom_to_mol: dict[str, tuple[str, int]] = {}
    for index, atom_ids in enumerate(ids_list):
        for atom_id in atom_ids:
            atom_to_mol[atom_id] = (smiles_list[index], index)
    return atom_to_mol


def _trajectory_atom_from_values(columns: list[str], values: list[str]) -> TrajectoryAtom:
    column_index = {column: index for index, column in enumerate(columns)}
    x_column = _first_available_column(column_index, ("xu", "x", "xs"))
    y_column = _first_available_column(column_index, ("yu", "y", "ys"))
    z_column = _first_available_column(column_index, ("zu", "z", "zs"))
    return TrajectoryAtom(
        atom_id=int(float(values[column_index["id"]])),
        atom_type=int(float(values[column_index.get("type", column_index["id"])])),
        x=float(values[column_index[x_column]]),
        y=float(values[column_index[y_column]]),
        z=float(values[column_index[z_column]]),
    )


def _first_available_column(column_index: dict[str, int], candidates: tuple[str, ...]) -> str:
    for column in candidates:
        if column in column_index:
            return column
    raise ValueError(f"Trajectory atom table lacks coordinate columns {candidates}")


def _store_bond_frame(
    timestep: int,
    atoms: dict[str, str],
    bonds: list[tuple[int, int, object]],
    molecule_cache: dict[tuple[tuple[str, ...], tuple[tuple[int, int, str], ...]], tuple[str, str]],
    atom_evolution: dict[str, list[str]],
    smiles: dict[int, list[str]],
    smiles_atoms: dict[int, list[list[str]]],
    chem_formulas: dict[int, list[str]],
) -> None:
    components = _bond_components(atoms, bonds)
    smiles_list: list[str] = []
    formula_list: list[str] = []
    mol_lmp_ids: list[list[str]] = []

    for component_ids in components:
        component_bonds = [
            (atom_i, atom_j, bond_type)
            for atom_i, atom_j, bond_type in bonds
            if str(atom_i) in component_ids and str(atom_j) in component_ids
        ]
        signature = _component_signature(component_ids, atoms, component_bonds)
        if signature not in molecule_cache:
            molecule_cache[signature] = _component_smiles_and_formula(component_ids, atoms, component_bonds)
        molecule_smiles, formula = molecule_cache[signature]
        smiles_list.append(molecule_smiles)
        formula_list.append(formula)
        mol_lmp_ids.append(component_ids)

    smiles[timestep] = smiles_list
    smiles_atoms[timestep] = mol_lmp_ids
    chem_formulas[timestep] = formula_list

    for index, fragment_ids in enumerate(mol_lmp_ids):
        for atom_id in fragment_ids:
            atom_evolution[atom_id].append(smiles_list[index])


class _FrameUnionFind:
    def __init__(self) -> None:
        self.root: dict[str, str] = {}

    def add(self, value: str) -> None:
        if value not in self.root:
            self.root[value] = value

    def find(self, value: str) -> str:
        if self.root[value] != value:
            self.root[value] = self.find(self.root[value])
        return self.root[value]

    def union(self, value1: str, value2: str) -> None:
        root1 = self.find(value1)
        root2 = self.find(value2)
        if root1 != root2:
            self.root[root2] = root1


def _bond_components(
    atoms: dict[str, str],
    bonds: list[tuple[int, int, object]],
) -> list[list[str]]:
    union_find = _FrameUnionFind()
    for atom_id in atoms:
        union_find.add(atom_id)
    for atom_i, atom_j, _bond_type in bonds:
        union_find.union(str(atom_i), str(atom_j))

    grouped: dict[str, list[str]] = defaultdict(list)
    for atom_id in atoms:
        grouped[union_find.find(atom_id)].append(atom_id)
    return [sorted(ids, key=int) for ids in sorted(grouped.values(), key=lambda values: min(map(int, values)))]


def _component_signature(
    component_ids: list[str],
    atoms: dict[str, str],
    component_bonds: list[tuple[int, int, object]],
) -> tuple[tuple[str, ...], tuple[tuple[int, int, str], ...]]:
    local_index = {atom_id: index for index, atom_id in enumerate(component_ids)}
    elements = tuple(atoms[atom_id] for atom_id in component_ids)
    bonds = tuple(
        sorted(
            (
                min(local_index[str(atom_i)], local_index[str(atom_j)]),
                max(local_index[str(atom_i)], local_index[str(atom_j)]),
                str(bond_type),
            )
            for atom_i, atom_j, bond_type in component_bonds
        )
    )
    return elements, bonds


def _component_smiles_and_formula(
    component_ids: list[str],
    atoms: dict[str, str],
    component_bonds: list[tuple[int, int, object]],
) -> tuple[str, str]:
    local_index = {atom_id: index for index, atom_id in enumerate(component_ids)}
    mol = Chem.RWMol()

    for atom_id in component_ids:
        rd_atom = Chem.Atom(atoms[atom_id])
        rd_atom.SetNoImplicit(True)
        mol.AddAtom(rd_atom)

    for atom_i, atom_j, bond_type in component_bonds:
        mol.AddBond(local_index[str(atom_i)], local_index[str(atom_j)], bond_type)

    rd_mol = mol.GetMol()
    formula = Descriptors.rdMolDescriptors.CalcMolFormula(rd_mol)
    rd_mol = Chem.AddHs(rd_mol)
    return Chem.MolToSmiles(rd_mol, allHsExplicit=True), formula


def _require_rdkit() -> None:
    if Chem is None or Descriptors is None:
        raise ImportError(
            "RDKit is required for bond parsing and SMILES visualization. "
            "Install it with conda-forge rdkit or pip install rdkit."
        )
