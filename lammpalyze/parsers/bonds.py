"""Parsers and molecule conversion helpers for ReaxFF bond files."""

from __future__ import annotations

from collections import defaultdict
from itertools import chain
from pathlib import Path

from lammpalyze.parsers.models import ReaxBond

try:  # RDKit is only needed for bond/SMILES parsing.
    from rdkit import Chem
    from rdkit.Chem import Descriptors
except ImportError:  # pragma: no cover - depends on optional external package.
    Chem = None
    Descriptors = None


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

            bonded_atoms = [int(value) for value in parts[3: 3 + n_bonds]]
            bond_orders = [float(value) for value in parts[4 + n_bonds: 4 + 2 * n_bonds]]
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
            bonded_atoms = [int(value) for value in parts[3: 3 + n_bonds]]
            bond_orders = [float(value) for value in parts[4 + n_bonds: 4 + 2 * n_bonds]]
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
    """Convert one completed ReaxFF bond frame into molecule records."""

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
    """Union-find helper for molecule fragments within one bond frame."""

    def __init__(self) -> None:
        """Initialize an empty disjoint-set forest for atom ids."""

        self.root: dict[str, str] = {}

    def add(self, value: str) -> None:
        """Register a value as its own set if it is not already known."""

        if value not in self.root:
            self.root[value] = value

    def find(self, value: str) -> str:
        """Return the representative root for ``value``."""

        if self.root[value] != value:
            self.root[value] = self.find(self.root[value])
        return self.root[value]

    def union(self, value1: str, value2: str) -> None:
        """Join two values into the same set."""

        root1 = self.find(value1)
        root2 = self.find(value2)
        if root1 != root2:
            self.root[root2] = root1


def _bond_components(
    atoms: dict[str, str],
    bonds: list[tuple[int, int, object]],
) -> list[list[str]]:
    """Return atom-id components connected by bonds within one frame."""

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
    """Return a cache key describing a molecule component topology."""

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
    """Build RDKit SMILES and formula values for one molecule component."""

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
    """Raise an informative error when RDKit bond parsing support is missing."""

    if Chem is None or Descriptors is None:
        raise ImportError(
            "RDKit is required for bond parsing and SMILES visualization. "
            "Install it with conda-forge rdkit or pip install rdkit."
        )
