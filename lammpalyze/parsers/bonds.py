"""Parsers and molecule conversion helpers for ReaxFF bond files."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from statistics import fmean, pstdev

from lammpalyze.parsers.models import (
    BondParseResult,
    ChargeStatistics,
    ComponentProperties,
    ReaxBond,
)

try:  # RDKit is only needed for bond/SMILES parsing.
    from rdkit import Chem
    from rdkit.Chem import Descriptors
except ImportError:  # pragma: no cover - depends on optional external package.
    Chem = None
    Descriptors = None


STRICT_VALENCE_ELEMENTS = {
    "H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I",
}


def parse_bonds(
    bond_file: str | Path,
    type_to_element: dict[int, str],
    *,
    default_bond_order_cutoff: float = 0.5,
    bond_order_cutoffs: Mapping[tuple[int, int], float] | None = None,
    bond_state_persistence_frames: int = 1,
    bond_state_persistence_timesteps: int = 0,
    bond_order_hysteresis: float = 0.0,
) -> tuple[dict[str, list[str]], dict[int, list[str]], dict[int, list[list[str]]], dict[int, list[str]]]:
    """Parse a ReaxFF bonds file into SMILES and chemical formula data.

    Returns ``atom_evolution, smiles, smiles_atoms, chem_formulas``. The atom
    identifiers are kept as strings to match the historical script behavior.
    """

    result = parse_bond_observations(
        bond_file,
        type_to_element,
        default_bond_order_cutoff=default_bond_order_cutoff,
        bond_order_cutoffs=bond_order_cutoffs,
        bond_state_persistence_frames=bond_state_persistence_frames,
        bond_state_persistence_timesteps=bond_state_persistence_timesteps,
        bond_order_hysteresis=bond_order_hysteresis,
        structure_quality_mode="keep",
    )
    return result.atom_evolution, result.smiles, result.smiles_atoms, result.chem_formulas


@dataclass(frozen=True)
class _RawBondFrame:
    """Atom and continuous bond-order values read from one ReaxFF frame."""

    timestep: int
    atoms: dict[str, str]
    atom_types: dict[str, int]
    bond_orders: dict[tuple[int, int], float]
    charges: dict[str, float]


@dataclass
class _BondState:
    """Accepted and pending discrete state for one temporally filtered bond."""

    stable_order: int = 0
    candidate_order: int | None = None
    candidate_frames: int = 0
    candidate_start_timestep: int | None = None
    candidate_start_frame_index: int | None = None


def parse_bond_observations(
    bond_file: str | Path,
    type_to_element: dict[int, str],
    *,
    default_bond_order_cutoff: float = 0.5,
    bond_order_cutoffs: Mapping[tuple[int, int], float] | None = None,
    bond_state_persistence_frames: int = 1,
    bond_state_persistence_timesteps: int = 0,
    bond_order_hysteresis: float = 0.0,
    structure_quality_mode: str = "flag",
    ion_charge_threshold: float = 0.5,
) -> BondParseResult:
    """Parse ReaxFF frames with temporal, charge, and structure metadata."""

    _require_rdkit()
    _validate_bond_analysis_options(
        bond_state_persistence_frames,
        bond_state_persistence_timesteps,
        bond_order_hysteresis,
        structure_quality_mode,
        ion_charge_threshold,
    )
    pair_cutoffs = bond_order_cutoffs or {}
    atom_evolution: dict[str, list[str]] = defaultdict(list)
    smiles: dict[int, list[str]] = {}
    smiles_atoms: dict[int, list[list[str]]] = {}
    chem_formulas: dict[int, list[str]] = {}
    atom_charges: dict[int, dict[str, float]] = {}
    charge_statistics: dict[int, dict[str, ChargeStatistics]] = {}
    component_properties: dict[int, list[ComponentProperties]] = {}
    excluded_components: dict[int, set[int]] = {}
    molecule_cache: dict[
        tuple[tuple[str, ...], tuple[tuple[int, int, str], ...]],
        tuple[str, str, tuple[str, ...]],
    ] = {}
    bond_states: dict[tuple[int, int], _BondState] = {}

    filtered_frames = _iter_temporally_filtered_bond_frames(
        _iter_raw_bond_frames(bond_file, type_to_element),
        bond_states,
        default_bond_order_cutoff,
        pair_cutoffs,
        bond_state_persistence_frames,
        bond_state_persistence_timesteps,
        bond_order_hysteresis,
    )
    for frame, bond_orders in filtered_frames:
        bonds = [
            (atom_i, atom_j, _rdkit_bond_from_order(order))
            for (atom_i, atom_j), order in sorted(bond_orders.items())
        ]
        _store_bond_frame(
            frame.timestep,
            frame.atoms,
            bonds,
            frame.charges,
            molecule_cache,
            atom_evolution,
            smiles,
            smiles_atoms,
            chem_formulas,
            atom_charges,
            charge_statistics,
            component_properties,
            excluded_components,
            structure_quality_mode,
            ion_charge_threshold,
        )

    return BondParseResult(
        atom_evolution=dict(atom_evolution),
        smiles=smiles,
        smiles_atoms=smiles_atoms,
        chem_formulas=chem_formulas,
        atom_charges=atom_charges,
        charge_statistics=charge_statistics,
        component_properties=component_properties,
        excluded_components=excluded_components,
    )


def _temporally_filtered_bond_order_frames(
    frames: list[_RawBondFrame],
    states: dict[tuple[int, int], _BondState],
    default_cutoff: float,
    pair_cutoffs: Mapping[tuple[int, int], float],
    persistence_frames: int,
    persistence_timesteps: int,
    hysteresis: float,
) -> list[dict[tuple[int, int], int]]:
    """Return finalized per-frame bond orders with accepted changes backdated."""

    return [
        bond_orders
        for _frame, bond_orders in _iter_temporally_filtered_bond_frames(
            frames,
            states,
            default_cutoff,
            pair_cutoffs,
            persistence_frames,
            persistence_timesteps,
            hysteresis,
        )
    ]


def _iter_temporally_filtered_bond_frames(
    frames,
    states: dict[tuple[int, int], _BondState],
    default_cutoff: float,
    pair_cutoffs: Mapping[tuple[int, int], float],
    persistence_frames: int,
    persistence_timesteps: int,
    hysteresis: float,
):
    """Yield finalized frames as soon as no pending change can backdate them."""

    buffered_frames = deque()
    for frame_index, frame in enumerate(frames):
        frame_bond_orders: dict[tuple[int, int], int] = {}
        observed_pairs = set(frame.bond_orders)
        for atom_pair in sorted(observed_pairs | set(states)):
            atom_i, atom_j = atom_pair
            if str(atom_i) not in frame.atom_types or str(atom_j) not in frame.atom_types:
                continue
            type_pair = tuple(sorted((frame.atom_types[str(atom_i)], frame.atom_types[str(atom_j)])))
            cutoff = pair_cutoffs.get(type_pair, default_cutoff)
            bond_order = frame.bond_orders.get(atom_pair, 0.0)
            state = states.setdefault(atom_pair, _BondState())

            if frame_index == 0:
                state.stable_order = _discrete_bond_order(bond_order) if bond_order >= cutoff else 0
                _clear_candidate_state(state)
            else:
                desired_order = _desired_bond_state(state.stable_order, bond_order, cutoff, hysteresis)
                accepted = _update_bond_state(
                    state,
                    desired_order,
                    frame.timestep,
                    frame_index,
                    persistence_frames,
                    persistence_timesteps,
                )
                if accepted is not None:
                    start_frame_index, accepted_order = accepted
                    _backdate_buffered_bond_order(
                        buffered_frames,
                        atom_pair,
                        start_frame_index,
                        accepted_order,
                    )

            if state.stable_order:
                frame_bond_orders[atom_pair] = state.stable_order
        buffered_frames.append((frame_index, frame, frame_bond_orders))

        pending_starts = [
            state.candidate_start_frame_index
            for state in states.values()
            if state.candidate_start_frame_index is not None
        ]
        safe_before = min(pending_starts) if pending_starts else frame_index + 1
        while buffered_frames and buffered_frames[0][0] < safe_before:
            _index, completed_frame, completed_orders = buffered_frames.popleft()
            yield completed_frame, completed_orders

    while buffered_frames:
        _index, completed_frame, completed_orders = buffered_frames.popleft()
        yield completed_frame, completed_orders


def _backdate_buffered_bond_order(
    buffered_frames,
    atom_pair: tuple[int, int],
    start_frame_index: int,
    accepted_order: int,
) -> None:
    """Apply an accepted state to buffered candidate frames."""

    for frame_index, _frame, frame_bond_orders in buffered_frames:
        if frame_index < start_frame_index:
            continue
        if accepted_order:
            frame_bond_orders[atom_pair] = accepted_order
        else:
            frame_bond_orders.pop(atom_pair, None)


def _iter_raw_bond_frames(
    bond_file: str | Path,
    type_to_element: dict[int, str],
):
    """Yield complete raw ReaxFF frames without applying connectivity cutoffs."""

    bond_path = Path(bond_file)
    atoms: dict[str, str] = {}
    atom_types: dict[str, int] = {}
    raw_bonds: dict[tuple[int, int], float] = {}
    charges: dict[str, float] = {}
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
            atom_types[parts[0]] = atom_type
            charges[parts[0]] = float(parts[-1])

            bonded_atoms = [int(value) for value in parts[3: 3 + n_bonds]]
            bond_orders = [float(value) for value in parts[4 + n_bonds: 4 + 2 * n_bonds]]
            for bonded_id, bond_order in zip(bonded_atoms, bond_orders, strict=False):
                if atom_id < bonded_id:
                    raw_bonds[(atom_id, bonded_id)] = bond_order

            counter += 1
            if counter == n_atoms:
                yield _RawBondFrame(
                    timestep=timestep,
                    atoms=atoms,
                    atom_types=atom_types,
                    bond_orders=raw_bonds,
                    charges=charges,
                )
                atoms = {}
                atom_types = {}
                raw_bonds = {}
                charges = {}
                counter = 0


def _validate_bond_analysis_options(
    persistence_frames: int,
    persistence_timesteps: int,
    hysteresis: float,
    quality_mode: str,
    ion_charge_threshold: float,
) -> None:
    """Validate temporal filtering and quality-analysis options."""

    if persistence_frames < 1:
        raise ValueError("bond_state_persistence_frames must be at least 1.")
    if persistence_timesteps < 0:
        raise ValueError("bond_state_persistence_timesteps must not be negative.")
    if hysteresis < 0:
        raise ValueError("bond_order_hysteresis must not be negative.")
    if quality_mode not in {"keep", "flag", "exclude", "skip"}:
        raise ValueError("structure_quality_mode must be 'keep', 'flag', 'exclude', or 'skip'.")
    if ion_charge_threshold < 0:
        raise ValueError("ion_charge_threshold must not be negative.")


def _desired_bond_state(stable_order: int, bond_order: float, cutoff: float, hysteresis: float) -> int:
    """Return the instantaneous discrete state using connectivity hysteresis."""

    if stable_order:
        if bond_order < max(0.0, cutoff - hysteresis):
            return 0
        return _discrete_bond_order(bond_order)
    if bond_order >= cutoff + hysteresis:
        return _discrete_bond_order(bond_order)
    return 0


def _update_bond_state(
    state: _BondState,
    desired_order: int,
    timestep: int,
    frame_index: int,
    persistence_frames: int,
    persistence_timesteps: int,
) -> tuple[int, int] | None:
    """Accept a new discrete state after it persists long enough."""

    if desired_order == state.stable_order:
        _clear_candidate_state(state)
        return None
    if state.candidate_order != desired_order:
        state.candidate_order = desired_order
        state.candidate_frames = 1
        state.candidate_start_timestep = timestep
        state.candidate_start_frame_index = frame_index
    else:
        state.candidate_frames += 1

    start = state.candidate_start_timestep
    duration = timestep - start if start is not None else 0
    if state.candidate_frames >= persistence_frames and duration >= persistence_timesteps:
        start_frame_index = state.candidate_start_frame_index
        state.stable_order = desired_order
        _clear_candidate_state(state)
        return (frame_index if start_frame_index is None else start_frame_index, desired_order)
    return None


def _clear_candidate_state(state: _BondState) -> None:
    """Discard a pending bond-state transition."""

    state.candidate_order = None
    state.candidate_frames = 0
    state.candidate_start_timestep = None
    state.candidate_start_frame_index = None


def _discrete_bond_order(bond_order: float) -> int:
    """Map a continuous ReaxFF order to integer single, double, or triple."""

    if bond_order >= 2.5:
        return 3
    if bond_order >= 1.5:
        return 2
    return 1


def _rdkit_bond_from_order(bond_order: int):
    """Return the RDKit bond type for a positive integer bond order."""

    return {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
    }[bond_order]


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


def index_reax_bond_frames(filename: str | Path) -> dict[int, int]:
    """Return byte offsets for every ReaxFF bond timestep."""

    frame_offsets = {}
    line_offset = 0
    with Path(filename).open("rb") as handle:
        for raw_line in handle:
            if raw_line.startswith(b"#") and b"Timestep" in raw_line:
                frame_offsets[int(raw_line.split()[-1])] = line_offset
            line_offset += len(raw_line)
    return frame_offsets


def read_reax_bonds_frame(
    filename: str | Path,
    target_timestep: int,
    frame_offset: int | None = None,
) -> list[ReaxBond]:
    """Read bonds from one ReaxFF bond-file frame."""

    bonds: list[ReaxBond] = []
    in_target = False
    n_atoms: int | None = None
    rows_read = 0

    with Path(filename).open(encoding="utf-8") as handle:
        if frame_offset is not None:
            handle.seek(frame_offset)
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("#"):
                if "Timestep" in line:
                    if in_target:
                        return bonds
                    timestep = int(line.split()[-1])
                    if frame_offset is not None and timestep != target_timestep:
                        break
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
    charges: dict[str, float],
    molecule_cache: dict[
        tuple[tuple[str, ...], tuple[tuple[int, int, str], ...]],
        tuple[str, str, tuple[str, ...]],
    ],
    atom_evolution: dict[str, list[str]],
    smiles: dict[int, list[str]],
    smiles_atoms: dict[int, list[list[str]]],
    chem_formulas: dict[int, list[str]],
    atom_charges: dict[int, dict[str, float]],
    charge_statistics: dict[int, dict[str, ChargeStatistics]],
    component_properties: dict[int, list[ComponentProperties]],
    excluded_components: dict[int, set[int]],
    quality_mode: str,
    ion_charge_threshold: float,
) -> None:
    """Convert one completed ReaxFF bond frame into molecule records."""

    components = _bond_components(atoms, bonds)
    smiles_list: list[str] = []
    formula_list: list[str] = []
    mol_lmp_ids: list[list[str]] = []
    property_list: list[ComponentProperties] = []
    excluded_indexes: set[int] = set()

    component_by_atom = {
        atom_id: component_index
        for component_index, component_ids in enumerate(components)
        for atom_id in component_ids
    }
    bonds_by_component: list[list[tuple[int, int, object]]] = [
        [] for _component in components
    ]
    for atom_i, atom_j, bond_type in bonds:
        component_index = component_by_atom[str(atom_i)]
        bonds_by_component[component_index].append((atom_i, atom_j, bond_type))

    for component_index, component_ids in enumerate(components):
        component_bonds = bonds_by_component[component_index]
        signature = _component_signature(component_ids, atoms, component_bonds)
        if signature not in molecule_cache:
            molecule_cache[signature] = _component_smiles_and_formula(component_ids, atoms, component_bonds)
        molecule_smiles, formula, warnings = molecule_cache[signature]
        component_charge = sum(charges.get(atom_id, 0.0) for atom_id in component_ids)
        properties = ComponentProperties(
            charge=component_charge,
            ion_candidate=_ion_candidate(component_charge, ion_charge_threshold),
            warnings=warnings,
        )
        smiles_list.append(molecule_smiles)
        formula_list.append(formula)
        mol_lmp_ids.append(component_ids)
        property_list.append(properties)
        if quality_mode in {"exclude", "skip"} and properties.suspicious:
            excluded_indexes.add(component_index)

    smiles[timestep] = smiles_list
    smiles_atoms[timestep] = mol_lmp_ids
    chem_formulas[timestep] = formula_list
    atom_charges[timestep] = dict(charges)
    charge_statistics[timestep] = _frame_charge_statistics(atoms, charges)
    component_properties[timestep] = property_list
    excluded_components[timestep] = excluded_indexes

    for index, fragment_ids in enumerate(mol_lmp_ids):
        for atom_id in fragment_ids:
            atom_evolution[atom_id].append(smiles_list[index])


def _frame_charge_statistics(
    atoms: dict[str, str],
    charges: dict[str, float],
) -> dict[str, ChargeStatistics]:
    """Calculate per-element mean and population deviation for one frame."""

    charges_by_element: dict[str, list[float]] = defaultdict(list)
    for atom_id, element in atoms.items():
        if atom_id in charges:
            charges_by_element[element].append(charges[atom_id])
    return {
        element: ChargeStatistics(
            mean=fmean(values),
            std=pstdev(values) if len(values) > 1 else 0.0,
            count=len(values),
        )
        for element, values in sorted(charges_by_element.items())
    }


def _ion_candidate(component_charge: float, threshold: float) -> str | None:
    """Classify a component by its continuous total partial charge."""

    if threshold <= 0:
        return None
    if component_charge >= threshold:
        return "cation candidate"
    if component_charge <= -threshold:
        return "anion candidate"
    return None


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
) -> tuple[str, str, tuple[str, ...]]:
    """Build RDKit SMILES, formula, and quality flags for one component."""

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
    warnings = _component_quality_warnings(rd_mol, component_ids, atoms)
    smiles_mol = Chem.AddHs(rd_mol)
    return Chem.MolToSmiles(smiles_mol, allHsExplicit=True), formula, warnings


def _component_quality_warnings(
    rd_mol,
    component_ids: list[str],
    atoms: dict[str, str],
) -> tuple[str, ...]:
    """Return conservative valence and RDKit sanitization warnings."""

    warnings = []
    periodic_table = Chem.GetPeriodicTable()
    for atom_index, atom_id in enumerate(component_ids):
        if atoms[atom_id] not in STRICT_VALENCE_ELEMENTS:
            continue
        atom = rd_mol.GetAtomWithIdx(atom_index)
        allowed = [value for value in periodic_table.GetValenceList(atom.GetAtomicNum()) if value >= 0]
        if not allowed:
            continue
        bond_valence = sum(bond.GetBondTypeAsDouble() for bond in atom.GetBonds())
        if bond_valence > max(allowed) + 1e-8:
            warnings.append(
                f"component atom {atom_index + 1} {atoms[atom_id]} bond valence {bond_valence:g} "
                f"exceeds supported valence {max(allowed)}"
            )

    all_atoms_support_strict_validation = all(
        atoms[atom_id] in STRICT_VALENCE_ELEMENTS for atom_id in component_ids
    )
    if not warnings and all_atoms_support_strict_validation:
        probe = Chem.Mol(rd_mol)
        probe.UpdatePropertyCache(strict=False)
        try:
            failure = Chem.SanitizeMol(probe, catchErrors=True)
        except Exception as exc:  # pragma: no cover - defensive RDKit boundary.
            warnings.append(f"RDKit sanitization raised {type(exc).__name__}")
        else:
            if failure != Chem.SanitizeFlags.SANITIZE_NONE:
                warnings.append(f"RDKit sanitization failed at {failure}")
    return tuple(warnings)


def _require_rdkit() -> None:
    """Raise an informative error when RDKit bond parsing support is missing."""

    if Chem is None or Descriptors is None:
        raise ImportError(
            "RDKit is required for bond parsing and SMILES visualization. "
            "Install it with conda-forge rdkit or pip install rdkit."
        )
