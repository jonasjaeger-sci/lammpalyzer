"""Read and validate the small ``lmplyz.inp`` project files."""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass, field
from pathlib import Path


TOPIC_PREFIXES = {
    "bond": ("BF", "BondF", "BondFile"),
    "species": ("SF", "SpeciesF", "SpeciesFile"),
    "thermo": ("ThermoF", "TF", "ThermoFile"),
    "trajectory": ("TrajF", "TrajectoryF", "TrajectoryFile"),
}
DEFAULT_BOND_ORDER_CUTOFF = 0.3
DEFAULT_BOND_STATE_PERSISTENCE_FRAMES = 1
DEFAULT_BOND_STATE_PERSISTENCE_TIMESTEPS = 0
DEFAULT_BOND_ORDER_HYSTERESIS = 0.0
DEFAULT_STRUCTURE_QUALITY_MODE = "flag"
DEFAULT_ION_CHARGE_THRESHOLD = 0.5
STRUCTURE_QUALITY_MODES = {"keep", "flag", "exclude"}
_BOND_ORDER_CUTOFF_HEADER_RE = re.compile(
    r"^\s*#?\s*bond[\s_-]+order[\s_-]+cutoffs?\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SimulationFiles:
    """File paths collected for one numbered simulation entry."""

    index: int
    bond: Path | None = None
    species: Path | None = None
    thermo: Path | None = None
    trajectory: Path | None = None


@dataclass(frozen=True)
class LammpalyzeConfig:
    """Parsed input file plus the simulation groups discovered inside it."""

    input_file: Path
    element_list: list[str]
    simulations: list[SimulationFiles]
    default_bond_order_cutoff: float = DEFAULT_BOND_ORDER_CUTOFF
    bond_order_cutoffs: dict[tuple[int, int], float] = field(default_factory=dict)
    bond_state_persistence_frames: int = DEFAULT_BOND_STATE_PERSISTENCE_FRAMES
    bond_state_persistence_timesteps: int = DEFAULT_BOND_STATE_PERSISTENCE_TIMESTEPS
    bond_order_hysteresis: float = DEFAULT_BOND_ORDER_HYSTERESIS
    structure_quality_mode: str = DEFAULT_STRUCTURE_QUALITY_MODE
    ion_charge_threshold: float = DEFAULT_ION_CHARGE_THRESHOLD

    @property
    def type_to_element(self) -> dict[int, str]:
        """Map LAMMPS atom type numbers onto element symbols."""

        return {idx + 1: element for idx, element in enumerate(self.element_list)}

    def bond_order_cutoff(self, atom_type_a: int, atom_type_b: int) -> float:
        """Return the configured cutoff for an unordered atom-type pair."""

        pair = tuple(sorted((atom_type_a, atom_type_b)))
        return self.bond_order_cutoffs.get(pair, self.default_bond_order_cutoff)


def parse_input_file(input_file: str | Path) -> LammpalyzeConfig:
    """Parse ``lmplyz.inp`` into paths grouped by simulation index.

    The parser accepts the current ``lmplyz.inp`` style, for example
    ``BF1 = bonds_R1.reax`` and ``element_list = ["C", "H"]``. Relative paths
    are resolved relative to the input file directory.
    """

    path = Path(input_file).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    assignments = _read_assignments(path)
    element_list = _parse_element_list(assignments)
    default_bond_order_cutoff, bond_order_cutoffs = _parse_bond_order_cutoffs(
        path,
        len(element_list),
    )
    grouped = _group_paths(assignments, path.parent)
    persistence_frames = _parse_int_setting(
        assignments,
        "bond_state_persistence_frames",
        DEFAULT_BOND_STATE_PERSISTENCE_FRAMES,
        minimum=1,
    )
    persistence_timesteps = _parse_int_setting(
        assignments,
        "bond_state_persistence_timesteps",
        DEFAULT_BOND_STATE_PERSISTENCE_TIMESTEPS,
        minimum=0,
    )
    hysteresis = _parse_float_setting(
        assignments,
        "bond_order_hysteresis",
        DEFAULT_BOND_ORDER_HYSTERESIS,
        minimum=0.0,
    )
    quality_mode = _parse_choice_setting(
        assignments,
        "structure_quality_mode",
        DEFAULT_STRUCTURE_QUALITY_MODE,
        STRUCTURE_QUALITY_MODES,
    )
    ion_charge_threshold = _parse_float_setting(
        assignments,
        "ion_charge_threshold",
        DEFAULT_ION_CHARGE_THRESHOLD,
        minimum=0.0,
    )

    indexes = sorted({idx for topic in grouped.values() for idx in topic})
    if not indexes:
        raise ValueError(
            "No simulation output files were found. Expected keys such as BF1, SF1, "
            "ThermoF1, or TrajF1."
        )

    simulations = [
        SimulationFiles(
            index=idx,
            bond=grouped["bond"].get(idx),
            species=grouped["species"].get(idx),
            thermo=grouped["thermo"].get(idx),
            trajectory=grouped["trajectory"].get(idx),
        )
        for idx in indexes
    ]
    return LammpalyzeConfig(
        input_file=path,
        element_list=element_list,
        simulations=simulations,
        default_bond_order_cutoff=default_bond_order_cutoff,
        bond_order_cutoffs=bond_order_cutoffs,
        bond_state_persistence_frames=persistence_frames,
        bond_state_persistence_timesteps=persistence_timesteps,
        bond_order_hysteresis=hysteresis,
        structure_quality_mode=quality_mode,
        ion_charge_threshold=ion_charge_threshold,
    )


def _parse_int_setting(
    assignments: dict[str, str],
    key: str,
    default: int,
    *,
    minimum: int,
) -> int:
    """Parse one integer analysis setting with a lower bound."""

    raw_value = assignments.get(key)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer, received {raw_value!r}.") from exc
    if value < minimum:
        raise ValueError(f"{key} must be at least {minimum}, received {value}.")
    return value


def _parse_float_setting(
    assignments: dict[str, str],
    key: str,
    default: float,
    *,
    minimum: float,
) -> float:
    """Parse one finite floating-point analysis setting with a lower bound."""

    raw_value = assignments.get(key)
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{key} must be a number, received {raw_value!r}.") from exc
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{key} must be finite and at least {minimum}, received {raw_value!r}.")
    return value


def _parse_choice_setting(
    assignments: dict[str, str],
    key: str,
    default: str,
    choices: set[str],
) -> str:
    """Parse one case-insensitive setting selected from fixed choices."""

    raw_value = assignments.get(key)
    if raw_value is None:
        return default
    value = _strip_quotes(raw_value).lower()
    if value not in choices:
        expected = ", ".join(sorted(choices))
        raise ValueError(f"{key} must be one of {expected}, received {raw_value!r}.")
    return value


def _parse_bond_order_cutoffs(
    path: Path,
    n_atom_types: int,
) -> tuple[float, dict[tuple[int, int], float]]:
    """Read the optional ``Bond Order cutoffs`` section."""

    default_cutoff = DEFAULT_BOND_ORDER_CUTOFF
    cutoffs: dict[tuple[int, int], float] = {}
    in_section = False

    with path.open(encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not in_section:
                if _BOND_ORDER_CUTOFF_HEADER_RE.match(stripped):
                    in_section = True
                continue

            if not stripped:
                continue
            if stripped.startswith("#") or stripped.startswith("---"):
                break

            value = raw_line.split("#", maxsplit=1)[0].strip()
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", value):
                break

            parts = value.replace("=", " ").split()
            if parts and parts[0].lower() == "default":
                if len(parts) != 2:
                    raise ValueError(
                        f"Invalid default bond-order cutoff on line {line_no} in {path}: "
                        "expected 'default <float>'."
                    )
                default_cutoff = _parse_cutoff_value(parts[1], line_no, path)
                continue

            if len(parts) != 3:
                raise ValueError(
                    f"Invalid bond-order cutoff on line {line_no} in {path}: "
                    "expected '<atom type> <atom type> <float>'."
                )
            atom_types_a = _parse_atom_type_selector(parts[0], n_atom_types, line_no, path)
            atom_types_b = _parse_atom_type_selector(parts[1], n_atom_types, line_no, path)
            cutoff = _parse_cutoff_value(parts[2], line_no, path)
            for atom_type_a in atom_types_a:
                for atom_type_b in atom_types_b:
                    pair = tuple(sorted((atom_type_a, atom_type_b)))
                    cutoffs[pair] = cutoff

    return default_cutoff, cutoffs


def _parse_atom_type_selector(
    raw_value: str,
    n_atom_types: int,
    line_no: int,
    path: Path,
) -> range:
    """Expand one atom type or an inclusive ``start*end`` type range."""

    match = re.fullmatch(r"(\d+)(?:\*(\d+))?", raw_value)
    if match is None:
        raise ValueError(
            f"Invalid atom type {raw_value!r} in bond-order cutoff on line {line_no} in {path}: "
            "expected an integer or an inclusive range such as '1*3'."
        )
    start = int(match.group(1))
    end = int(match.group(2) or start)
    if start > end:
        raise ValueError(
            f"Invalid atom-type range {raw_value!r} in bond-order cutoff on line {line_no} "
            f"in {path}: the range start must not exceed its end."
        )
    if start < 1 or end > n_atom_types:
        raise ValueError(
            f"Invalid atom type selector {raw_value!r} in bond-order cutoff on line {line_no} "
            f"in {path}: element_list defines types 1 through {n_atom_types}."
        )
    return range(start, end + 1)


def _parse_cutoff_value(raw_value: str, line_no: int, path: Path) -> float:
    """Parse and validate one non-negative bond-order cutoff."""

    try:
        cutoff = float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid bond-order cutoff {raw_value!r} on line {line_no} in {path}: "
            "expected a float."
        ) from exc
    if not math.isfinite(cutoff) or cutoff < 0:
        raise ValueError(
            f"Invalid bond-order cutoff {raw_value!r} on line {line_no} in {path}: "
            "expected a finite, non-negative value."
        )
    return cutoff


def _read_assignments(path: Path) -> dict[str, str]:
    """Extract simple ``key = value`` pairs while ignoring comments."""

    assignments: dict[str, str] = {}
    assignment_re = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")

    with path.open(encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.split("#", maxsplit=1)[0].strip()
            if not line or line.startswith("---"):
                continue
            match = assignment_re.match(line)
            if not match:
                continue
            key, value = match.groups()
            if not value:
                raise ValueError(f"Missing value for {key!r} on line {line_no} in {path}")
            assignments[key] = value.strip()

    return assignments


def _parse_element_list(assignments: dict[str, str]) -> list[str]:
    """Interpret the required atom-type mapping from the assignment table."""

    raw_value = assignments.get("element_list")
    if raw_value is None:
        raise ValueError("Missing required setting: element_list = [\"C\", \"H\", ...]")

    try:
        parsed = ast.literal_eval(raw_value)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Could not parse element_list: {raw_value}") from exc

    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError("element_list must be a Python-style list of strings.")
    return parsed


def _group_paths(assignments: dict[str, str], base_dir: Path) -> dict[str, dict[int, Path]]:
    """Sort path assignments into bond/species/thermo/trajectory buckets."""

    grouped: dict[str, dict[int, Path]] = {topic: {} for topic in TOPIC_PREFIXES}

    for key, raw_value in assignments.items():
        if key == "element_list":
            continue
        topic = _topic_for_key(key)
        if topic is None:
            continue
        index = _suffix_number(key)
        value = _strip_quotes(raw_value)
        output_path = Path(value).expanduser()
        if not output_path.is_absolute():
            output_path = base_dir / output_path
        grouped[topic][index] = output_path.resolve()

    return grouped


def _topic_for_key(key: str) -> str | None:
    """Work out which file category a user-supplied key names."""

    for topic, prefixes in TOPIC_PREFIXES.items():
        if any(key.startswith(prefix) for prefix in prefixes):
            return topic
    return None


def _suffix_number(key: str) -> int:
    """Read the simulation number from a key such as ``BF2``."""

    match = re.search(r"(\d+)$", key)
    if match:
        return int(match.group(1))
    return 1


def _strip_quotes(value: str) -> str:
    """Allow paths to be written with or without surrounding quotes."""

    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
