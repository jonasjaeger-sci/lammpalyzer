"""Configuration parsing for lammpalyze input files."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path


TOPIC_PREFIXES = {
    "bond": ("BF", "BondF", "BondFile"),
    "species": ("SF", "SpeciesF", "SpeciesFile"),
    "thermo": ("ThermoF", "TF", "ThermoFile"),
    "trajectory": ("TrajF", "TrajectoryF", "TrajectoryFile"),
}


@dataclass(frozen=True)
class SimulationFiles:
    """Paths belonging to one simulation replica/run."""

    index: int
    bond: Path | None = None
    species: Path | None = None
    thermo: Path | None = None
    trajectory: Path | None = None


@dataclass(frozen=True)
class LammpalyzeConfig:
    """Parsed lammpalyze input file."""

    input_file: Path
    element_list: list[str]
    simulations: list[SimulationFiles]

    @property
    def type_to_element(self) -> dict[int, str]:
        """Return the LAMMPS atom-type to element mapping."""

        return {idx + 1: element for idx, element in enumerate(self.element_list)}


def parse_input_file(input_file: str | Path, *, validate: bool = True) -> LammpalyzeConfig:
    """Parse a lammpalyze input file.

    The parser accepts the current ``lmplyz.inp`` style, for example
    ``BF1 = bonds_R1.reax`` and ``element_list = ["C", "H"]``. Relative paths
    are resolved relative to the input file directory.
    """

    path = Path(input_file).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    assignments = _read_assignments(path)
    element_list = _parse_element_list(assignments)
    grouped = _group_paths(assignments, path.parent)

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
    config = LammpalyzeConfig(path, element_list, simulations)

    if validate:
        validate_config(config)
    return config


def validate_config(config: LammpalyzeConfig) -> None:
    """Validate referenced paths and required settings."""

    if not config.element_list:
        raise ValueError("element_list is empty. Add for example: element_list = [\"C\", \"H\", \"O\"]")

    missing: list[str] = []
    for simulation in config.simulations:
        for topic in ("bond", "species", "thermo", "trajectory"):
            value = getattr(simulation, topic)
            if value is not None and not value.exists():
                missing.append(f"simulation {simulation.index} {topic}: {value}")

    if missing:
        details = "\n".join(f"  - {item}" for item in missing)
        raise FileNotFoundError(f"Referenced output file(s) do not exist:\n{details}")


def _read_assignments(path: Path) -> dict[str, str]:
    """Read ``key = value`` assignments from a lammpalyze input file."""

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
    """Parse the required ``element_list`` assignment as a list of strings."""

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
    """Group output-file assignments by topic and simulation index."""

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
    """Return the output topic represented by an input-file assignment key."""

    for topic, prefixes in TOPIC_PREFIXES.items():
        if any(key.startswith(prefix) for prefix in prefixes):
            return topic
    return None


def _suffix_number(key: str) -> int:
    """Return the trailing integer suffix from ``key``, defaulting to one."""

    match = re.search(r"(\d+)$", key)
    if match:
        return int(match.group(1))
    return 1


def _strip_quotes(value: str) -> str:
    """Remove matching single or double quotes around an assignment value."""

    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
