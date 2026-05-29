"""Lightweight project checks run before the expensive analysis step."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lammpalyze.config import LammpalyzeConfig, parse_input_file

TRAJECTORY_COORDINATE_GROUPS = (
    ("xu", "yu", "zu"),
    ("x", "y", "z"),
    ("xs", "ys", "zs"),
)


@dataclass(frozen=True)
class ValidationIssue:
    """One problem or caution found while inspecting an input file."""

    severity: str
    message: str


@dataclass(frozen=True)
class ValidationReport:
    """Collected validation results for a lammpalyze project."""

    input_file: Path | None
    issues: list[ValidationIssue]

    @property
    def has_errors(self) -> bool:
        """Whether the report contains at least one blocking problem."""

        return any(issue.severity == "error" for issue in self.issues)


def validate_input_file(input_file: str | Path) -> ValidationReport:
    """Inspect a project file and report issues without loading full datasets."""

    issues: list[ValidationIssue] = []
    try:
        config = parse_input_file(input_file, validate=False)
    except Exception as exc:
        return ValidationReport(input_file=None, issues=[ValidationIssue("error", str(exc))])

    _check_missing_files(config, issues)
    _check_simulation_index_consistency(config, issues)
    _check_atom_types(config, issues)
    _check_trajectory_columns(config, issues)

    if not issues:
        issues.append(ValidationIssue("ok", "Input file passed validation checks."))
    return ValidationReport(input_file=config.input_file, issues=issues)


def format_validation_report(report: ValidationReport) -> str:
    """Format validation results for the command-line interface."""

    lines = ["Validation report"]
    if report.input_file is not None:
        lines.append(f"Input file: {report.input_file}")
    for issue in report.issues:
        lines.append(f"{issue.severity.upper()}: {issue.message}")
    return "\n".join(lines)


def _check_missing_files(config: LammpalyzeConfig, issues: list[ValidationIssue]) -> None:
    """Report every referenced file that does not exist."""

    for simulation in config.simulations:
        for topic in ("bond", "species", "thermo", "trajectory"):
            value = getattr(simulation, topic)
            if value is not None and not value.exists():
                issues.append(
                    ValidationIssue(
                        "error",
                        f"Simulation {simulation.index} {topic} file is missing: {value}",
                    )
                )


def _check_simulation_index_consistency(config: LammpalyzeConfig, issues: list[ValidationIssue]) -> None:
    """Warn when file suffixes do not line up cleanly across simulations."""

    expected_topics = {
        topic
        for simulation in config.simulations
        for topic in ("bond", "species", "thermo", "trajectory")
        if getattr(simulation, topic) is not None
    }
    if len(config.simulations) <= 1 or len(expected_topics) <= 1:
        return

    for simulation in config.simulations:
        present = {
            topic
            for topic in ("bond", "species", "thermo", "trajectory")
            if getattr(simulation, topic) is not None
        }
        missing = sorted(expected_topics - present)
        if missing:
            issues.append(
                ValidationIssue(
                    "warning",
                    "Simulation "
                    f"{simulation.index} has {', '.join(sorted(present)) or 'no recognized files'} "
                    f"but is missing {', '.join(missing)}. Check the numeric suffixes in lmplyz.inp.",
                )
            )


def _check_atom_types(config: LammpalyzeConfig, issues: list[ValidationIssue]) -> None:
    """Look for atom types not covered by ``element_list``."""

    known_types = set(config.type_to_element)
    unknown_by_source: dict[str, set[int]] = {}
    for simulation in config.simulations:
        if simulation.bond is not None and simulation.bond.exists():
            unknown = _unknown_bond_types(simulation.bond, known_types)
            if unknown:
                unknown_by_source[f"simulation {simulation.index} bond file"] = unknown
        if simulation.trajectory is not None and simulation.trajectory.exists():
            unknown = _unknown_trajectory_types(simulation.trajectory, known_types)
            if unknown:
                unknown_by_source[f"simulation {simulation.index} trajectory file"] = unknown

    for source, unknown_types in unknown_by_source.items():
        issues.append(
            ValidationIssue(
                "error",
                f"{source} uses atom type(s) {', '.join(map(str, sorted(unknown_types)))} "
                "not covered by element_list.",
            )
        )


def _check_trajectory_columns(config: LammpalyzeConfig, issues: list[ValidationIssue]) -> None:
    """Check whether trajectory atom tables have columns lammpalyze can read."""

    for simulation in config.simulations:
        trajectory = simulation.trajectory
        if trajectory is None or not trajectory.exists():
            continue
        columns = _first_trajectory_columns(trajectory)
        if columns is None:
            issues.append(
                ValidationIssue(
                    "error",
                    f"Simulation {simulation.index} trajectory has no ITEM: ATOMS table: {trajectory}",
                )
            )
            continue

        column_set = set(columns)
        missing_required = sorted({"id", "type"} - column_set)
        has_coordinates = any(all(column in column_set for column in group) for group in TRAJECTORY_COORDINATE_GROUPS)
        if missing_required or not has_coordinates:
            details = []
            if missing_required:
                details.append(f"missing required column(s): {', '.join(missing_required)}")
            if not has_coordinates:
                details.append("needs one coordinate set: xu/yu/zu, x/y/z, or xs/ys/zs")
            issues.append(
                ValidationIssue(
                    "error",
                    f"Simulation {simulation.index} trajectory uses unsupported atom columns "
                    f"({', '.join(columns)}): {'; '.join(details)}.",
                )
            )


def _unknown_bond_types(bond_file: Path, known_types: set[int]) -> set[int]:
    """Read enough of a ReaxFF bond file to find undeclared atom types."""

    unknown = set()
    with bond_file.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                atom_type = int(parts[1])
                if atom_type not in known_types:
                    unknown.add(atom_type)
    return unknown


def _unknown_trajectory_types(trajectory_file: Path, known_types: set[int]) -> set[int]:
    """Scan the first trajectory atom table for atom types outside element_list."""

    unknown = set()
    with trajectory_file.open(encoding="utf-8") as handle:
        n_atoms = 0
        while True:
            line = handle.readline()
            if not line:
                break
            if line.startswith("ITEM: NUMBER OF ATOMS"):
                n_atoms = int(handle.readline().strip())
                continue
            if not line.startswith("ITEM: ATOMS"):
                continue
            columns = line.split()[2:]
            if "type" not in columns:
                return unknown
            type_index = columns.index("type")
            for _ in range(n_atoms):
                values = handle.readline().split()
                atom_type = int(float(values[type_index]))
                if atom_type not in known_types:
                    unknown.add(atom_type)
            break
    return unknown


def _first_trajectory_columns(trajectory_file: Path) -> list[str] | None:
    """Return the first LAMMPS atom-table header found in a trajectory file."""

    with trajectory_file.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("ITEM: ATOMS"):
                return line.split()[2:]
    return None
