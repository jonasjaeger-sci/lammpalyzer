"""Parsers for computed time-series and pairwise local-dump output."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


TIMESTEP_COLUMN = "Timestep"
PAIR_COLUMN = "Pair"
PARTICLE_1_COLUMN = "Particle 1"
PARTICLE_2_COLUMN = "Particle 2"
PAIR_METADATA_COLUMNS = [TIMESTEP_COLUMN, PAIR_COLUMN, PARTICLE_1_COLUMN, PARTICLE_2_COLUMN]
_TIMESTEP_HEADERS = {"step", "timestep", "time_step"}


def eval_msd(msd_file: str | Path) -> pd.DataFrame:
    """Read a LAMMPS computed MSD table into a numeric data frame.

    Files written by ``fix ave/time`` usually contain a comment header such as
    ``# TimeStep c_msd_C[1] ...``.  The first column is normalized to
    :data:`TIMESTEP_COLUMN`; computed column names are preserved verbatim.
    """

    path = Path(msd_file)
    columns: list[str] | None = None
    rows: list[dict[str, float]] = []
    data_columns: list[str] = []

    with path.open(encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue

            if stripped.startswith("#"):
                candidate = stripped.lstrip("#").strip().split()
                if candidate and candidate[0].lower() in _TIMESTEP_HEADERS:
                    columns = _validated_msd_columns(candidate, path, line_no)
                    data_columns = _append_unique(data_columns, columns[1:])
                continue

            values = stripped.split()
            if columns is None:
                if values and values[0].lower() in _TIMESTEP_HEADERS:
                    columns = _validated_msd_columns(values, path, line_no)
                    data_columns = _append_unique(data_columns, columns[1:])
                    continue
                raise ValueError(
                    f"MSD data appears before a TimeStep header on line {line_no} in {path}."
                )
            if len(values) != len(columns):
                raise ValueError(
                    f"MSD row on line {line_no} in {path} has {len(values)} values; "
                    f"expected {len(columns)}."
                )
            try:
                numeric = [float(value) for value in values]
            except ValueError as exc:
                raise ValueError(f"Non-numeric MSD value on line {line_no} in {path}.") from exc
            row = {TIMESTEP_COLUMN: numeric[0]}
            row.update(zip(columns[1:], numeric[1:], strict=True))
            rows.append(row)

    if columns is None:
        raise ValueError(f"No TimeStep header found in MSD file {path}.")
    if not rows:
        raise ValueError(f"No MSD observations found in {path}.")
    return pd.DataFrame(rows).reindex(columns=[TIMESTEP_COLUMN, *data_columns])


def eval_pairwise_dump(pairwise_file: str | Path) -> pd.DataFrame:
    """Read a LAMMPS local dump containing index, two particle IDs, and data.

    The local row index is discarded. Particle IDs are sorted into a stable
    ``low-high`` descriptor so neighbor-list ordering changes do not split one
    physical pair into two plot series.
    """

    path = Path(pairwise_file)
    rows: list[dict[str, float | int | str]] = []
    data_columns: list[str] = []
    saw_entries = False

    with path.open(encoding="utf-8") as handle:
        while True:
            timestep = _next_pairwise_timestep(handle, path)
            if timestep is None:
                break
            entry_count, entry_columns = _read_pairwise_header(handle, path, timestep)
            saw_entries = True
            if len(entry_columns) < 4:
                raise ValueError(
                    f"Pairwise entries in {path} need index, two particle IDs, and at least one data column."
                )
            frame_data_columns = entry_columns[3:]
            data_columns = _append_unique(data_columns, frame_data_columns)

            for _ in range(entry_count):
                row_line = _required_line(handle, path, "pairwise entry")
                values = row_line.split()
                if len(values) != len(entry_columns):
                    raise ValueError(
                        f"Pairwise entry at timestep {timestep:g} in {path} has {len(values)} values; "
                        f"expected {len(entry_columns)}."
                    )
                try:
                    particle_a = int(float(values[1]))
                    particle_b = int(float(values[2]))
                    numeric_values = [float(value) for value in values[3:]]
                except ValueError as exc:
                    raise ValueError(f"Non-numeric pairwise entry at timestep {timestep:g} in {path}.") from exc
                particle_1, particle_2 = sorted((particle_a, particle_b))
                row: dict[str, float | int | str] = {
                    TIMESTEP_COLUMN: timestep,
                    PAIR_COLUMN: f"{particle_1}-{particle_2}",
                    PARTICLE_1_COLUMN: particle_1,
                    PARTICLE_2_COLUMN: particle_2,
                }
                row.update(zip(frame_data_columns, numeric_values, strict=True))
                rows.append(row)

    if not saw_entries:
        raise ValueError(f"No ITEM: ENTRIES table found in pairwise dump {path}.")
    if not rows:
        return pd.DataFrame(columns=[*PAIR_METADATA_COLUMNS, *data_columns])
    return pd.DataFrame(rows).reindex(columns=[*PAIR_METADATA_COLUMNS, *data_columns])


def msd_data_columns(frame: pd.DataFrame) -> list[str]:
    """Return selectable computed columns from a parsed MSD frame."""

    return [column for column in frame.columns if column != TIMESTEP_COLUMN]


def pairwise_data_columns(frame: pd.DataFrame) -> list[str]:
    """Return selectable values from a parsed pairwise local dump."""

    return [column for column in frame.columns if column not in PAIR_METADATA_COLUMNS]


def _validated_msd_columns(columns: list[str], path: Path, line_no: int) -> list[str]:
    """Validate and normalize an MSD table header."""

    if len(columns) < 2:
        raise ValueError(f"MSD header on line {line_no} in {path} has no data columns.")
    if len(set(columns[1:])) != len(columns[1:]):
        raise ValueError(f"MSD header on line {line_no} in {path} contains duplicate columns.")
    return [TIMESTEP_COLUMN, *columns[1:]]


def _append_unique(existing: list[str], additions: list[str]) -> list[str]:
    """Append unseen column names while preserving discovery order."""

    result = list(existing)
    for value in additions:
        if value not in result:
            result.append(value)
    return result


def _required_line(handle, path: Path, description: str) -> str:
    """Read one required line and report a contextual truncated-file error."""

    line = handle.readline()
    if not line:
        raise ValueError(f"Unexpected end of {path} while reading {description}.")
    return line


def _next_pairwise_timestep(handle, path: Path) -> float | None:
    """Scan to the next local-dump frame and read its timestep."""

    for raw_line in handle:
        if not raw_line.strip().startswith("ITEM: TIMESTEP"):
            continue
        timestep_line = _required_line(handle, path, "timestep")
        try:
            return float(timestep_line.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid timestep {timestep_line.strip()!r} in {path}.") from exc
    return None


def _read_pairwise_header(handle, path: Path, timestep: float) -> tuple[int, list[str]]:
    """Read a frame's entry count and ``ITEM: ENTRIES`` column names."""

    entry_count: int | None = None
    for item_line in handle:
        stripped = item_line.strip()
        if stripped.startswith("ITEM: NUMBER OF ENTRIES"):
            count_line = _required_line(handle, path, "entry count")
            try:
                entry_count = int(count_line.strip())
            except ValueError as exc:
                raise ValueError(f"Invalid entry count {count_line.strip()!r} in {path}.") from exc
            continue
        if not stripped.startswith("ITEM: ENTRIES"):
            continue
        if entry_count is None:
            raise ValueError(f"Pairwise frame at timestep {timestep:g} has no entry count in {path}.")
        return entry_count, stripped.split()[2:]
    raise ValueError(f"Unexpected end of pairwise dump before ITEM: ENTRIES in {path}.")
