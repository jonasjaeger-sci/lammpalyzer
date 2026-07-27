"""Parser for LAMMPS thermodynamic log output."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def eval_thermo(
    thermo_file: str | Path,
    indicator1: str = "Step",
    indicator2: str = "Loop",
) -> tuple[dict[str, list[float]], pd.DataFrame]:
    """Parse all thermo tables from a LAMMPS log file."""

    thermo_path = Path(thermo_file)
    thermo_cols: list[str] = []
    first_thermo_cols: list[str] | None = None
    thermo_rows: list[dict[str, float]] = []
    in_table = False

    with thermo_path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue

            if stripped.startswith(indicator1):
                thermo_cols = stripped.split()
                if first_thermo_cols is None:
                    first_thermo_cols = thermo_cols
                in_table = thermo_cols == first_thermo_cols
                continue

            if stripped.startswith(indicator2):
                in_table = False
                thermo_cols = []
                continue

            if in_table:
                values = stripped.split()
                if len(values) < len(thermo_cols):
                    continue
                try:
                    row = {
                        col: float(value)
                        for value, col in zip(values, thermo_cols, strict=False)
                    }
                except ValueError:
                    continue
                thermo_rows.append(row)

    if not thermo_rows:
        raise ValueError(f"No thermo table starting with {indicator1!r} found in {thermo_path}")
    thermo_frame = pd.DataFrame(thermo_rows)
    return thermo_frame.to_dict(orient="list"), thermo_frame
