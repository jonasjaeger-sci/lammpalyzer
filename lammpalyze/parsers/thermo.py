"""Parser for LAMMPS thermodynamic log output."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


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
