"""Parser for LAMMPS reaxff/species output."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


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
