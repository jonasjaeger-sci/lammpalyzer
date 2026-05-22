"""SMILES utilities and molecule rendering."""

from __future__ import annotations

from io import BytesIO

try:
    from PIL import Image, ImageTk
except ImportError:  # pragma: no cover - optional GUI dependency.
    Image = None
    ImageTk = None

try:
    from rdkit import Chem
    from rdkit.Chem import Draw
except ImportError:  # pragma: no cover - optional external package.
    Chem = None
    Draw = None


def canonicalize_smiles(smiles: str) -> str:
    """Return canonical SMILES, raising ``ValueError`` for invalid strings."""

    _require_rdkit()
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    mol.UpdatePropertyCache(strict=False)
    return Chem.MolToSmiles(mol)


def molecule_image(smiles: str, size: tuple[int, int] = (420, 320)):
    """Render a SMILES string to a PIL image."""

    _require_rdkit()
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    mol.UpdatePropertyCache(strict=False)
    return Draw.MolToImage(mol, size=size)


def molecule_photo_image(smiles: str, size: tuple[int, int] = (420, 320)):
    """Render a SMILES string to a Tkinter-compatible image."""

    if ImageTk is None:
        raise ImportError("Pillow is required for molecule images in the Tkinter GUI.")
    image = molecule_image(smiles, size=size)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return ImageTk.PhotoImage(Image.open(buffer))


def formulas_for_simulation(chem_formulas: dict[int, list[str]]) -> list[str]:
    """Return sorted formulas observed during a simulation."""

    return sorted({formula for formulas in chem_formulas.values() for formula in formulas})


def smiles_for_formula(
    chem_formulas: dict[int, list[str]],
    smiles_by_time: dict[int, list[str]],
    formula: str,
) -> list[str]:
    """Return sorted unique SMILES strings observed for ``formula``."""

    values: set[str] = set()
    for timestep, formulas in chem_formulas.items():
        for index, observed_formula in enumerate(formulas):
            if observed_formula == formula:
                values.add(smiles_by_time[timestep][index])
    return sorted(values)


def _require_rdkit() -> None:
    if Chem is None or Draw is None:
        raise ImportError(
            "RDKit is required for SMILES visualization. "
            "Install it with conda-forge rdkit or pip install rdkit."
        )
