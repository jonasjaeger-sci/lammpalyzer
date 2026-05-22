# lammpalyze

`lammpalyze` analyzes output files from LAMMPS/ReaxFF simulations. It parses
species, thermo, trajectory, and bond output files; counts reaction paths in
SMILES notation; and provides a Tkinter GUI for interactive plotting.

## Install

```bash
pip install -e .
```

For bond parsing and SMILES molecule rendering, install RDKit as well:

```bash
conda install -c conda-forge rdkit
```

## Input File

```text
--- LAMMPALYZER Input ---

element_list = ["C","H","Li","O"]

BF1 = bonds_R1.reax
BF2 = bonds_R2.reax

SF1 = species_R1_main.out
SF2 = species_R2_main.out

ThermoF1 = main_thermo_R1.log
ThermoF2 = main_thermo_R2.log

TrajF1 = md.lammpstrj_R1
TrajF2 = md.lammpstrj_R2
```

Relative paths are resolved relative to the input file.

## Usage

```bash
lammpalyze -i lmplyz.inp
```

On machines with a display, this writes `paths.out` and opens the GUI. On
headless machines, it writes `paths.out` without launching the GUI.

```bash
lammpalyze -i lmplyz.inp --no-gui
lammpalyze -i lmplyz.inp --gui -o reaction_paths.tsv
```

## Tests

```bash
pip install -e ".[test]"
python -m pytest -q
```

## `paths.out`

The reaction path output is a tab-separated table:

```text
Reaction	Count
['[H][H]'] -> ['[H]', '[H]']	3
['[Li]', '[O]'] -> ['[Li][O]']	1
```

## GUI

The GUI uses Tkinter because it is included with standard Python
installations and embeds Matplotlib without adding a heavy GUI dependency.

It contains three tabs:

- Species analysis: select simulations and species, then plot counts over time.
- Thermodynamic data: select a parameter such as `Temp`, `PotEng`, or `Density`.
  The GUI creates per-simulation plots plus an average with standard deviation.
- Molecule visualization: select simulation, formula/species, and SMILES string,
  then render a molecule image with RDKit.

## Package Layout

```text
lammpalyze/
  __init__.py
  cli.py
  config.py
  parsers.py
  analysis.py
  reactions.py
  plotting.py
  gui.py
  smiles.py
tests/
pyproject.toml
README.md
```

The old script-level logic from `execute.py` has been moved into reusable
functions. Reaction-path counting lives in `lammpalyze.reactions`, SMILES image
generation lives in `lammpalyze.smiles`, and file readers live in
`lammpalyze.parsers`.
