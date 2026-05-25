# lammpalyze

`lammpalyze` analyzes output files from LAMMPS/ReaxFF simulations. It reads
species, thermodynamic, trajectory, and bond files; counts reaction paths in
SMILES notation; and provides a Tkinter GUI for plotting and visualization.

## Quick Start

Clone or enter the repository, then install the package:

```bash
pip install -r requirements.txt
pip install -e .
```

The first command installs Python dependencies from `requirements.txt`. The
second command installs `lammpalyze` itself in editable mode, so local code
changes are picked up without reinstalling.

Optional conda environment:

```bash
conda create -n lammpalyzer python=3.12
conda activate lammpalyzer
pip install -r requirements.txt
pip install -e .
```

If RDKit installation through pip fails on your system, install it with conda
instead:

```bash
conda install -c conda-forge rdkit
```

## Input File

Create an input file, usually named `lmplyz.inp`, that points to your simulation
outputs. Relative paths are resolved relative to the input file.

```text
--- LAMMPALYZER Input ---

element_list = ["C", "H", "Li", "O"]

# Bond files
BF1 = bonds_R1.reax
BF2 = bonds_R2.reax

# Species files
SF1 = species_R1_main.out
SF2 = species_R2_main.out

# Thermodynamic log files
ThermoF1 = main_thermo_R1.log
ThermoF2 = main_thermo_R2.log

# Trajectory files
TrajF1 = md.lammpstrj_R1
TrajF2 = md.lammpstrj_R2
```

The number at the end of each key groups files into simulations. For example,
`BF1`, `SF1`, `ThermoF1`, and `TrajF1` belong to simulation 1.

`element_list` maps LAMMPS atom types to element symbols. In the example above,
atom type 1 is `C`, type 2 is `H`, type 3 is `Li`, and type 4 is `O`.

## Running

Run the package with:

```bash
lammpalyze -i lmplyz.inp
```

This loads the configured files, writes reaction path counts to `paths.out`, and
opens the GUI when a display is available.

Useful command examples:

```bash
# Run without opening the GUI
lammpalyze -i lmplyz.inp --no-gui

# Force the GUI to open
lammpalyze -i lmplyz.inp --gui

# Write reaction paths to a custom file
lammpalyze -i lmplyz.inp -o reaction_paths.tsv
```

## GUI Overview

The GUI contains tabs for common analysis tasks:

- `Species analysis`: plot selected species counts over time.
- `Thermodynamic data`: plot selected thermodynamic parameters, choose
  simulations, edit legend labels, and adjust x/y axis ranges.
- `Radial distribution`: calculate RDF curves for selected element pairs such as
  `Li-Li` or `Li-O`, with selectable simulations, timestep range, and bin width.
- `Molecule visualization`: render a selected SMILES molecule image.
- `Reaction paths`: view total and per-simulation reaction path counts, then copy
  only the reaction path string.
- `Reaction visualization`: open the first occurrence of a selected reaction in
  OVITO, if OVITO is installed.

## Output: `paths.out`

The reaction path output is a tab-separated table:

```text
Reaction	Count
['[H][H]'] -> ['[H]', '[H]']	3
['[Li]', '[O]'] -> ['[Li][O]']	1
```

## Tests

Install test dependencies from `requirements.txt`, then run all tests:

```bash
python -m pytest
```

Run one test file:

```bash
python -m pytest tests/test_parsers.py
```

Run one specific test:

```bash
python -m pytest tests/test_rdf.py::test_compute_rdf_averages_selected_timestep_range
```

Run tests with short output:

```bash
python -m pytest -q
```

## Linting

Run pylint on the package:

```bash
python -m pylint lammpalyze
```

## Package Layout

```text
lammpalyze/
  cli.py          command-line entry point
  config.py       input-file parsing and validation
  parsers.py      species, thermo, bond, and trajectory readers
  analysis.py     project loading and shared numerical helpers
  reactions.py    reaction path counting and occurrence lookup
  rdf.py          radial distribution function calculations
  plotting.py     Matplotlib plotting helpers
  gui.py          Tkinter GUI
  smiles.py       SMILES utilities and molecule rendering
  ovito.py        OVITO scene generation
examples/
  example_NVT_vs_NPT/
  example_Temperature/
  example_thermal_dampening/
tests/
requirements.txt
pyproject.toml
README.md
```
