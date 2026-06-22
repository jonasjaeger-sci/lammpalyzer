# lammpalyze

`lammpalyze` analyzes output files from LAMMPS/ReaxFF simulations. It reads
species, thermodynamic, trajectory, and bond files; counts reaction paths in
SMILES notation; and provides a Tkinter GUI for plotting and visualization.

## Quick Start

Clone the repository, then enter it and install the package:

```bash
git clone git@github.com:jonasjaeger-sci/lammpalyzer.git
cd lammpalyzer
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

# Bond Order cutoffs
default 0.3
3 3 0.8
3 4 0.55
3 1*2 0.3

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

### Input Rules

Most useful lines in `lmplyz.inp` are comments or `key = value` assignments.
Comments start with `#`, and blank lines are ignored. The optional bond-order
cutoff section described below uses space-separated values instead.

`element_list` is required and must be written as a Python-style list of element
symbols:

```text
element_list = ["C", "H", "Li", "O"]
```

To omit weak bonds from RDKit molecule construction, add an optional
`# Bond Order cutoffs` section before the file sections:

```text
# Bond Order cutoffs
default 0.5
1 2 0.3
3 4 0.55
3 1*2 0.3

# Bond files
BF1 = bonds_R1.reax
```

Each pair row contains two atom types followed by its cutoff. Pairs are
unordered, so `1 3` and `3 1` configure the same bond. An inclusive range such
as `1*2` expands to types 1 and 2. The `default` row changes the fallback cutoff
for pairs without their own row; if it is omitted, the fallback is `0.3`. Bond
orders below the applicable cutoff are ignored, while values equal to it are
retained.

### How SMILES Are Constructed

For every ReaxFF bond-file frame, atom types are mapped through `element_list`,
bonds below their configured cutoff are removed, and retained bond orders are
mapped to single (`<1.5`), double (`1.5` to `<2.5`), or triple (`>=2.5`) RDKit
bonds. Each connected component becomes one canonical SMILES string. Implicit
hydrogens are disabled, so hydrogen atoms come from the simulation and are
written explicitly.

These SMILES are labels for the thresholded snapshot graph, not validated Lewis
structures. Coordinates, partial charges, total atom bond order, and lone-pair
values are currently ignored; formal charges, aromaticity, and stereochemistry
are not inferred. Consequently, cutoff crossings and short-lived or unusual
ReaxFF configurations can produce fragmented, high-valence, or otherwise
chemically implausible strings. Interpret reaction counts with the chosen
cutoffs and bond-file sampling interval in mind.

File entries use a short prefix plus an optional simulation number. For example,
`BF1`, `SF1`, `ThermoF1`, and `TrajF1` are grouped as simulation 1; `BF2`,
`SF2`, `ThermoF2`, and `TrajF2` are grouped as simulation 2. If no number is
given, lammpalyze treats the entry as simulation 1.

Use these prefixes:

```text
BF, BondF, BondFile                  -> ReaxFF bond file
SF, SpeciesF, SpeciesFile            -> species file
ThermoF, TF, ThermoFile              -> thermodynamic log file
TrajF, TrajectoryF, TrajectoryFile   -> trajectory file
```

Unknown assignments are ignored. Repeated keys overwrite earlier values.
Comments start at `#`, so generated paths should not contain `#` characters.
Relative paths are resolved relative to the directory containing `lmplyz.inp`.

## Running

Run the package with:

```bash
lammpalyze -i lmplyz.inp
```

This loads the configured files, writes reaction path counts to `paths.csv`, and
opens the GUI when a display is available.

Useful command examples:

```bash
# Run without opening the GUI
lammpalyze -i lmplyz.inp --no-gui

# Force the GUI to open
lammpalyze -i lmplyz.inp --gui

# Write reaction paths to a custom file
lammpalyze -i lmplyz.inp -o reaction_paths.csv
```

Before running a full analysis, validate the input file and the first readable
parts of the referenced outputs:

```bash
lammpalyze validate -i lmplyz.inp
```

The validator reports missing files, likely suffix/index mismatches, atom types
not covered by `element_list`, and trajectory atom columns that lammpalyze cannot
read.

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
- `Connected pathways`: view connected reaction states by pathway depth in
  chemical formula or SMILES notation, filter by minimum occurrence count, and
  export the visible pathway table as CSV.
- `Reaction visualization`: open the first occurrence of a selected reaction in
  OVITO, if OVITO is installed.

## Output: `paths.csv`

The reaction path output is a CSV file. It starts with a small metadata block,
then lists reaction counts per simulation and as a total:

```csv
Metadata,Value
input_file,/path/to/lmplyz.inp
run_date,2026-05-29T15:20:30+02:00
simulation_ids,1;2
software_version,1.3.0

Reaction,Simulation 1,Simulation 2,Sum
"['[H][H]'] -> ['[H]', '[H]']",2,1,3
"['[Li]', '[O]'] -> ['[Li][O]']",1,0,1
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

Run the configured style and lint checks:

```bash
python -m pycodestyle lammpalyze tests
python -m pydocstyle lammpalyze tests
python -m pylint lammpalyze
```

The `lammpalyze` and `tests` arguments mean the checks cover both package
source code and tests. `pycodestyle` reads its line-length setting from
`setup.cfg`; `pytest`, `pydocstyle`, and `pylint` read their project settings
from `pyproject.toml`.

## Package Layout

```text
lammpalyze/
  cli.py          command-line entry point
  config.py       input-file parsing
  validation.py   input preflight validation
  analysis.py     project loading and shared numerical helpers
  reactions.py    reaction path counting and occurrence lookup
  rdf.py          radial distribution function calculations
  plotting.py     Matplotlib plotting helpers
  parsers/        species, thermo, bond, and trajectory readers
  gui/            Tkinter GUI tabs and application shell
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
