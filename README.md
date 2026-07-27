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

bond_state_persistence_frames = 2
bond_state_persistence_timesteps = 0
bond_order_hysteresis = 0.05
structure_quality_mode = flag
ion_charge_threshold = 0.5

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

# Pairwise local dumps
Dump1 = pair_distances_R1.dat
Dump2 = pair_distances_R2.dat

# Mean-square-displacement data
MSD1 = msd_R1.dat
MSD2 = msd_R2.dat
```

The number at the end of each key groups files into simulations. For example,
`BF1`, `SF1`, `ThermoF1`, `TrajF1`, `Dump1`, and `MSD1` belong to simulation 1.

`element_list` maps LAMMPS atom types to element symbols. In the example above,
atom type 1 is `C`, type 2 is `H`, type 3 is `Li`, and type 4 is `O`.
The filtering values in the example intentionally opt into two-frame filtering;
omitting them uses the defaults listed below.

### Input Rules

Most useful lines in `lmplyz.inp` are comments or `key = value` assignments.
Comments start with `#`, and blank lines are ignored. The optional bond-order
cutoff section described below uses space-separated values instead.

`element_list` is required and must be written as a Python-style list of element
symbols:

```text
element_list = ["C", "H", "Li", "O"]
```

The following top-level analysis keywords are optional. Put them before the
`# Bond Order cutoffs` section; encountering another `key = value` assignment
ends that space-separated cutoff section.

| Keyword | Default | Accepted values | Purpose |
| --- | ---: | --- | --- |
| `bond_state_persistence_frames` | `1` | integer >= 1 | Consecutive sampled bond frames required before accepting a changed bond state. |
| `bond_state_persistence_timesteps` | `0` | integer >= 0 | Minimum elapsed LAMMPS timesteps for a candidate state; `0` disables this requirement. |
| `bond_order_hysteresis` | `0.0` | finite number >= 0 | Forms connectivity at `cutoff + value` and breaks it below `cutoff - value`. |
| `structure_quality_mode` | `flag` | `keep`, `flag`, `exclude`, `skip` | Controls how suspicious components affect reporting and reaction analysis. |
| `ion_charge_threshold` | `0.5` | finite number >= 0 | Component partial-charge magnitude used for cation/anion candidate labels; `0` disables labels. |

The two persistence requirements use **AND** logic. Leaving the first three
keywords at `1`, `0`, and `0.0` reproduces unfiltered snapshot behavior.

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
values are not used to assign bonds; formal charges, aromaticity, and
stereochemistry are not inferred. Consequently, cutoff crossings and unusual
ReaxFF configurations can produce fragmented or chemically implausible strings.

### Temporal Filtering

Temporal filtering is optional and disabled by default. A new bond state must
satisfy both configured persistence limits before it is accepted:

```text
bond_state_persistence_frames = 2       # sampled bond frames, minimum 1
bond_state_persistence_timesteps = 100  # elapsed LAMMPS steps, 0 disables
bond_order_hysteresis = 0.05            # form above cutoff+value, break below cutoff-value
```

The first bond-file frame establishes the baseline immediately. After that,
persistence applies to bond formation, breaking, and single/double/triple order
changes. If frames are written every 100 timesteps, a candidate first seen at
100 and retained at 200 satisfies `bond_state_persistence_frames = 2` at frame
200. Adding `bond_state_persistence_timesteps = 500` delays acceptance until the
first later sampled frame that is both the required consecutive observation and
at least 500 timesteps after the candidate began. Once accepted, changes are
backdated to the first sampled frame where the candidate state appeared, so
brief unresolved fluctuations still use the previous stable state while
confirmed changes appear from their onset.

Frame counts are convenient for fixed dump frequencies; elapsed timesteps remain
meaningful when sampling intervals differ. Hysteresis applies to connectivity,
while persistence also suppresses brief changes across the single/double/triple
boundaries at 1.5 and 2.5. A ReaxFF bond file cannot recover bond orders already
omitted by the coarse cutoff used when LAMMPS wrote it.

Filtering is applied during project loading and has no separate GUI control.
Reload lammpalyze after changing these keywords. Its effects appear in
`Reaction paths`, `Connected pathways`, `Reaction visualization`, exported
`paths.csv`, and the structures available in `Molecule visualization`. It does
not alter species-file, thermodynamic, RDF, or trajectory-backed atomic data.

### Structure Quality Modes and Partial Charges

Lammpalyze records component partial-charge totals, per-element atomic charge
means and population standard deviations, ion candidates whose component charge
magnitude reaches `ion_charge_threshold` (default `0.5` e), and conservative
valence/sanitization warnings for common covalent nonmetals. Metals are not
judged by ordinary covalent valence limits. Partial charges remain continuous
values and are not converted directly into formal SMILES charges. Choose how
suspicious components affect reaction analysis with:

```text
structure_quality_mode = keep     # retain all components
structure_quality_mode = flag     # retain and report suspicious components (default)
structure_quality_mode = exclude  # skip reactions touching suspicious components
structure_quality_mode = skip     # bridge suspicious intermediates between clean states
```

For `H`, `B`, `C`, `N`, `O`, `F`, `Si`, `P`, `S`, `Cl`, `Br`, and `I`, the
discrete single/double/triple bond-valence sum is compared with RDKit's supported
valences. Components composed entirely of those elements and without an obvious
excess are also passed through RDKit sanitization. Ordinary covalent valence
limits are not applied to metals such as Li; checked nonmetal atoms in mixed
components are still tested.

- `keep` retains every component in reaction analysis and suppresses the load
  warning summary, while keeping quality metadata available for inspection.
- `flag` retains every component, reports the number of suspicious observations
  during loading, and is the default exploratory mode.
- `exclude` preserves raw components and metadata but skips any complete
  reaction event touching a suspicious reactant or product. This avoids creating
  artificial molecule-appearance or disappearance reactions.
- `skip` preserves the same raw data but follows atom IDs across consecutive
  suspicious observations. Once the complete connected atom lineage is clean
  again, it registers a direct reaction between the last clean baseline and the
  recovered clean state. For example, `A -> B -> C` with suspicious `B` becomes
  `A -> C`. A partially suspicious split such as `A -> B + D -> C + D` becomes
  `A -> C + D`, while unrelated clean reactions in those frames remain intact.

If a trajectory begins with a suspicious structure and has no earlier clean
baseline, `skip` cannot invent one; it waits for a clean state and resumes
normal tracking without registering a bridged reaction. Bridged occurrences use
the clean endpoint timesteps in reaction tables and visualization.

The molecule tab summarizes component-charge ranges, ion-candidate counts, and
suspicious-observation counts for a selected SMILES. The `Atomic data` tab reads
charge and other optional scalar values from trajectory `ITEM: ATOMS` tables;
it therefore works without a ReaxFF bond file. Partial charges remain continuous
ReaxFF values and are not rounded into formal SMILES charges.

The `Pairwise data` tab reads LAMMPS local dumps whose `ITEM: ENTRIES` table
contains a local index, two particle IDs, and one or more numeric data columns.
The local index is discarded, reversed particle-ID orders are combined into one
stable `low-high` pair label, and the data column and particle pairs can be
selected independently, with buttons to select or deselect all pairs. An
optional atom selector adds the molecule containing that atom on a second
y-axis. Molecules are assigned stable integer values in first-observation order,
while the tick labels show either their chemical formulas or SMILES notation.
The pairwise legend placement is configurable.

The `Distances and angles` tab calculates geometry directly from configured
trajectory files. Choose `Distance` or `Angle`, select one or more simulations,
and enter either one atom ID per field or equally sized lists such as `[1, 4]`
and `[2, 5]`. List elements at the same position form one measurement. Angles
use the second atom as the vertex (`atom 1 - atom 2 - atom 3`). Distances and
angle arms use periodic minimum-image displacements, and the resulting values
are plotted against trajectory timestep in Å or degrees. Optionally, enter one
or more atom IDs in the chemical-state field to add a secondary y-axis showing
the molecule containing each atom over time, labelled by either chemical formula
or SMILES notation.

All embedded line plots show the nearest series label and its x/y coordinates
when the mouse pointer is close to a plotted data point.

The `Mean-square displacement` tab reads computed tables with a `TimeStep`
header followed by numeric data columns. Every file/column combination appears
in its scrollable selector, for example `MSD1 - c_msd_C[1]`, with select-all and
deselect-all buttons. As in the thermodynamic-data tab, it creates a
selected-series plot and a second aligned mean/standard-deviation plot. Optional
semicolon-separated simulation groups such as `1,2; 3,4` produce separately
labelled averages. Both legends share a configurable placement.

File entries use a short prefix plus an optional simulation number. For example,
`BF1`, `SF1`, `ThermoF1`, `TrajF1`, `Dump1`, and `MSD1` are grouped as
simulation 1; keys ending in `2` are grouped as simulation 2. If no number is
given, lammpalyze treats the entry as simulation 1.

Use these prefixes:

```text
BF, BondF, BondFile                  -> ReaxFF bond file
SF, SpeciesF, SpeciesFile            -> species file
ThermoF, TF, ThermoFile              -> thermodynamic log file
TrajF, TrajectoryF, TrajectoryFile   -> trajectory file
Dump, DumpF, PairF, PairwiseF, PairwiseFile -> pairwise local-dump file
MSD, MSDF, MSDFile                   -> computed mean-square-displacement file
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
  simulations, edit legend labels, and adjust x/y axis ranges. Existing plots
  update automatically when either range is edited or reset.
- `Radial distribution`: calculate RDF curves for selected element pairs such as
  `Li-Li` or `Li-O`, with selectable simulations, timestep range, bin width,
  optional point-based running averages, and a second cross-simulation mean and
  standard-deviation plot over the shared radial range. A dropdown controls the
  legend placement in both plots.
- `Structural relaxation`: calculate a preliminary static structure factor
  `S(q)` from uniformly sampled production frames, then use the first `S(q)`
  peak to calculate the incoherent scattering function `F_s(q,t)`. Select one
  or more simulations, all atoms or one element, the production start timestep,
  number of frames, number of time origins, maximum q-vector integer index, and
  number of uncertainty blocks.
- `Atomic data`: stream flexible `ITEM: ATOMS` trajectory fields and plot a
  selected property by element or by atom ID. Charge (`q`), force components
  (`fx`, `fy`, `fz`), and velocity components (`vx`, `vy`, `vz`) are available
  when present. Complete component sets also provide force magnitude `f` and
  velocity magnitude `v`. Element plots support population-standard-deviation
  bands or error bars. The optional `Plot individual atoms` setting adds a
  second figure for selections of up to 200 atoms; broader selections report a
  clear error instead of overloading the plot. Atom-ID plots show one line per
  explicitly selected atom. Atom IDs accept comma/space-separated values and
  inclusive ranges such as `1, 4-8`. A progress indicator remains visible while
  large trajectory files are read in the background.
- `Molecule visualization`: render one or all observed SMILES structures for a
  formula and summarize component charges, ion candidates, and quality flags.
- `Reaction paths`: view total and per-simulation reaction path counts, then copy
  only the reaction path string.
- `Connected pathways`: view connected reaction states by pathway depth in
  chemical formula or SMILES notation, filter by minimum occurrence count, and
  export the visible pathway table as CSV.
- `Reaction visualization`: open the first occurrence of a selected reaction in
  OVITO, if OVITO is installed.

### Structural Relaxation Calculations

The `Structural relaxation` tab reads configured trajectory files and starts
from the user-entered production timestep. From all frames at or after that
timestep, it selects the requested number of frames uniformly over the available
range. The default is 100 frames. The same atom selection is used for both
`S(q)` and `F_s(q,t)`: choose `All` for every atom or choose one element from
`element_list`.

Wave vectors are generated as:

```text
q = 2*pi/L * (n_x, n_y, n_z)
```

where `n_x`, `n_y`, and `n_z` are integers between `-max_q_index` and
`max_q_index`, excluding `(0, 0, 0)`. `L` is the mean selected-frame box length
in each direction. Vectors with the same rounded magnitude `|q|` are grouped
into shells, and each shell is averaged to reduce directional noise. For each
frame and q shell, lammpalyze evaluates:

```text
S(q) = < |sum_j exp(i q . r_j)|^2 / N >_shell
```

The plotted `S(q)` value is the frame average for each `|q|` shell. Error bars
are estimated by splitting the sampled frames into contiguous blocks, averaging
inside each block, and taking the standard error of the block means.

The first local maximum in the averaged `S(q)` curve is used as the preliminary
structural wave vector for the incoherent scattering calculation. If no local
maximum is found, the global maximum is used. For each uniformly spaced time
origin, default 10 origins, lammpalyze computes:

```text
F_s(q,t) = < cos(q . [r_j(t0 + t) - r_j(t0)]) >_{j, shell, origins}
```

Only time origins with available future frames contribute at each lag time.
Uncertainty bands are estimated by block averaging over the selected time
origins. The time axis is reported in trajectory timestep units.

## Output: `paths.csv`

The reaction path output is a CSV file. It starts with a small metadata block,
then lists reaction counts per simulation and as a total:

```csv
Metadata,Value
input_file,/path/to/lmplyz.inp
run_date,2026-05-29T15:20:30+02:00
simulation_ids,1;2
software_version,1.4.0
default_bond_order_cutoff,0.3
bond_order_cutoffs,3-3:0.8;3-4:0.55
bond_state_persistence_frames,2
bond_state_persistence_timesteps,0
bond_order_hysteresis,0.05
structure_quality_mode,flag
ion_charge_threshold,0.5

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

The style and docstring checks cover both package source and tests; Pylint checks
the package. `pycodestyle` reads its line-length setting from `setup.cfg`, while
`pytest`, `pydocstyle`, and `pylint` read project settings from `pyproject.toml`.

## Package Layout

```text
lammpalyze/
  cli.py          command-line entry point
  config.py       input-file parsing
  validation.py   input preflight validation
  analysis.py     project loading and shared numerical helpers
  reactions.py    reaction path counting and occurrence lookup
  rdf.py          radial distribution function calculations
  rdf_plotting.py radial distribution plotting and cross-simulation averaging
  structure.py    static structure factor and incoherent scattering calculations
  structure_plotting.py structural-relaxation plotting
  geometry.py     trajectory pair-distance and three-atom-angle calculations
  geometry_plotting.py distance-and-angle plotting
  plotting.py     Matplotlib plotting helpers
  parsers/        species, thermo, computed-data, bond, and trajectory readers
  gui/            Tkinter GUI tabs and application shell
    charge_tab.py trajectory-backed atomic-data plotting tab
    computed_tabs.py pairwise-data and mean-square-displacement tabs
    geometry_tab.py trajectory distance-and-angle tab
    structure_tab.py structural-relaxation tab
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
