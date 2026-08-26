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

boundary p p p

bond_state_persistence_frames = 2
bond_state_persistence_timesteps = 0
bond_order_hysteresis = 0.05
structure_quality_mode = flag
ion_charge_threshold = 0.5

# Bond Order cutoffs
default 0.5
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
| `boundary` | `p p p` | three values, each `p` or `f` | Sets periodic (`p`) or fixed/nonperiodic (`f`) behavior along x, y, and z. May be written as `boundary p p f` or `boundary = p p f`. |

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
for pairs without their own row; if it is omitted, the fallback is `0.5`. Bond
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

The `Geometry` tab calculates distances and angles directly from configured
trajectory files. Angles retain the three atom-ID fields and use the second atom
as the vertex (`atom 1 - atom 2 - atom 3`). Distance endpoints can independently
be atoms, mass-weighted centers of mass (COMs) selected by atom IDs or trajectory
`mol` IDs, or planes defined by three atom IDs. A flat COM atom list such as
`[1,2,3]` defines one COM; a nested list such as `[[1,2],[3,4]]` defines two.
Planes use the same nested syntax when more than one plane is needed. Equal-size
endpoint selections are paired by position. Unequal selections use their full
Cartesian product, so atoms `[1,3]` against `[4,5,6]` produce six curves.

Point-to-plane measurements report unsigned orthogonal distances. Plane-to-plane
selection is rejected because two arbitrary planes do not define one unique
scalar distance. Intramolecule mode instead uses only the first field. A flat
atom list calculates every unique pair inside that group, while nested lists
keep groups separate; molecule-ID mode expands each selected `mol` value into
its constituent atom pairs. COMs are periodically unwrapped before mass
weighting. All distances and angle arms respect the configured periodic/fixed
`boundary` modes, and results are plotted against trajectory timestep in Å or
degrees. The optional chemical-state field adds a formula- or SMILES-labelled
secondary y-axis for selected atoms.

Enable `Only while all measurement atoms are part of` to restrict geometry to
one or more bond-derived molecule descriptors. Enter one formula/SMILES or a
Python-style string list such as `["C3H4LiO3", "C2H4O2"]`. `Auto-detect`, the
default, compares every entry against both the formula and SMILES of each
component; explicit formula and SMILES modes remain available. At an exact
shared trajectory/bond timestep, a point is retained only when every atom
contributing to that individual distance or angle belongs to the same connected
component and that component exactly matches any one selected descriptor. This
includes every atom contributing to a COM or plane. Missing observations,
descriptor changes, and component splits become gaps rather than lines
connecting across a broken interval.

The `Snapshot` tab displays bond-derived system state at an entered timestep.
Atom view shows every atom in the matching trajectory frame, its atom type, its
calculated `mol_id`, and the formula or SMILES of its component for up to five
available bond observations before and after the selected observation. Molecule
view shows every calculated component at the selected timestep together with
its notation and constituent atom IDs. Snapshot `mol_id` values are zero-based
connected-component indexes calculated independently for each observation; they
are not trajectory-provided persistent molecule IDs. Click any displayed
formula or SMILES cell and use `Ctrl+C`, double-click, or `Copy notation` to put
it on the clipboard. The formula and SMILES selectors in `Molecule
visualization` are editable, so copied values can be pasted and rendered there.

`Molecule visualization` also builds reusable Geometry descriptor lists. Every
rendered tile in this tab has an `Include in list` checkbox selected by default.
Choose Formula or SMILES list mode, adjust the tile selections, and click `Add`.
The two list modes accumulate independently while simulations and species are
changed, retain first-added order, and ignore duplicates. `Copy list` places the
visible quoted list on the clipboard for direct use in Geometry's molecule
membership filter; `Clear` resets only the currently visible Formula or SMILES
list.

All embedded line plots show the nearest series label and its x/y coordinates
when the mouse pointer is close to a plotted data point.

The GUI `Plot settings` bar applies to newly generated plots. It can display
timestep-like x-axes either as raw timesteps or as real time using a global
timestep size, defaulting to `0.5 fs`, and a display unit, defaulting to `ps`.
It can reset the displayed x-origin to zero for production-run chunks, and it
also provides independent logarithmic x- and y-axis toggles.

Plot legends are hidden by default in every analysis tab. Each tab with
labelled curves provides a `Legend placement` selector for showing the legend
inside the axes or outside it above, below, to the left, or to the right.

The `Mean-square displacement` tab reads computed tables with a `TimeStep`
header followed by numeric data columns. Every file/column combination appears
in its scrollable selector, for example `MSD1 - c_msd_C[1]`, with select-all and
deselect-all buttons. As in the thermodynamic-data tab, it creates a
selected-series plot and a second aligned mean/standard-deviation plot. Optional
semicolon-separated simulation groups such as `1,2; 3,4` produce separately
labelled averages. Both legends share a configurable placement. An optional
linear-fit timestep range overlays per-series fits and reports diffusion
coefficients as `D = slope / 6`, with MSD units of Å² per selected x-axis unit.

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

Large trajectory and thermo files can be sliced before analysis with the
streaming range tools:

```bash
# Extract complete dump frames from timestep 124770 through 125500
pieceoftraj -i file.traj -s 124770 -e 125500

# Extract thermo rows whose first column is in the same timestep range
chopthermo -i file.log -s 124770 -e 125500
```

If `-o/--output` is omitted, the output is written beside the input as
`file_124770_125500.traj` or `file_124770_125500.log`. Trajectory frame headers
and thermo table headers are preserved when present.

### RDF shell normalization and boundaries

RDF histograms use the exact three-dimensional volume of every spherical shell,

```text
4 pi / 3 * (r_outer^3 - r_inner^3)
```

rather than the thin-shell approximation `4 pi * r_center^2 * bin_width`.
The difference is most important in the first few bins or when using a coarse
bin width. The final bin is shortened when necessary so it ends at the largest
supported radius instead of extending beyond it.

The optional project-level `boundary` setting controls RDF distance wrapping.
For example,

```text
boundary p p f
```

uses minimum-image wrapping along x and y, but treats z as a fixed,
nonperiodic direction. If `boundary` is omitted, `p p p` is used.

Fixed boundaries also reduce the fraction of a radial shell that is accessible
to a particle. Lammpalyze applies an analytic finite-window correction. For a
displacement vector `d`, each fixed axis contributes the overlap factor
`1 - abs(d_i) / L_i`; the product of those factors is integrated over the exact
spherical shell. Consequently a uniform ideal gas remains normalized to
`g(r) = 1` for any combination of `p` and `f`. The maximum radius is limited to
half the box length along periodic axes and one box length along fixed axes,
using the most restrictive axis and frame.

This correction removes finite-box geometric bias, but it cannot turn a truly
inhomogeneous confined system into a homogeneous bulk system. If density varies
systematically with distance from a wall or surface, the resulting RDF is a
global, window-corrected pair correlation that mixes those density gradients
with local structure. Boundary-aware RDFs currently assume an orthorhombic box;
triclinic tilt factors are not supported.

## GUI Overview

The GUI contains tabs for common analysis tasks:

- `Species analysis`: plot selected species counts over time.
- `Thermodynamic data`: plot selected thermodynamic parameters, choose
  simulations, edit legend labels, and adjust x/y axis ranges. Existing plots
  update automatically when either range is edited or reset.
- `Radial distribution`: calculate RDF curves from explicit atom-type groups,
  including multiple force-field types for the same element, or from
  mass-weighted molecular centers of mass. Molecule mode is enabled only when
  every selected trajectory contains a `mol` atom-table column. Atom types and
  molecule IDs accept lists such as `1,3,4`; inclusive `*` ranges such as
  `1*11,15,17` are also supported. The panel includes the configured atom-type
  mapping and editable names for both RDF selections. Simulations, timestep
  range, positive sampling frequency, bin width, optional point-based running
  averages, and a second cross-simulation mean and standard-deviation plot
  remain selectable. Selection names form the regular-plot legend label, for
  example `Na+ - PF6-`. Snapshot mode appends each newly calculated selection
  to the existing normalized RDF plot; its average plot is rebuilt from every
  currently displayed curve. With snapshot mode disabled, a new calculation
  replaces the displayed curves. Running-average curves use the inverse color
  of their raw RDF curve for clearer visual separation. Sampling is anchored at
  the entered start timestep, so `400000` through `416000` with frequency
  `1000` uses `400000, 401000, ..., 416000`.

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
- `Atomic index generator`: generate a sorted atom-ID list from atom types or a
  trajectory-provided `mol` column in the first frame. Selection values accept
  inclusive `*` ranges such as `1,3,4*7`, and each matched ID can optionally be
  repeated. The generated list remains editable so individual IDs can be
  removed or changed before copying it into another tab.
- `Geometry`: plot atom, COM, and orthogonal point-to-plane distances or
  three-atom angles, including intramolecular unique-pair expansion and optional
  same-molecule formula/SMILES filtering.
- `Snapshot`: inspect atom membership across nearby analyzed observations or
  list calculated molecules at one entered timestep. Formula/SMILES cells can
  be copied individually.
- `Molecule visualization`: render one or all observed SMILES structures for a
  formula and summarize component charges, ion candidates, and quality flags.
  Formula and SMILES fields accept pasted Snapshot notation, and checked
  structure tiles can be accumulated into copyable Geometry descriptor lists.
- `Reaction paths`: view total and per-simulation reaction path counts, copy a
  selected path, export an inclusive timestep range from any configured
  trajectory, and show every occurrence with simulation, timesteps, and atom
  IDs beneath the reactant/product visualization.
- `Connected pathways`: view connected reaction states by pathway depth in
  chemical formula or SMILES notation, filter by minimum occurrence count, and
  export the visible pathway table as CSV.
- `Pathway graph`: inspect a selected connected pathway as a scrollable,
  top-down reaction graph in formula or SMILES notation. State snapshots are
  rendered in the background with visible progress, using OVITO's Python
  renderer when available and a built-in Matplotlib fallback otherwise.
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
default_bond_order_cutoff,0.5
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
  atomic_indices.py trajectory atom-ID selection and list generation
  chop.py         streaming trajectory and thermo range extraction
  reactions.py    reaction path counting and occurrence lookup
  rdf.py          radial distribution function calculations
  rdf_plotting.py radial distribution plotting and cross-simulation averaging
  structure.py    static structure factor and incoherent scattering calculations
  structure_plotting.py structural-relaxation plotting
  geometry.py     atom, COM, plane, intramolecule, and angle calculations
  geometry_plotting.py geometry distance-and-angle plotting
  snapshot.py     atom and molecule Snapshot table construction
  plotting.py     Matplotlib plotting helpers
  parsers/        species, thermo, computed-data, bond, and trajectory readers
  gui/            Tkinter GUI tabs and application shell
    atomic_indices_tab.py atom-ID generator controls and output
    charge_tab.py trajectory-backed atomic-data plotting tab
    computed_tabs.py pairwise-data and mean-square-displacement tabs
    geometry_tab.py trajectory Geometry tab
    snapshot_tab.py atom and molecule Snapshot tab
    reactions_tab.py reaction tables, range export, occurrence listing, and OVITO controls
    pathway_graph.py connected-pathway graph data and layout helpers
    pathway_graph_tab.py pathway graph controls and background rendering
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
