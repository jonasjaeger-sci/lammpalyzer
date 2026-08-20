"""Tkinter application shell for lammpalyze."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from lammpalyze.analysis import LammpalyzeProject
from lammpalyze.gui.canvas import CanvasMixin
from lammpalyze.gui.charge_tab import ChargeTabMixin
from lammpalyze.gui.atomic_indices_tab import AtomicIndicesTabMixin
from lammpalyze.gui.computed_tabs import ComputedDataTabMixin
from lammpalyze.gui.geometry_tab import GeometryTabMixin
from lammpalyze.gui.molecule_tab import MoleculeTabMixin
from lammpalyze.gui.pathway_graph_tab import PathwayGraphTabMixin
from lammpalyze.gui.rdf_tab import RdfTabMixin
from lammpalyze.gui.reactions_tab import ReactionTabMixin
from lammpalyze.gui.species_tab import SpeciesTabMixin
from lammpalyze.gui.snapshot_tab import SnapshotTabMixin
from lammpalyze.gui.structure_tab import StructuralRelaxationTabMixin
from lammpalyze.gui.thermo_tab import ThermoTabMixin
from lammpalyze.plotting import PlotSettings


# Tkinter tabs are intentionally separated into focused mixins.
# pylint: disable=too-many-ancestors
class LammpalyzeGUI(
    AtomicIndicesTabMixin,
    SnapshotTabMixin,
    SpeciesTabMixin,
    ThermoTabMixin,
    ComputedDataTabMixin,
    GeometryTabMixin,
    ChargeTabMixin,
    RdfTabMixin,
    StructuralRelaxationTabMixin,
    MoleculeTabMixin,
    PathwayGraphTabMixin,
    ReactionTabMixin,
    CanvasMixin,
):
    """GUI for simulation outputs, molecular structures, and reactions."""

    def __init__(self, project: LammpalyzeProject) -> None:
        """Create the main window and initialize project-backed GUI state."""

        self.project = project
        (
            self._reaction_simulation_indices,
            self._reaction_paths,
            self._reaction_counts_by_simulation,
        ) = project.reaction_path_table()
        self.root = tk.Tk()
        self.root.title("Lammpalyzer")
        self.root.geometry("1100x760")
        self._species_canvases: list[FigureCanvasTkAgg] = []
        self._thermo_canvases: list[FigureCanvasTkAgg] = []
        self._rdf_canvases: list[FigureCanvasTkAgg] = []
        self._charge_canvases: list[FigureCanvasTkAgg] = []
        self._pairwise_canvas: FigureCanvasTkAgg | None = None
        self._geometry_canvas: FigureCanvasTkAgg | None = None
        self._msd_canvases: list[FigureCanvasTkAgg] = []
        self._rdf_timesteps_by_simulation: dict[int, list[int]] = {}
        self._rdf_exact_timesteps_by_simulation: dict[int, list[int]] | None = None
        self._rdf_exact_timestep_entry_values: tuple[str, str, str] | None = None
        self._rdf_snapshot_results = []
        self._structure_canvases: list[FigureCanvasTkAgg] = []
        self._structure_timesteps_by_simulation: dict[int, list[int]] = {}
        self._molecule_photo = None
        self._molecule_smiles: str | None = None
        self._molecule_gallery_mode = "single"
        self._molecule_gallery_photos = []
        self._molecule_gallery_vars = []
        self._reaction_path_gallery_photos = []
        self._reaction_path_gallery_vars = []
        self._reaction_path_reactant_photos = []
        self._reaction_path_reactant_vars = []
        self._reaction_path_product_photos = []
        self._reaction_path_product_vars = []
        self._molecule_image_size: tuple[int, int] | None = None
        self._molecule_resize_job: str | None = None
        self._closed = False
        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<Control-q>", lambda _event: self.close())
        self.root.bind("<Control-Q>", lambda _event: self.close())

    def run(self) -> None:
        """Start the Tkinter event loop."""

        self.root.mainloop()

    def _build(self) -> None:
        """Build the tabbed GUI layout."""

        top_bar = ttk.Frame(self.root)
        top_bar.pack(fill="x", padx=8, pady=(8, 0))
        self._build_plot_settings(top_bar)
        ttk.Button(top_bar, text="Quit", command=self.close).pack(side="right")

        tabs = ttk.Notebook(self.root)
        tabs.pack(fill="both", expand=True, padx=8, pady=8)

        species_tab = ttk.Frame(tabs)
        thermo_tab = ttk.Frame(tabs)
        pairwise_tab = ttk.Frame(tabs)
        geometry_tab = ttk.Frame(tabs)
        atomic_indices_tab = ttk.Frame(tabs)
        snapshot_tab = ttk.Frame(tabs)
        msd_tab = ttk.Frame(tabs)
        rdf_tab = ttk.Frame(tabs)
        structure_tab = ttk.Frame(tabs)
        charge_tab = ttk.Frame(tabs)
        smiles_tab = ttk.Frame(tabs)
        reaction_table_tab = ttk.Frame(tabs)
        connected_pathways_tab = ttk.Frame(tabs)
        pathway_graph_tab = ttk.Frame(tabs)
        reaction_tab = ttk.Frame(tabs)
        tabs.add(species_tab, text="Species analysis")
        tabs.add(thermo_tab, text="Thermodynamic data")
        tabs.add(pairwise_tab, text="Pairwise data")
        tabs.add(geometry_tab, text="Geometry")
        tabs.add(atomic_indices_tab, text="Atomic index generator")
        tabs.add(snapshot_tab, text="Snapshot")
        tabs.add(msd_tab, text="Mean-square displacement")
        tabs.add(rdf_tab, text="Radial distribution")
        tabs.add(structure_tab, text="Structural relaxation")
        tabs.add(charge_tab, text="Atomic data")
        tabs.add(smiles_tab, text="Molecule visualization")
        tabs.add(reaction_table_tab, text="Reaction paths")
        tabs.add(connected_pathways_tab, text="Connected pathways")
        tabs.add(pathway_graph_tab, text="Pathway graph")
        tabs.add(reaction_tab, text="Reaction visualization")

        self._build_species_tab(species_tab)
        self._build_thermo_tab(thermo_tab)
        self._build_pairwise_tab(pairwise_tab)
        self._build_geometry_tab(geometry_tab)
        self._build_atomic_indices_tab(atomic_indices_tab)
        self._build_snapshot_tab(snapshot_tab)
        self._build_msd_tab(msd_tab)
        self._build_rdf_tab(rdf_tab)
        self._build_structure_tab(structure_tab)
        self._build_charge_tab(charge_tab)
        self._build_smiles_tab(smiles_tab)
        self._build_reaction_table_tab(reaction_table_tab)
        self._build_connected_pathways_tab(connected_pathways_tab)
        self._build_pathway_graph_tab(pathway_graph_tab)
        self._build_reaction_tab(reaction_tab)

    def _build_plot_settings(self, parent: ttk.Frame) -> None:
        """Build shared plot display settings."""

        settings = ttk.LabelFrame(parent, text="Plot settings")
        settings.pack(side="left", fill="x", expand=True, padx=(0, 8))

        self.plot_x_axis_mode = tk.StringVar(value="Timesteps")
        ttk.Label(settings, text="X-axis").pack(side="left", padx=(8, 4))
        ttk.Combobox(
            settings,
            textvariable=self.plot_x_axis_mode,
            values=["Timesteps", "Real time"],
            state="readonly",
            width=11,
        ).pack(side="left", padx=(0, 8))

        self.plot_timestep_size_fs = tk.StringVar(value="0.5")
        ttk.Label(settings, text="Step fs").pack(side="left", padx=(0, 4))
        ttk.Entry(settings, textvariable=self.plot_timestep_size_fs, width=7).pack(side="left", padx=(0, 8))

        self.plot_time_unit = tk.StringVar(value="ps")
        ttk.Label(settings, text="Unit").pack(side="left", padx=(0, 4))
        ttk.Combobox(
            settings,
            textvariable=self.plot_time_unit,
            values=["fs", "ps", "ns", "us", "ms", "s"],
            state="readonly",
            width=5,
        ).pack(side="left", padx=(0, 8))

        self.plot_reset_x_origin = tk.BooleanVar(value=False)
        self.plot_log_x = tk.BooleanVar(value=False)
        self.plot_log_y = tk.BooleanVar(value=False)
        ttk.Checkbutton(settings, text="Reset x=0", variable=self.plot_reset_x_origin).pack(
            side="left",
            padx=(0, 6),
        )
        ttk.Checkbutton(settings, text="Log x", variable=self.plot_log_x).pack(side="left", padx=(0, 6))
        ttk.Checkbutton(settings, text="Log y", variable=self.plot_log_y).pack(side="left", padx=(0, 8))

    def _plot_settings(self) -> PlotSettings:
        """Return validated shared plot display settings."""

        return PlotSettings(
            x_axis="time" if self.plot_x_axis_mode.get() == "Real time" else "timestep",
            timestep_size_fs=float(self.plot_timestep_size_fs.get()),
            time_unit=self.plot_time_unit.get(),
            reset_x_origin=self.plot_reset_x_origin.get(),
            log_x=self.plot_log_x.get(),
            log_y=self.plot_log_y.get(),
        )

    def close(self) -> None:
        """Close the GUI and release Matplotlib/Tk resources promptly."""

        if self._closed:
            return
        self._closed = True
        self._close_atomic_plot_worker()
        self._close_thermo_axis_update()
        self._close_pathway_graph_tab()

        for canvas in self._species_canvases:
            self._destroy_canvas(canvas)
        self._species_canvases = []

        for canvas in self._rdf_canvases:
            self._destroy_canvas(canvas)
        self._rdf_canvases = []

        for canvas in self._structure_canvases:
            self._destroy_canvas(canvas)
        self._structure_canvases = []

        for canvas in self._charge_canvases:
            self._destroy_canvas(canvas)
        self._charge_canvases = []

        if self._pairwise_canvas is not None:
            self._destroy_canvas(self._pairwise_canvas)
            self._pairwise_canvas = None

        if self._geometry_canvas is not None:
            self._destroy_canvas(self._geometry_canvas)
            self._geometry_canvas = None

        for canvas in self._msd_canvases:
            self._destroy_canvas(canvas)
        self._msd_canvases = []

        for canvas in self._thermo_canvases:
            self._destroy_canvas(canvas)
        self._thermo_canvases = []

        if self._molecule_resize_job is not None:
            self.root.after_cancel(self._molecule_resize_job)
        self._molecule_resize_job = None
        self._molecule_smiles = None
        self._molecule_gallery_photos = []
        self._molecule_gallery_vars = []
        self._reaction_path_gallery_photos = []
        self._reaction_path_gallery_vars = []
        self._reaction_path_reactant_photos = []
        self._reaction_path_reactant_vars = []
        self._reaction_path_product_photos = []
        self._reaction_path_product_vars = []
        self._molecule_image_size = None
        self._molecule_photo = None
        plt.close("all")

        try:
            self.root.quit()
            self.root.destroy()
        except tk.TclError:
            pass


def launch_gui(project: LammpalyzeProject) -> None:
    """Launch the lammpalyze GUI."""

    gui = LammpalyzeGUI(project)
    try:
        gui.run()
    finally:
        gui.close()
