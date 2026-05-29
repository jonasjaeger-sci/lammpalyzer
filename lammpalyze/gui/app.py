"""Tkinter application shell for lammpalyze."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from lammpalyze.analysis import LammpalyzeProject
from lammpalyze.gui.canvas import CanvasMixin
from lammpalyze.gui.molecule_tab import MoleculeTabMixin
from lammpalyze.gui.rdf_tab import RdfTabMixin
from lammpalyze.gui.reactions_tab import ReactionTabMixin
from lammpalyze.gui.species_tab import SpeciesTabMixin
from lammpalyze.gui.thermo_tab import ThermoTabMixin


class LammpalyzeGUI(
    SpeciesTabMixin,
    ThermoTabMixin,
    RdfTabMixin,
    MoleculeTabMixin,
    ReactionTabMixin,
    CanvasMixin,
):
    """GUI for species, thermo, SMILES, and reaction visualization."""

    def __init__(self, project: LammpalyzeProject) -> None:
        """Create the main window and initialize project-backed GUI state."""

        self.project = project
        (
            self._reaction_simulation_indices,
            self._reaction_paths,
            self._reaction_counts_by_simulation,
        ) = project.reaction_path_table()
        self.root = tk.Tk()
        self.root.title("lammpalyze")
        self.root.geometry("1100x760")
        self._species_canvas: FigureCanvasTkAgg | None = None
        self._thermo_canvases: list[FigureCanvasTkAgg] = []
        self._rdf_canvas: FigureCanvasTkAgg | None = None
        self._rdf_timesteps_by_simulation: dict[int, list[int]] = {}
        self._molecule_photo = None
        self._molecule_smiles: str | None = None
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
        ttk.Button(top_bar, text="Quit", command=self.close).pack(side="right")

        tabs = ttk.Notebook(self.root)
        tabs.pack(fill="both", expand=True, padx=8, pady=8)

        species_tab = ttk.Frame(tabs)
        thermo_tab = ttk.Frame(tabs)
        rdf_tab = ttk.Frame(tabs)
        smiles_tab = ttk.Frame(tabs)
        reaction_table_tab = ttk.Frame(tabs)
        reaction_tab = ttk.Frame(tabs)
        tabs.add(species_tab, text="Species analysis")
        tabs.add(thermo_tab, text="Thermodynamic data")
        tabs.add(rdf_tab, text="Radial distribution")
        tabs.add(smiles_tab, text="Molecule visualization")
        tabs.add(reaction_table_tab, text="Reaction paths")
        tabs.add(reaction_tab, text="Reaction visualization")

        self._build_species_tab(species_tab)
        self._build_thermo_tab(thermo_tab)
        self._build_rdf_tab(rdf_tab)
        self._build_smiles_tab(smiles_tab)
        self._build_reaction_table_tab(reaction_table_tab)
        self._build_reaction_tab(reaction_tab)

    def close(self) -> None:
        """Close the GUI and release Matplotlib/Tk resources promptly."""

        if self._closed:
            return
        self._closed = True

        if self._species_canvas is not None:
            self._destroy_canvas(self._species_canvas)
            self._species_canvas = None

        if self._rdf_canvas is not None:
            self._destroy_canvas(self._rdf_canvas)
            self._rdf_canvas = None

        for canvas in self._thermo_canvases:
            self._destroy_canvas(canvas)
        self._thermo_canvases = []

        if self._molecule_resize_job is not None:
            self.root.after_cancel(self._molecule_resize_job)
            self._molecule_resize_job = None
        self._molecule_smiles = None
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
