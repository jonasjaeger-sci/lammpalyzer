"""Species-analysis tab for the Tkinter GUI."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from lammpalyze.plotting import plot_species


class SpeciesTabMixin:
    """Build and manage the species-analysis tab."""

    def _build_species_tab(self, parent: ttk.Frame) -> None:
        """Create controls and output area for species plotting."""

        controls = ttk.Frame(parent)
        controls.pack(side="left", fill="y", padx=8, pady=8)
        plot_area = ttk.Frame(parent)
        plot_area.pack(side="right", fill="both", expand=True, padx=8, pady=8)
        self._species_plot_area = plot_area

        ttk.Label(controls, text="Simulations").pack(anchor="w")
        self.species_sim_list = tk.Listbox(controls, selectmode="multiple", exportselection=False, height=6)
        for simulation in self.project.simulations:
            if simulation.species_df is not None:
                self.species_sim_list.insert("end", f"Simulation {simulation.index}")
        self.species_sim_list.pack(fill="x", pady=(0, 12))

        ttk.Label(controls, text="Species").pack(anchor="w")
        species = sorted(
            {
                name
                for simulation in self.project.simulations
                if simulation.species
                for name in simulation.species
            }
        )
        self.species_list = tk.Listbox(controls, selectmode="multiple", exportselection=False, height=18)
        for name in species:
            self.species_list.insert("end", name)
        if species:
            self.species_list.select_set(0, "end")
        self.species_list.bind("<<ListboxSelect>>", lambda _event: self._update_species_toggle_label())
        self.species_list.pack(fill="both", expand=True, pady=(0, 12))

        self.species_toggle_button = ttk.Button(
            controls,
            text="Deselect all species",
            command=self._toggle_species_selection,
        )
        self.species_toggle_button.pack(fill="x", pady=(0, 8))
        ttk.Button(controls, text="Plot", command=self._plot_species).pack(fill="x")
        ttk.Button(controls, text="Save plot", command=self._save_species_plot).pack(fill="x", pady=(8, 0))
        self._update_species_toggle_label()

    def _plot_species(self) -> None:
        """Plot selected species for selected simulations."""

        try:
            simulations = self._selected_species_simulations()
            selected_species = [self.species_list.get(index) for index in self.species_list.curselection()]
            if not simulations or not selected_species:
                raise ValueError("Select at least one simulation and one species.")
            figure = plot_species(simulations, selected_species)
            self._replace_canvas("_species_canvas", self._species_plot_area, figure)
        except Exception as exc:  # pragma: no cover - GUI feedback.
            messagebox.showerror("Species plotting failed", str(exc))

    def _save_species_plot(self) -> None:
        """Save the current species plot to an image file."""

        self._save_canvas_figure(self._species_canvas, "Save species plot", "species_analysis.png")

    def _selected_species_simulations(self):
        """Return species-capable simulations selected in the species listbox."""

        available = [simulation for simulation in self.project.simulations if simulation.species_df is not None]
        return [available[index] for index in self.species_sim_list.curselection()]

    def _toggle_species_selection(self) -> None:
        """Select or clear all species in the species listbox."""

        if self.species_list.size() == 0:
            return
        if len(self.species_list.curselection()) == self.species_list.size():
            self.species_list.selection_clear(0, "end")
        else:
            self.species_list.select_set(0, "end")
        self._update_species_toggle_label()

    def _update_species_toggle_label(self) -> None:
        """Set the species toggle button text from the current selection."""

        if not hasattr(self, "species_toggle_button"):
            return
        if self.species_list.size() == 0:
            self.species_toggle_button.configure(text="Select all species")
        elif len(self.species_list.curselection()) == self.species_list.size():
            self.species_toggle_button.configure(text="Deselect all species")
        else:
            self.species_toggle_button.configure(text="Select all species")
