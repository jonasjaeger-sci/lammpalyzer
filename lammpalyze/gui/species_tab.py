"""Species-analysis tab for the Tkinter GUI."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from lammpalyze.gui.helpers import parse_reference_lines, parse_timestep_values
from lammpalyze.plotting import (
    plot_species,
    simulation_has_species_source,
    species_names_for_source,
)

SPECIES_SOURCE_CHOICES = {
    "LAMMPS species": "lammps",
    "Bond analysis (formula)": "formula",
    "Bond analysis (SMILES)": "smiles",
}


class SpeciesTabMixin:
    """Build and manage the species-analysis tab."""

    def _build_species_tab(self, parent: ttk.Frame) -> None:
        """Create controls and output area for species plotting."""

        controls_container = ttk.Frame(parent)
        controls_container.pack(side="left", fill="y", padx=8, pady=8)
        self._species_controls_canvas = tk.Canvas(controls_container, highlightthickness=0, width=260)
        species_controls_scrollbar = ttk.Scrollbar(
            controls_container,
            orient="vertical",
            command=self._species_controls_canvas.yview,
        )
        controls = ttk.Frame(self._species_controls_canvas)
        self._species_controls_window = self._species_controls_canvas.create_window(
            (0, 0),
            window=controls,
            anchor="nw",
        )
        self._species_controls_canvas.configure(yscrollcommand=species_controls_scrollbar.set)
        controls.bind(
            "<Configure>",
            lambda _event: self._species_controls_canvas.configure(
                scrollregion=self._species_controls_canvas.bbox("all")
            ),
        )
        self._species_controls_canvas.bind(
            "<Configure>",
            lambda event: self._species_controls_canvas.itemconfigure(
                self._species_controls_window,
                width=event.width,
            ),
        )
        self._species_controls_canvas.bind("<Enter>", self._bind_species_controls_mousewheel)
        self._species_controls_canvas.bind("<Leave>", self._unbind_species_controls_mousewheel)
        controls.bind("<Enter>", self._bind_species_controls_mousewheel)
        controls.bind("<Leave>", self._unbind_species_controls_mousewheel)
        self._species_controls_canvas.pack(side="left", fill="y", expand=True)
        species_controls_scrollbar.pack(side="right", fill="y")

        plot_area = ttk.Frame(parent)
        plot_area.pack(side="right", fill="both", expand=True, padx=8, pady=8)
        self._species_scroll_canvas = tk.Canvas(plot_area, highlightthickness=0, background="#0b1020")
        species_scrollbar = ttk.Scrollbar(
            plot_area,
            orient="vertical",
            command=self._species_scroll_canvas.yview,
        )
        self._species_plot_area = ttk.Frame(self._species_scroll_canvas)
        self._species_window = self._species_scroll_canvas.create_window(
            (0, 0),
            window=self._species_plot_area,
            anchor="nw",
        )
        self._species_scroll_canvas.configure(yscrollcommand=species_scrollbar.set)
        self._species_plot_area.bind(
            "<Configure>",
            lambda _event: self._species_scroll_canvas.configure(
                scrollregion=self._species_scroll_canvas.bbox("all")
            ),
        )
        self._species_scroll_canvas.bind(
            "<Configure>",
            lambda event: self._species_scroll_canvas.itemconfigure(self._species_window, width=event.width),
        )
        self._species_scroll_canvas.pack(side="left", fill="both", expand=True)
        species_scrollbar.pack(side="right", fill="y")

        self.species_source = tk.StringVar(value=self._initial_species_source_label())
        ttk.Label(controls, text="Data source").pack(anchor="w")
        species_source_box = ttk.Combobox(
            controls,
            textvariable=self.species_source,
            values=self._available_species_source_labels(),
            state="readonly",
        )
        species_source_box.bind("<<ComboboxSelected>>", lambda _event: self._update_species_source_controls())
        species_source_box.pack(fill="x", pady=(0, 12))

        ttk.Label(controls, text="Simulations").pack(anchor="w")
        self.species_sim_list = tk.Listbox(controls, selectmode="multiple", exportselection=False, height=6)
        self.species_sim_list.pack(fill="x", pady=(0, 12))

        self.species_theme = tk.StringVar(value="Dark")
        ttk.Label(controls, text="Background").pack(anchor="w")
        ttk.Combobox(
            controls,
            textvariable=self.species_theme,
            values=["Dark", "Bright"],
            state="readonly",
        ).pack(fill="x", pady=(0, 12))

        ttk.Label(controls, text="Species").pack(anchor="w")
        species_list_frame = ttk.Frame(controls)
        species_list_frame.pack(fill="both", expand=True, pady=(0, 12))
        self.species_list = tk.Listbox(species_list_frame, selectmode="multiple", exportselection=False, height=18)
        species_list_scrollbar = ttk.Scrollbar(
            species_list_frame,
            orient="vertical",
            command=self.species_list.yview,
        )
        self.species_list.configure(yscrollcommand=species_list_scrollbar.set)
        self.species_list.bind("<<ListboxSelect>>", lambda _event: self._update_species_toggle_label())
        self.species_list.bind("<Enter>", self._unbind_species_controls_mousewheel)
        self.species_list.bind("<Leave>", self._bind_species_controls_mousewheel)
        self.species_list.pack(side="left", fill="both", expand=True)
        species_list_scrollbar.pack(side="right", fill="y")

        self.species_toggle_button = ttk.Button(
            controls,
            text="Deselect all species",
            command=self._toggle_species_selection,
        )
        self.species_toggle_button.pack(fill="x", pady=(0, 8))

        ttk.Label(controls, text="Step range").pack(anchor="w")
        self.species_step_min = tk.StringVar()
        self.species_step_max = tk.StringVar()
        ttk.Label(controls, text="Minimum").pack(anchor="w")
        ttk.Entry(controls, textvariable=self.species_step_min).pack(fill="x", pady=(0, 4))
        ttk.Label(controls, text="Maximum").pack(anchor="w")
        ttk.Entry(controls, textvariable=self.species_step_max).pack(fill="x", pady=(0, 8))
        ttk.Label(controls, text="Exclude timesteps").pack(anchor="w")
        self.species_excluded_timesteps = tk.StringVar()
        ttk.Entry(controls, textvariable=self.species_excluded_timesteps).pack(fill="x", pady=(0, 12))

        ttk.Label(controls, text="Vertical lines").pack(anchor="w")
        self.species_vertical_lines = tk.StringVar()
        ttk.Entry(controls, textvariable=self.species_vertical_lines).pack(fill="x", pady=(0, 8))
        ttk.Label(controls, text="Horizontal lines").pack(anchor="w")
        self.species_horizontal_lines = tk.StringVar()
        ttk.Entry(controls, textvariable=self.species_horizontal_lines).pack(fill="x", pady=(0, 12))

        ttk.Button(controls, text="Plot", command=self._plot_species).pack(fill="x")
        ttk.Button(controls, text="Export PNG", command=self._save_species_plot).pack(fill="x", pady=(8, 0))
        self._species_simulations = []
        self._update_species_source_controls()

    def _plot_species(self) -> None:
        """Plot selected species for selected simulations."""

        try:
            simulations = self._selected_species_simulations()
            selected_species = [self.species_list.get(index) for index in self.species_list.curselection()]
            if not simulations or not selected_species:
                raise ValueError("Select at least one simulation and one species.")
            for canvas in self._species_canvases:
                self._destroy_canvas(canvas)
            self._species_canvases = []
            figures = plot_species(
                simulations,
                selected_species,
                reference_lines=self._species_reference_lines(),
                step_range=self._species_step_range(),
                excluded_timesteps=self._species_excluded_timesteps(),
                theme=self.species_theme.get(),
                data_source=self._selected_species_source(),
                plot_settings=self._plot_settings(),
            )
            for figure in figures:
                canvas = self._create_figure_canvas(figure, self._species_plot_area)
                canvas.get_tk_widget().pack(fill="x", expand=False, pady=(0, 12))
                self._species_canvases.append(canvas)
            self._species_scroll_canvas.yview_moveto(0)
        except Exception as exc:  # pragma: no cover - GUI feedback.
            messagebox.showerror("Species plotting failed", str(exc))

    def _save_species_plot(self) -> None:
        """Save the current species plot to an image file."""

        self._export_canvas_figures_png(
            self._species_canvases,
            "Export species plots",
            "species_analysis.png",
            ["selected_species", "total_molecules"],
        )

    def _selected_species_simulations(self):
        """Return species-capable simulations selected in the species listbox."""

        return [self._species_simulations[index] for index in self.species_sim_list.curselection()]

    def _selected_species_source(self) -> str:
        """Return the normalized species data source selected in the GUI."""

        return SPECIES_SOURCE_CHOICES[self.species_source.get()]

    def _initial_species_source_label(self) -> str:
        """Return the first available species source label."""

        labels = self._available_species_source_labels()
        return labels[0] if labels else "LAMMPS species"

    def _available_species_source_labels(self) -> list[str]:
        """Return data-source labels with at least one available simulation."""

        labels = []
        for label, source in SPECIES_SOURCE_CHOICES.items():
            if any(simulation_has_species_source(simulation, source) for simulation in self.project.simulations):
                labels.append(label)
        return labels

    def _update_species_source_controls(self) -> None:
        """Refresh simulation and species choices for the selected source."""

        if self.species_source.get() not in SPECIES_SOURCE_CHOICES:
            self.species_source.set(self._initial_species_source_label())
        selected_source = self._selected_species_source()
        selected_simulations = {
            self._species_simulations[index].index
            for index in self.species_sim_list.curselection()
            if index < len(self._species_simulations)
        }
        selected_species = {self.species_list.get(index) for index in self.species_list.curselection()}

        self._species_simulations = [
            simulation
            for simulation in self.project.simulations
            if simulation_has_species_source(simulation, selected_source)
        ]
        self.species_sim_list.delete(0, "end")
        for simulation in self._species_simulations:
            self.species_sim_list.insert("end", f"Simulation {simulation.index}")
            if simulation.index in selected_simulations:
                self.species_sim_list.select_set("end")

        species = species_names_for_source(self._species_simulations, selected_source)
        self.species_list.delete(0, "end")
        selected_any_species = False
        for name in species:
            self.species_list.insert("end", name)
            if name in selected_species:
                self.species_list.select_set("end")
                selected_any_species = True
        if species and not selected_any_species:
            self.species_list.select_set(0, "end")
        self._update_species_toggle_label()

    def _species_reference_lines(self) -> tuple[list[float], list[float]]:
        """Return vertical and horizontal reference lines for the species plot."""

        return (
            parse_reference_lines(self.species_vertical_lines.get()),
            parse_reference_lines(self.species_horizontal_lines.get()),
        )

    def _species_step_range(self) -> tuple[float, float] | None:
        """Return the selected species step range, or ``None`` for auto limits."""

        minimum = self.species_step_min.get().strip()
        maximum = self.species_step_max.get().strip()
        if not minimum and not maximum:
            return None
        if not minimum or not maximum:
            raise ValueError("Enter both species step minimum and maximum, or leave both blank.")
        return tuple(sorted((float(minimum), float(maximum))))

    def _species_excluded_timesteps(self) -> list[int]:
        """Return timesteps excluded from species plots."""

        return parse_timestep_values(self.species_excluded_timesteps.get())

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

    def _bind_species_controls_mousewheel(self, _event) -> None:
        """Bind global mouse-wheel scrolling while the pointer is over species controls."""

        self.root.bind_all("<MouseWheel>", self._on_species_controls_mousewheel)
        self.root.bind_all("<Button-4>", self._on_species_controls_mousewheel)
        self.root.bind_all("<Button-5>", self._on_species_controls_mousewheel)

    def _unbind_species_controls_mousewheel(self, _event) -> None:
        """Remove global mouse-wheel bindings for species controls scrolling."""

        self.root.unbind_all("<MouseWheel>")
        self.root.unbind_all("<Button-4>")
        self.root.unbind_all("<Button-5>")

    def _on_species_controls_mousewheel(self, event) -> None:
        """Scroll the species controls canvas from mouse-wheel events."""

        if getattr(event, "num", None) == 4:
            delta = -1
        elif getattr(event, "num", None) == 5:
            delta = 1
        else:
            delta = -1 * int(event.delta / 120)
        self._species_controls_canvas.yview_scroll(delta, "units")
