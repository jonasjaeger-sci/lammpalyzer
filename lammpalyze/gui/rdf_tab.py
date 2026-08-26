"""Radial-distribution tab for the Tkinter GUI."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from lammpalyze.analysis import LoadedSimulation
from lammpalyze.gui.helpers import LEGEND_PLACEMENTS, parse_reference_lines
from lammpalyze.parsers import trajectory_atom_columns
from lammpalyze.rdf import RDFResult, compute_rdf, parse_rdf_ids
from lammpalyze.rdf_plotting import plot_rdf


def rdf_snapshot_results(
    existing: list[RDFResult],
    new_results: list[RDFResult],
    snapshot_enabled: bool,
) -> list[RDFResult]:
    """Append new RDF curves in snapshot mode, otherwise replace the plot."""

    if snapshot_enabled:
        return [*existing, *new_results]
    return list(new_results)


class RdfTabMixin:
    """Build and manage the radial-distribution tab."""

    def _build_rdf_tab(self, parent: ttk.Frame) -> None:
        """Create controls and output area for RDF plotting."""

        controls_container = ttk.Frame(parent)
        controls_container.pack(side="left", fill="y", padx=8, pady=8)
        controls_canvas = tk.Canvas(controls_container, highlightthickness=0, width=275)
        controls_scrollbar = ttk.Scrollbar(
            controls_container,
            orient="vertical",
            command=controls_canvas.yview,
        )
        controls = ttk.Frame(controls_canvas)
        controls_window = controls_canvas.create_window((0, 0), window=controls, anchor="nw")
        controls_canvas.configure(yscrollcommand=controls_scrollbar.set)
        controls.bind(
            "<Configure>",
            lambda _event: controls_canvas.configure(scrollregion=controls_canvas.bbox("all")),
        )
        controls_canvas.bind(
            "<Configure>",
            lambda event: controls_canvas.itemconfigure(controls_window, width=event.width),
        )
        controls_canvas.pack(side="left", fill="y", expand=True)
        controls_scrollbar.pack(side="right", fill="y")

        plot_container = ttk.Frame(parent)
        plot_container.pack(side="right", fill="both", expand=True, padx=8, pady=8)
        self._rdf_scroll_canvas = tk.Canvas(
            plot_container,
            highlightthickness=0,
            background="#0b1020",
        )
        plot_scrollbar = ttk.Scrollbar(
            plot_container,
            orient="vertical",
            command=self._rdf_scroll_canvas.yview,
        )
        self._rdf_plot_area = ttk.Frame(self._rdf_scroll_canvas)
        plot_window = self._rdf_scroll_canvas.create_window(
            (0, 0),
            window=self._rdf_plot_area,
            anchor="nw",
        )
        self._rdf_scroll_canvas.configure(yscrollcommand=plot_scrollbar.set)
        self._rdf_plot_area.bind(
            "<Configure>",
            lambda _event: self._rdf_scroll_canvas.configure(
                scrollregion=self._rdf_scroll_canvas.bbox("all")
            ),
        )
        self._rdf_scroll_canvas.bind(
            "<Configure>",
            lambda event: self._rdf_scroll_canvas.itemconfigure(plot_window, width=event.width),
        )
        self._rdf_scroll_canvas.pack(side="left", fill="both", expand=True)
        plot_scrollbar.pack(side="right", fill="y")

        ttk.Label(controls, text="Simulations").pack(anchor="w")
        self.rdf_sim_list = tk.Listbox(controls, selectmode="multiple", exportselection=False, height=6)
        self._rdf_simulations = [
            simulation
            for simulation in self.project.simulations
            if simulation.trajectory_path is not None and simulation.type_to_element is not None
        ]
        for simulation in self._rdf_simulations:
            self.rdf_sim_list.insert("end", f"Simulation {simulation.index}")
        if self.rdf_sim_list.size():
            self.rdf_sim_list.select_set(0, "end")
        self.rdf_sim_list.bind(
            "<<ListboxSelect>>",
            lambda _event: self._rdf_simulation_selection_changed(),
        )
        self.rdf_sim_list.pack(fill="x", pady=(0, 12))

        elements = self.project.config.element_list
        default_a = "Li" if "Li" in elements else (elements[0] if elements else "")
        default_b = "O" if "O" in elements else (elements[1] if len(elements) > 1 else default_a)
        type_to_element = self.project.config.type_to_element
        default_types_a = [
            atom_type
            for atom_type, element in type_to_element.items()
            if element == default_a
        ]
        default_types_b = [
            atom_type
            for atom_type, element in type_to_element.items()
            if element == default_b
        ]

        particle_selection = ttk.LabelFrame(controls, text="RDF particles", padding=6)
        particle_selection.pack(fill="x", pady=(0, 12))
        self.rdf_particle_mode = tk.StringVar(value="atom")
        self.rdf_atom_mode_button = ttk.Radiobutton(
            particle_selection,
            text="Atom types",
            variable=self.rdf_particle_mode,
            value="atom",
            command=self._update_rdf_particle_controls,
        )
        self.rdf_atom_mode_button.pack(anchor="w")
        self.rdf_molecule_mode_button = ttk.Radiobutton(
            particle_selection,
            text="Molecule centers of mass",
            variable=self.rdf_particle_mode,
            value="molecule",
            command=self._update_rdf_particle_controls,
        )
        self.rdf_molecule_mode_button.pack(anchor="w", pady=(0, 6))

        self.rdf_atom_types_a = tk.StringVar(
            value=",".join(str(atom_type) for atom_type in default_types_a)
        )
        self.rdf_atom_types_b = tk.StringVar(
            value=",".join(str(atom_type) for atom_type in default_types_b)
        )
        self.rdf_atom_name_a = tk.StringVar(value=default_a)
        self.rdf_atom_name_b = tk.StringVar(value=default_b)
        self.rdf_atom_fields = self._build_rdf_particle_fields(
            particle_selection,
            "Atom types",
            self.rdf_atom_types_a,
            self.rdf_atom_types_b,
            self.rdf_atom_name_a,
            self.rdf_atom_name_b,
        )

        self.rdf_molecule_ids_a = tk.StringVar()
        self.rdf_molecule_ids_b = tk.StringVar()
        self.rdf_molecule_name_a = tk.StringVar(value="Molecule A")
        self.rdf_molecule_name_b = tk.StringVar(value="Molecule B")
        self.rdf_molecule_fields = self._build_rdf_particle_fields(
            particle_selection,
            "Molecule IDs",
            self.rdf_molecule_ids_a,
            self.rdf_molecule_ids_b,
            self.rdf_molecule_name_a,
            self.rdf_molecule_name_b,
        )
        self.rdf_molecule_hint = ttk.Label(
            self.rdf_molecule_fields,
            text="Use commas or inclusive ranges, e.g. 1*11,15,17.",
            wraplength=240,
        )
        self.rdf_molecule_hint.grid(row=3, column=0, columnspan=3, sticky="w", pady=(3, 0))

        self.rdf_molecule_availability = ttk.Label(
            particle_selection,
            text="",
            wraplength=240,
        )
        self.rdf_molecule_availability.pack(fill="x", pady=(0, 6))

        self.rdf_type_table_frame = ttk.Frame(particle_selection)
        self.rdf_type_table_frame.pack(fill="x")
        ttk.Label(self.rdf_type_table_frame, text="Atom-type mapping").pack(anchor="w")
        table_body = ttk.Frame(self.rdf_type_table_frame)
        table_body.pack(fill="x")
        self.rdf_type_table = ttk.Treeview(
            table_body,
            columns=("type", "element"),
            show="headings",
            height=min(8, max(1, len(type_to_element))),
        )
        self.rdf_type_table.heading("type", text="Atom type")
        self.rdf_type_table.heading("element", text="Element")
        self.rdf_type_table.column("type", width=80, anchor="center", stretch=False)
        self.rdf_type_table.column("element", width=115, anchor="w")
        for atom_type, element in sorted(type_to_element.items()):
            self.rdf_type_table.insert("", "end", values=(atom_type, element))
        type_scrollbar = ttk.Scrollbar(
            table_body,
            orient="vertical",
            command=self.rdf_type_table.yview,
        )
        self.rdf_type_table.configure(yscrollcommand=type_scrollbar.set)
        self.rdf_type_table.pack(side="left", fill="x", expand=True)
        type_scrollbar.pack(side="right", fill="y")

        self.rdf_timestep_start = tk.StringVar()
        self.rdf_timestep_end = tk.StringVar()
        ttk.Label(controls, text="Timestep start").pack(anchor="w")
        start_entry = ttk.Entry(controls, textvariable=self.rdf_timestep_start)
        start_entry.bind("<KeyRelease>", lambda _event: self._clear_rdf_exact_timesteps())
        start_entry.pack(fill="x", pady=(0, 8))
        ttk.Label(controls, text="Timestep end").pack(anchor="w")
        end_entry = ttk.Entry(controls, textvariable=self.rdf_timestep_end)
        end_entry.bind("<KeyRelease>", lambda _event: self._clear_rdf_exact_timesteps())
        end_entry.pack(fill="x", pady=(0, 8))
        self.rdf_sampling_frequency = tk.StringVar(value="1")
        ttk.Label(controls, text="Sampling frequency [timesteps]").pack(anchor="w")
        sampling_entry = ttk.Entry(controls, textvariable=self.rdf_sampling_frequency)
        sampling_entry.bind("<KeyRelease>", lambda _event: self._clear_rdf_exact_timesteps())
        sampling_entry.pack(fill="x", pady=(0, 8))
        ttk.Button(controls, text="Last 5 timesteps", command=self._set_rdf_last_timesteps).pack(
            fill="x", pady=(0, 12)
        )

        self.rdf_bin_width = tk.StringVar(value="0.1")
        ttk.Label(controls, text="Bin width").pack(anchor="w")
        ttk.Entry(controls, textvariable=self.rdf_bin_width).pack(fill="x", pady=(0, 12))

        self.rdf_running_average_enabled = tk.BooleanVar(value=False)
        self.rdf_running_average_points = tk.StringVar(value="10")
        ttk.Checkbutton(
            controls,
            text="Show running average",
            variable=self.rdf_running_average_enabled,
        ).pack(anchor="w", pady=(0, 4))
        ttk.Label(controls, text="Average points").pack(anchor="w")
        ttk.Spinbox(
            controls,
            from_=1,
            to=1000000,
            textvariable=self.rdf_running_average_points,
        ).pack(fill="x", pady=(0, 12))

        self.rdf_theme = tk.StringVar(value="Dark")
        ttk.Label(controls, text="Background").pack(anchor="w")
        ttk.Combobox(
            controls,
            textvariable=self.rdf_theme,
            values=["Dark", "Bright"],
            state="readonly",
        ).pack(fill="x", pady=(0, 12))

        self.rdf_legend_location = tk.StringVar(value="None")
        ttk.Label(controls, text="Legend placement").pack(anchor="w")
        ttk.Combobox(
            controls,
            textvariable=self.rdf_legend_location,
            values=LEGEND_PLACEMENTS,
            state="readonly",
        ).pack(fill="x", pady=(0, 12))

        self.rdf_gradient_enabled = tk.BooleanVar(value=False)
        self.rdf_gradient_start = tk.StringVar(value="#f9c74f")
        self.rdf_gradient_end = tk.StringVar(value="#7209b7")
        ttk.Checkbutton(
            controls,
            text="Use gradient colors",
            variable=self.rdf_gradient_enabled,
        ).pack(anchor="w", pady=(0, 4))
        ttk.Label(controls, text="Gradient start").pack(anchor="w")
        ttk.Entry(controls, textvariable=self.rdf_gradient_start).pack(fill="x", pady=(0, 4))
        ttk.Label(controls, text="Gradient end").pack(anchor="w")
        ttk.Entry(controls, textvariable=self.rdf_gradient_end).pack(fill="x", pady=(0, 12))

        ttk.Label(controls, text="Vertical lines").pack(anchor="w")
        self.rdf_vertical_lines = tk.StringVar()
        ttk.Entry(controls, textvariable=self.rdf_vertical_lines).pack(fill="x", pady=(0, 8))
        ttk.Label(controls, text="Horizontal lines").pack(anchor="w")
        self.rdf_horizontal_lines = tk.StringVar()
        ttk.Entry(controls, textvariable=self.rdf_horizontal_lines).pack(fill="x", pady=(0, 12))

        self.rdf_snapshot_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            controls,
            text="Snapshot mode: append new curves",
            variable=self.rdf_snapshot_enabled,
        ).pack(anchor="w", pady=(0, 8))
        ttk.Button(controls, text="Plot", command=self._plot_rdf).pack(fill="x")
        ttk.Button(controls, text="Export PNG", command=self._save_rdf_plot).pack(fill="x", pady=(8, 0))

        self.rdf_status = ttk.Label(self._rdf_plot_area, text="", wraplength=620, justify="left")
        self.rdf_status.pack(anchor="nw", padx=8, pady=8)
        self._set_rdf_last_timesteps()
        self._update_rdf_molecule_availability()
        self._update_rdf_particle_controls()

    def _plot_rdf(self) -> None:
        """Compute and plot RDF curves from the selected GUI values."""

        try:
            simulations = self._selected_rdf_simulations()
            if not simulations:
                raise ValueError("Select at least one simulation.")
            name_a, name_b, selection_arguments = self._rdf_particle_selection()
            start = int(self.rdf_timestep_start.get())
            end = int(self.rdf_timestep_end.get())
            sampling_frequency = int(self.rdf_sampling_frequency.get())
            if sampling_frequency <= 0:
                raise ValueError("RDF sampling frequency must be a positive integer.")
            bin_width = float(self.rdf_bin_width.get())

            results = compute_rdf(
                simulations,
                name_a,
                name_b,
                (start, end),
                bin_width,
                timesteps_by_simulation=self._rdf_exact_timesteps_for_plot(simulations),
                sampling_frequency=sampling_frequency,
                **selection_arguments,
            )
            displayed_results = rdf_snapshot_results(
                self._rdf_snapshot_results,
                results,
                self.rdf_snapshot_enabled.get(),
            )
            figures = plot_rdf(
                displayed_results,
                name_a,
                name_b,
                reference_lines=self._rdf_reference_lines(),
                running_average_points=self._rdf_running_average_points(),
                legend_location=self.rdf_legend_location.get(),
                theme=self.rdf_theme.get(),
                gradient_colors=self._rdf_gradient_colors(),
                plot_settings=self._plot_settings(),
            )
            self._rdf_snapshot_results = displayed_results
            for canvas in self._rdf_canvases:
                self._destroy_canvas(canvas)
            self._rdf_canvases = []
            for figure in figures:
                canvas = self._create_figure_canvas(figure, self._rdf_plot_area)
                canvas.get_tk_widget().pack(fill="x", expand=False, pady=(0, 12))
                self._rdf_canvases.append(canvas)
            self._rdf_scroll_canvas.configure(
                background="#f8fafc" if self.rdf_theme.get() == "Bright" else "#0b1020"
            )
            self._rdf_scroll_canvas.yview_moveto(0)
            self.rdf_status.configure(text=self._rdf_status_text(displayed_results))
        except Exception as exc:  # pragma: no cover - GUI feedback.
            messagebox.showerror("RDF plotting failed", str(exc))

    def _save_rdf_plot(self) -> None:
        """Save the current RDF plot to an image file."""

        self._export_canvas_figures_png(
            self._rdf_canvases,
            "Export RDF plots",
            "radial_distribution.png",
            ["selected_simulations", "average"],
        )

    def _selected_rdf_simulations(self):
        """Return trajectory-capable simulations selected in the RDF listbox."""

        return [self._rdf_simulations[index] for index in self.rdf_sim_list.curselection()]

    def _build_rdf_particle_fields(
        self,
        parent,
        selector_label: str,
        selector_a,
        selector_b,
        name_a,
        name_b,
    ):
        """Build paired selector/name entries for atom or molecule RDFs."""

        fields = ttk.Frame(parent)
        ttk.Label(fields, text=selector_label).grid(row=0, column=1, sticky="w")
        ttk.Label(fields, text="Name").grid(row=0, column=2, sticky="w")
        for row, label, selector, name in (
            (1, "A", selector_a, name_a),
            (2, "B", selector_b, name_b),
        ):
            ttk.Label(fields, text=label).grid(row=row, column=0, sticky="w", padx=(0, 4))
            ttk.Entry(fields, textvariable=selector, width=16).grid(
                row=row,
                column=1,
                sticky="ew",
                padx=(0, 4),
                pady=2,
            )
            ttk.Entry(fields, textvariable=name, width=11).grid(
                row=row,
                column=2,
                sticky="ew",
                pady=2,
            )
        fields.columnconfigure(1, weight=2)
        fields.columnconfigure(2, weight=1)
        return fields

    def _rdf_simulation_selection_changed(self) -> None:
        """Refresh timestep defaults and molecule-mode availability."""

        self._set_rdf_last_timesteps()
        self._update_rdf_molecule_availability()

    def _update_rdf_molecule_availability(self) -> None:
        """Enable molecule RDF only when every selected trajectory has ``mol``."""

        simulations = self._selected_rdf_simulations()
        unavailable = [
            simulation.index
            for simulation in simulations
            if not self._rdf_trajectory_has_molecule_ids(simulation)
        ]
        molecule_mode_available = bool(simulations) and not unavailable
        self.rdf_molecule_mode_button.configure(
            state="normal" if molecule_mode_available else "disabled"
        )
        if unavailable:
            indexes = ", ".join(str(index) for index in unavailable)
            availability = f"Molecule RDF unavailable: simulation(s) {indexes} lack a mol column."
        elif not simulations:
            availability = "Select a simulation to check molecule-ID availability."
        else:
            availability = "Molecule RDF available: all selected trajectories contain mol."
        self.rdf_molecule_availability.configure(text=availability)
        if not molecule_mode_available and self.rdf_particle_mode.get() == "molecule":
            self.rdf_particle_mode.set("atom")
        self._update_rdf_particle_controls()

    def _rdf_trajectory_has_molecule_ids(self, simulation: LoadedSimulation) -> bool:
        """Return cached first-frame ``mol`` column availability."""

        if not hasattr(self, "_rdf_mol_column_by_simulation"):
            self._rdf_mol_column_by_simulation = {}
        if simulation.index not in self._rdf_mol_column_by_simulation:
            if simulation.trajectory_path is None:
                has_mol = False
            else:
                has_mol = "mol" in trajectory_atom_columns(simulation.trajectory_path)
            self._rdf_mol_column_by_simulation[simulation.index] = has_mol
        return self._rdf_mol_column_by_simulation[simulation.index]

    def _update_rdf_particle_controls(self) -> None:
        """Show the selector fields matching the current RDF particle mode."""

        if not hasattr(self, "rdf_atom_fields"):
            return
        self.rdf_atom_fields.pack_forget()
        self.rdf_molecule_fields.pack_forget()
        selected_fields = (
            self.rdf_molecule_fields
            if self.rdf_particle_mode.get() == "molecule"
            else self.rdf_atom_fields
        )
        selected_fields.pack(fill="x", pady=(0, 6), before=self.rdf_molecule_availability)

    def _rdf_particle_selection(self) -> tuple[str, str, dict]:
        """Validate GUI particle fields and return RDF calculation arguments."""

        if self.rdf_particle_mode.get() == "molecule":
            self._update_rdf_molecule_availability()
            if self.rdf_particle_mode.get() != "molecule":
                raise ValueError("Molecule RDF requires a mol column in every selected trajectory.")
            name_a = self.rdf_molecule_name_a.get().strip()
            name_b = self.rdf_molecule_name_b.get().strip()
            arguments = {
                "molecule_ids_a": parse_rdf_ids(self.rdf_molecule_ids_a.get()),
                "molecule_ids_b": parse_rdf_ids(self.rdf_molecule_ids_b.get()),
            }
        else:
            name_a = self.rdf_atom_name_a.get().strip()
            name_b = self.rdf_atom_name_b.get().strip()
            atom_types_a = parse_rdf_ids(self.rdf_atom_types_a.get())
            atom_types_b = parse_rdf_ids(self.rdf_atom_types_b.get())
            known_types = set(self.project.config.type_to_element)
            unknown_types = sorted((set(atom_types_a) | set(atom_types_b)) - known_types)
            if unknown_types:
                values = ", ".join(str(atom_type) for atom_type in unknown_types)
                raise ValueError(f"Unknown RDF atom type(s): {values}.")
            arguments = {
                "atom_types_a": atom_types_a,
                "atom_types_b": atom_types_b,
            }
        if not name_a or not name_b:
            raise ValueError("Enter a name for both RDF particle selections.")
        return name_a, name_b, arguments

    def _rdf_reference_lines(self) -> tuple[list[float], list[float]]:
        """Return vertical and horizontal reference lines for the RDF plot."""

        return (
            parse_reference_lines(self.rdf_vertical_lines.get()),
            parse_reference_lines(self.rdf_horizontal_lines.get()),
        )

    def _rdf_running_average_points(self) -> int | None:
        """Return the optional point window for smoothing selected RDF curves."""

        if not self.rdf_running_average_enabled.get():
            return None
        points = int(self.rdf_running_average_points.get())
        if points < 1:
            raise ValueError("RDF running-average points must be at least 1.")
        return points

    def _rdf_gradient_colors(self) -> tuple[str, str] | None:
        """Return the optional RDF line-color gradient."""

        if not self.rdf_gradient_enabled.get():
            return None
        start = self.rdf_gradient_start.get().strip()
        end = self.rdf_gradient_end.get().strip()
        if not start or not end:
            raise ValueError("Enter both RDF gradient start and end colors.")
        return start, end

    def _set_rdf_last_timesteps(self) -> None:
        """Populate RDF timestep entries from the last five steps in each simulation."""

        selected_by_simulation = {}
        for simulation in self._selected_rdf_simulations():
            timesteps = self._rdf_timesteps(simulation)
            if timesteps:
                selected_by_simulation[simulation.index] = timesteps[-5:]
        if not selected_by_simulation:
            self._rdf_exact_timesteps_by_simulation = None
            self._rdf_exact_timestep_entry_values = None
            self.rdf_timestep_start.set("")
            self.rdf_timestep_end.set("")
            return
        selected = sorted(
            timestep
            for timesteps in selected_by_simulation.values()
            for timestep in timesteps
        )
        start = str(selected[0])
        end = str(selected[-1])
        self._rdf_exact_timesteps_by_simulation = selected_by_simulation
        self._rdf_exact_timestep_entry_values = (
            start,
            end,
            self.rdf_sampling_frequency.get(),
        )
        self.rdf_timestep_start.set(start)
        self.rdf_timestep_end.set(end)

    def _clear_rdf_exact_timesteps(self) -> None:
        """Switch RDF plotting back to the inclusive range in the entry fields."""

        self._rdf_exact_timesteps_by_simulation = None
        self._rdf_exact_timestep_entry_values = None

    def _rdf_exact_timesteps_for_plot(
        self,
        simulations: list[LoadedSimulation],
    ) -> dict[int, list[int]] | None:
        """Return exact per-simulation timesteps when the Last-5 fields still match."""

        if self._rdf_exact_timesteps_by_simulation is None:
            return None
        if self._rdf_exact_timestep_entry_values != (
            self.rdf_timestep_start.get(),
            self.rdf_timestep_end.get(),
            self.rdf_sampling_frequency.get(),
        ):
            self._clear_rdf_exact_timesteps()
            return None
        selected_indexes = {simulation.index for simulation in simulations}
        exact_indexes = set(self._rdf_exact_timesteps_by_simulation)
        if selected_indexes != exact_indexes:
            self._clear_rdf_exact_timesteps()
            return None
        return self._rdf_exact_timesteps_by_simulation

    def _rdf_status_text(self, results) -> str:
        """Summarize the timesteps that contributed to the plotted RDF curves."""

        parts = []
        for result in results:
            timesteps = result.timesteps
            selection = result.label or "RDF"
            parts.append(
                f"{selection}, simulation {result.simulation_index}: "
                f"{len(timesteps)} timestep(s), "
                f"{timesteps[0]} to {timesteps[-1]}"
            )
        return "Used " + "; ".join(parts)

    def _rdf_timesteps(self, simulation: LoadedSimulation) -> list[int]:
        """Return cached trajectory timesteps for one simulation."""

        if simulation.index not in self._rdf_timesteps_by_simulation:
            self._rdf_timesteps_by_simulation[simulation.index] = simulation.trajectory_timesteps()
        return self._rdf_timesteps_by_simulation[simulation.index]
