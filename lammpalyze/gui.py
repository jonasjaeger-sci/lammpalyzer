"""Tkinter graphical interface for lammpalyze projects."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from lammpalyze.analysis import LammpalyzeProject, LoadedSimulation
from lammpalyze.ovito import OvitoNotAvailableError, create_reaction_scene, launch_ovito_scene, normalize_reaction_path
from lammpalyze.parsers import list_lammpstrj_timesteps
from lammpalyze.plotting import plot_rdf, plot_species, plot_thermo
from lammpalyze.rdf import compute_rdf
from lammpalyze.reactions import ReactionPath, count_reaction_paths
from lammpalyze.smiles import formulas_for_simulation, molecule_photo_image, smiles_for_formula


THERMO_DEFAULTS = ["Temp", "PotEng", "KinEng", "Press", "Volume", "Density"]


def reaction_path_table_data(
    simulations: list[LoadedSimulation],
) -> tuple[list[int], list[ReactionPath], dict[str, dict[int, int]]]:
    """Return simulation indexes, total paths, and per-simulation counts."""

    simulation_indices = []
    counts_by_reaction: dict[str, dict[int, int]] = {}
    totals: dict[str, int] = {}
    for simulation in simulations:
        if simulation.smiles is None or simulation.smiles_id is None:
            continue
        simulation_indices.append(simulation.index)
        for path in count_reaction_paths(simulation.smiles, simulation.smiles_id):
            counts_by_reaction.setdefault(path.reaction, {})[simulation.index] = path.count
            totals[path.reaction] = totals.get(path.reaction, 0) + path.count

    paths = [
        ReactionPath(reaction, count)
        for reaction, count in sorted(totals.items(), key=lambda item: item[1], reverse=True)
    ]
    return simulation_indices, paths, counts_by_reaction


class LammpalyzeGUI:
    """GUI for species, thermo, SMILES, and reaction visualization."""

    def __init__(self, project: LammpalyzeProject) -> None:
        self.project = project
        (
            self._reaction_simulation_indices,
            self._reaction_paths,
            self._reaction_counts_by_simulation,
        ) = reaction_path_table_data(project.simulations)
        self.root = tk.Tk()
        self.root.title("lammpalyze")
        self.root.geometry("1100x760")
        self._species_canvas: FigureCanvasTkAgg | None = None
        self._thermo_canvases: list[FigureCanvasTkAgg] = []
        self._rdf_canvas: FigureCanvasTkAgg | None = None
        self._rdf_timesteps_by_simulation: dict[int, list[int]] = {}
        self._molecule_photo = None
        self._closed = False
        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<Control-q>", lambda _event: self.close())
        self.root.bind("<Control-Q>", lambda _event: self.close())

    def run(self) -> None:
        """Start the Tkinter event loop."""

        self.root.mainloop()

    def _build(self) -> None:
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

    def _build_species_tab(self, parent: ttk.Frame) -> None:
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
        self._update_species_toggle_label()

    def _build_thermo_tab(self, parent: ttk.Frame) -> None:
        controls = ttk.Frame(parent)
        controls.pack(side="left", fill="y", padx=8, pady=8)
        plot_area = ttk.Frame(parent)
        plot_area.pack(side="right", fill="both", expand=True, padx=8, pady=8)
        self._thermo_scroll_canvas = tk.Canvas(plot_area, highlightthickness=0, background="#0b1020")
        thermo_scrollbar = ttk.Scrollbar(
            plot_area,
            orient="vertical",
            command=self._thermo_scroll_canvas.yview,
        )
        self._thermo_plot_area = ttk.Frame(self._thermo_scroll_canvas)
        self._thermo_window = self._thermo_scroll_canvas.create_window(
            (0, 0),
            window=self._thermo_plot_area,
            anchor="nw",
        )
        self._thermo_scroll_canvas.configure(yscrollcommand=thermo_scrollbar.set)
        self._thermo_plot_area.bind(
            "<Configure>",
            lambda _event: self._thermo_scroll_canvas.configure(
                scrollregion=self._thermo_scroll_canvas.bbox("all")
            ),
        )
        self._thermo_scroll_canvas.bind(
            "<Configure>",
            lambda event: self._thermo_scroll_canvas.itemconfigure(self._thermo_window, width=event.width),
        )
        self._thermo_scroll_canvas.bind("<Enter>", self._bind_thermo_mousewheel)
        self._thermo_scroll_canvas.bind("<Leave>", self._unbind_thermo_mousewheel)
        self._thermo_scroll_canvas.pack(side="left", fill="both", expand=True)
        thermo_scrollbar.pack(side="right", fill="y")

        ttk.Label(controls, text="Simulations").pack(anchor="w")
        self.thermo_sim_list = tk.Listbox(controls, selectmode="multiple", exportselection=False, height=6)
        self._thermo_simulations = [
            simulation
            for simulation in self.project.simulations
            if simulation.thermo_df is not None
        ]
        for simulation in self.project.simulations:
            if simulation.thermo_df is not None:
                self.thermo_sim_list.insert("end", f"Simulation {simulation.index}")
        if self.thermo_sim_list.size():
            self.thermo_sim_list.select_set(0, "end")
        self.thermo_sim_list.bind(
            "<<ListboxSelect>>",
            lambda _event: self._update_thermo_range_controls(preserve=True),
        )
        self.thermo_sim_list.pack(fill="x", pady=(0, 12))

        ttk.Label(controls, text="Legend labels").pack(anchor="w")
        self.thermo_label_vars: dict[int, tk.StringVar] = {}
        labels_frame = ttk.Frame(controls)
        labels_frame.pack(fill="x", pady=(0, 12))
        for simulation in self._thermo_simulations:
            ttk.Label(labels_frame, text=f"Simulation {simulation.index}").pack(anchor="w")
            label_var = tk.StringVar()
            self.thermo_label_vars[simulation.index] = label_var
            ttk.Entry(labels_frame, textvariable=label_var).pack(fill="x", pady=(0, 6))

        available = sorted(
            {
                column
                for simulation in self.project.simulations
                if simulation.thermo_df is not None
                for column in simulation.thermo_df.columns
                if column != "Step"
            }
        )
        values = [value for value in THERMO_DEFAULTS if value in available] or available
        self.thermo_parameter = tk.StringVar(value=values[0] if values else "")
        ttk.Label(controls, text="Parameter").pack(anchor="w")
        self.thermo_parameter_combo = ttk.Combobox(
            controls,
            textvariable=self.thermo_parameter,
            values=values,
            state="readonly",
        )
        self.thermo_parameter_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._update_thermo_range_controls(preserve=False),
        )
        self.thermo_parameter_combo.pack(fill="x", pady=(0, 12))
        ttk.Label(controls, text="Step range").pack(anchor="w")
        self._thermo_step_bounds: tuple[float, float] | None = None
        self._updating_thermo_step_controls = False
        self.thermo_step_min = tk.DoubleVar()
        self.thermo_step_max = tk.DoubleVar()
        self.thermo_step_label = tk.StringVar()
        ttk.Label(controls, textvariable=self.thermo_step_label).pack(anchor="w", pady=(0, 4))
        self.thermo_step_min_slider = ttk.Scale(
            controls,
            orient="horizontal",
            variable=self.thermo_step_min,
            command=lambda _value: self._on_thermo_step_slider("min"),
        )
        self.thermo_step_min_slider.pack(fill="x", pady=(0, 4))
        self.thermo_step_max_slider = ttk.Scale(
            controls,
            orient="horizontal",
            variable=self.thermo_step_max,
            command=lambda _value: self._on_thermo_step_slider("max"),
        )
        self.thermo_step_max_slider.pack(fill="x", pady=(0, 8))
        ttk.Button(
            controls,
            text="Full step range",
            command=lambda: self._update_thermo_step_controls(preserve=False),
        ).pack(fill="x", pady=(0, 12))

        ttk.Label(controls, text="Y-axis range").pack(anchor="w")
        self.thermo_y_min = tk.StringVar()
        self.thermo_y_max = tk.StringVar()
        ttk.Label(controls, text="Minimum").pack(anchor="w")
        ttk.Entry(controls, textvariable=self.thermo_y_min).pack(fill="x", pady=(0, 4))
        ttk.Label(controls, text="Maximum").pack(anchor="w")
        ttk.Entry(controls, textvariable=self.thermo_y_max).pack(fill="x", pady=(0, 8))
        ttk.Button(
            controls,
            text="Full y range",
            command=lambda: self._update_thermo_y_controls(preserve=False),
        ).pack(fill="x", pady=(0, 12))
        self._update_thermo_step_controls(preserve=False)
        self._update_thermo_y_controls(preserve=False)
        ttk.Button(controls, text="Plot", command=self._plot_thermo).pack(fill="x")

    def _build_rdf_tab(self, parent: ttk.Frame) -> None:
        controls = ttk.Frame(parent)
        controls.pack(side="left", fill="y", padx=8, pady=8)
        plot_area = ttk.Frame(parent)
        plot_area.pack(side="right", fill="both", expand=True, padx=8, pady=8)
        self._rdf_plot_area = plot_area

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
        self.rdf_sim_list.bind("<<ListboxSelect>>", lambda _event: self._set_rdf_last_timesteps())
        self.rdf_sim_list.pack(fill="x", pady=(0, 12))

        elements = self.project.config.element_list
        default_a = "Li" if "Li" in elements else (elements[0] if elements else "")
        default_b = "O" if "O" in elements else (elements[1] if len(elements) > 1 else default_a)
        self.rdf_element_a = tk.StringVar(value=default_a)
        self.rdf_element_b = tk.StringVar(value=default_b)

        ttk.Label(controls, text="Element A").pack(anchor="w")
        ttk.Combobox(controls, textvariable=self.rdf_element_a, values=elements, state="readonly").pack(
            fill="x", pady=(0, 12)
        )
        ttk.Label(controls, text="Element B").pack(anchor="w")
        ttk.Combobox(controls, textvariable=self.rdf_element_b, values=elements, state="readonly").pack(
            fill="x", pady=(0, 12)
        )

        self.rdf_timestep_start = tk.StringVar()
        self.rdf_timestep_end = tk.StringVar()
        ttk.Label(controls, text="Timestep start").pack(anchor="w")
        ttk.Entry(controls, textvariable=self.rdf_timestep_start).pack(fill="x", pady=(0, 8))
        ttk.Label(controls, text="Timestep end").pack(anchor="w")
        ttk.Entry(controls, textvariable=self.rdf_timestep_end).pack(fill="x", pady=(0, 8))
        ttk.Button(controls, text="Last 5 timesteps", command=self._set_rdf_last_timesteps).pack(
            fill="x", pady=(0, 12)
        )

        self.rdf_bin_width = tk.StringVar(value="0.1")
        ttk.Label(controls, text="Bin width").pack(anchor="w")
        ttk.Entry(controls, textvariable=self.rdf_bin_width).pack(fill="x", pady=(0, 12))
        ttk.Button(controls, text="Plot", command=self._plot_rdf).pack(fill="x")

        self.rdf_status = ttk.Label(plot_area, text="", wraplength=620, justify="left")
        self.rdf_status.pack(anchor="nw", padx=8, pady=8)
        self._set_rdf_last_timesteps()

    def _build_smiles_tab(self, parent: ttk.Frame) -> None:
        controls = ttk.Frame(parent)
        controls.pack(side="left", fill="y", padx=8, pady=8)
        output = ttk.Frame(parent)
        output.pack(side="right", fill="both", expand=True, padx=8, pady=8)

        sim_values = [str(sim.index) for sim in self.project.simulations if sim.has_bond_data]
        self.smiles_simulation = tk.StringVar(value=sim_values[0] if sim_values else "")
        self.smiles_formula = tk.StringVar()
        self.smiles_value = tk.StringVar()

        ttk.Label(controls, text="Simulation").pack(anchor="w")
        self.smiles_sim_combo = ttk.Combobox(
            controls,
            textvariable=self.smiles_simulation,
            values=sim_values,
            state="readonly",
        )
        self.smiles_sim_combo.pack(fill="x", pady=(0, 12))
        self.smiles_sim_combo.bind("<<ComboboxSelected>>", lambda _event: self._refresh_formula_options())

        ttk.Label(controls, text="Formula/species").pack(anchor="w")
        self.smiles_formula_combo = ttk.Combobox(controls, textvariable=self.smiles_formula, state="readonly")
        self.smiles_formula_combo.pack(fill="x", pady=(0, 12))
        self.smiles_formula_combo.bind("<<ComboboxSelected>>", lambda _event: self._refresh_smiles_options())

        ttk.Label(controls, text="SMILES").pack(anchor="w")
        self.smiles_combo = ttk.Combobox(controls, textvariable=self.smiles_value, state="readonly", width=42)
        self.smiles_combo.pack(fill="x", pady=(0, 12))

        ttk.Button(controls, text="Generate", command=self._generate_molecule).pack(fill="x")

        self.molecule_label = ttk.Label(output)
        self.molecule_label.pack(fill="both", expand=True)
        self._refresh_formula_options()

    def _build_reaction_table_tab(self, parent: ttk.Frame) -> None:
        table_frame = ttk.Frame(parent)
        table_frame.pack(fill="both", expand=True, padx=8, pady=8)

        simulation_columns = [f"simulation_{index}" for index in self._reaction_simulation_indices]
        columns = ("count", *simulation_columns, "reaction")
        self.reaction_table = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
        )
        self.reaction_table.heading("count", text="Total")
        self.reaction_table.heading("reaction", text="Reaction path (SMILES)")
        self.reaction_table.column("count", width=90, minwidth=70, anchor="e", stretch=False)
        for column, index in zip(simulation_columns, self._reaction_simulation_indices, strict=False):
            self.reaction_table.heading(column, text=f"Simulation {index}")
            self.reaction_table.column(column, width=105, minwidth=90, anchor="e", stretch=False)
        self.reaction_table.column("reaction", width=980, minwidth=360, anchor="w", stretch=True)

        y_scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.reaction_table.yview)
        x_scrollbar = ttk.Scrollbar(table_frame, orient="horizontal", command=self.reaction_table.xview)
        self.reaction_table.configure(yscrollcommand=y_scrollbar.set, xscrollcommand=x_scrollbar.set)

        self.reaction_table.grid(row=0, column=0, sticky="nsew")
        y_scrollbar.grid(row=0, column=1, sticky="ns")
        x_scrollbar.grid(row=1, column=0, sticky="ew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        for path in self._reaction_paths:
            per_simulation_counts = self._reaction_counts_by_simulation.get(path.reaction, {})
            values = (
                path.count,
                *[
                    per_simulation_counts.get(index, 0)
                    for index in self._reaction_simulation_indices
                ],
                path.reaction,
            )
            self.reaction_table.insert("", "end", values=values)
        self.reaction_table.bind("<<TreeviewSelect>>", self._sync_reaction_path_copy_field)
        self.reaction_table.bind("<Control-c>", self._copy_selected_reaction_path)
        self.reaction_table.bind("<Control-C>", self._copy_selected_reaction_path)

        copy_frame = ttk.Frame(parent)
        copy_frame.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Label(copy_frame, text="Selected reaction path").pack(anchor="w")
        self.reaction_path_copy_value = tk.StringVar()
        self.reaction_path_copy_entry = ttk.Entry(
            copy_frame,
            textvariable=self.reaction_path_copy_value,
            state="readonly",
        )
        self.reaction_path_copy_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(copy_frame, text="Copy", command=self._copy_selected_reaction_path).pack(side="right")

        children = self.reaction_table.get_children()
        if children:
            self.reaction_table.selection_set(children[0])
            self.reaction_table.focus(children[0])
            self._sync_reaction_path_copy_field()

    def _build_reaction_tab(self, parent: ttk.Frame) -> None:
        controls = ttk.Frame(parent)
        controls.pack(side="left", fill="y", padx=8, pady=8)
        output = ttk.Frame(parent)
        output.pack(side="right", fill="both", expand=True, padx=8, pady=8)

        reaction_values = [path.reaction for path in self._reaction_paths]
        self.reaction_path_value = tk.StringVar(value=reaction_values[0] if reaction_values else "")

        ttk.Label(controls, text="Reaction path").pack(anchor="w")
        self.reaction_path_combo = ttk.Combobox(
            controls,
            textvariable=self.reaction_path_value,
            values=reaction_values,
            width=70,
        )
        self.reaction_path_combo.pack(fill="x", pady=(0, 12))
        ttk.Button(controls, text="Open first occurrence in OVITO", command=self._open_reaction_in_ovito).pack(
            fill="x"
        )

        self.reaction_status = ttk.Label(
            output,
            text="Select or paste a reaction path from paths.out, then open it in OVITO.",
            wraplength=620,
            justify="left",
        )
        self.reaction_status.pack(anchor="nw", padx=8, pady=8)

    def _plot_species(self) -> None:
        try:
            simulations = self._selected_species_simulations()
            selected_species = [self.species_list.get(index) for index in self.species_list.curselection()]
            if not simulations or not selected_species:
                raise ValueError("Select at least one simulation and one species.")
            figure = plot_species(simulations, selected_species)
            self._replace_canvas("_species_canvas", self._species_plot_area, figure)
        except Exception as exc:  # pragma: no cover - GUI feedback.
            messagebox.showerror("Species plotting failed", str(exc))

    def _plot_thermo(self) -> None:
        try:
            parameter = self.thermo_parameter.get()
            if not parameter:
                raise ValueError("Select a thermodynamic parameter.")
            simulations = self._selected_thermo_simulations()
            if not simulations:
                raise ValueError("Select at least one simulation.")
            for canvas in self._thermo_canvases:
                self._destroy_canvas(canvas)
            self._thermo_canvases = []
            legend_labels = self._thermo_legend_labels()
            figures = plot_thermo(
                simulations,
                parameter,
                legend_labels=legend_labels,
                step_range=self._thermo_step_range(),
                y_range=self._thermo_y_range(),
            )
            for figure in figures:
                canvas = FigureCanvasTkAgg(figure, master=self._thermo_plot_area)
                canvas.draw()
                canvas.get_tk_widget().pack(fill="x", expand=False, pady=(0, 12))
                self._thermo_canvases.append(canvas)
            self._thermo_scroll_canvas.yview_moveto(0)
        except Exception as exc:  # pragma: no cover - GUI feedback.
            messagebox.showerror("Thermo plotting failed", str(exc))

    def _plot_rdf(self) -> None:
        try:
            simulations = self._selected_rdf_simulations()
            if not simulations:
                raise ValueError("Select at least one simulation.")
            element_a = self.rdf_element_a.get()
            element_b = self.rdf_element_b.get()
            if not element_a or not element_b:
                raise ValueError("Select two elements.")
            start = int(self.rdf_timestep_start.get())
            end = int(self.rdf_timestep_end.get())
            bin_width = float(self.rdf_bin_width.get())

            results = compute_rdf(simulations, element_a, element_b, (start, end), bin_width)
            figure = plot_rdf(results, element_a, element_b)
            self._replace_canvas("_rdf_canvas", self._rdf_plot_area, figure)
            used_timesteps = sorted({timestep for result in results for timestep in result.timesteps})
            self.rdf_status.configure(
                text=(
                    f"Used {len(used_timesteps)} timestep(s): "
                    f"{used_timesteps[0]} to {used_timesteps[-1]}"
                )
            )
        except Exception as exc:  # pragma: no cover - GUI feedback.
            messagebox.showerror("RDF plotting failed", str(exc))

    def _generate_molecule(self) -> None:
        try:
            smiles = self.smiles_value.get()
            if not smiles:
                raise ValueError("Select a SMILES string.")
            self._molecule_photo = molecule_photo_image(smiles)
            self.molecule_label.configure(image=self._molecule_photo)
        except Exception as exc:  # pragma: no cover - GUI feedback.
            messagebox.showerror("SMILES visualization failed", str(exc))

    def _open_reaction_in_ovito(self) -> None:
        try:
            reaction = normalize_reaction_path(self.reaction_path_value.get())
            if not reaction:
                raise ValueError("Select or paste a reaction path.")
            simulation, occurrence = self.project.first_reaction_occurrence(reaction)
            scene = create_reaction_scene(simulation, occurrence)
            launch_ovito_scene(scene)
            self.reaction_status.configure(
                text=(
                    f"Opened OVITO scene for simulation {simulation.index}: "
                    f"{occurrence.timestep_reactants} -> {occurrence.timestep_products}\n"
                    f"Scene files: {scene.directory}"
                )
            )
        except OvitoNotAvailableError as exc:  # pragma: no cover - GUI feedback.
            self.reaction_status.configure(text=str(exc))
            messagebox.showwarning("OVITO not available", str(exc))
        except Exception as exc:  # pragma: no cover - GUI feedback.
            messagebox.showerror("OVITO visualization failed", str(exc))

    def _selected_species_simulations(self):
        available = [simulation for simulation in self.project.simulations if simulation.species_df is not None]
        return [available[index] for index in self.species_sim_list.curselection()]

    def _selected_thermo_simulations(self):
        return [self._thermo_simulations[index] for index in self.thermo_sim_list.curselection()]

    def _selected_rdf_simulations(self):
        return [self._rdf_simulations[index] for index in self.rdf_sim_list.curselection()]

    def _thermo_legend_labels(self) -> dict[int, str]:
        return {index: label_var.get() for index, label_var in self.thermo_label_vars.items()}

    def _set_rdf_last_timesteps(self) -> None:
        timesteps = sorted(
            {
                timestep
                for simulation in self._selected_rdf_simulations()
                for timestep in self._rdf_timesteps(simulation)
            }
        )
        if not timesteps:
            self.rdf_timestep_start.set("")
            self.rdf_timestep_end.set("")
            return
        selected = timesteps[-5:]
        self.rdf_timestep_start.set(str(selected[0]))
        self.rdf_timestep_end.set(str(selected[-1]))

    def _rdf_timesteps(self, simulation: LoadedSimulation) -> list[int]:
        if simulation.index not in self._rdf_timesteps_by_simulation:
            if simulation.trajectory_path is None:
                self._rdf_timesteps_by_simulation[simulation.index] = []
            else:
                self._rdf_timesteps_by_simulation[simulation.index] = list_lammpstrj_timesteps(
                    simulation.trajectory_path
                )
        return self._rdf_timesteps_by_simulation[simulation.index]

    def _thermo_step_range(self) -> tuple[float, float] | None:
        if self._thermo_step_bounds is None:
            return None
        return tuple(sorted((self.thermo_step_min.get(), self.thermo_step_max.get())))

    def _thermo_y_range(self) -> tuple[float, float] | None:
        minimum = self.thermo_y_min.get().strip()
        maximum = self.thermo_y_max.get().strip()
        if not minimum and not maximum:
            return None
        if not minimum or not maximum:
            raise ValueError("Enter both y-axis minimum and maximum, or reset to the full y range.")
        return tuple(sorted((float(minimum), float(maximum))))

    def _update_thermo_range_controls(self, preserve: bool) -> None:
        self._update_thermo_step_controls(preserve=preserve)
        self._update_thermo_y_controls(preserve=preserve)

    def _update_thermo_step_controls(self, preserve: bool) -> None:
        bounds = self._thermo_step_bounds_for_simulations(self._selected_thermo_simulations())
        if bounds is None:
            bounds = self._thermo_step_bounds_for_simulations(self._thermo_simulations)
        previous_bounds = self._thermo_step_bounds
        self._thermo_step_bounds = bounds
        if bounds is None:
            self.thermo_step_label.set("No step data")
            self.thermo_step_min_slider.configure(state="disabled")
            self.thermo_step_max_slider.configure(state="disabled")
            return

        lower, upper = bounds
        if preserve and previous_bounds is not None:
            current_lower = self.thermo_step_min.get()
            current_upper = self.thermo_step_max.get()
            lower_value = min(max(current_lower, lower), upper)
            upper_value = min(max(current_upper, lower), upper)
            if lower_value > upper_value or (lower_value == upper_value and lower != upper):
                lower_value, upper_value = lower, upper
        else:
            lower_value, upper_value = lower, upper

        self._updating_thermo_step_controls = True
        self.thermo_step_min_slider.configure(from_=lower, to=upper)
        self.thermo_step_max_slider.configure(from_=lower, to=upper)
        self.thermo_step_min.set(lower_value)
        self.thermo_step_max.set(upper_value)
        state = "normal" if lower != upper else "disabled"
        self.thermo_step_min_slider.configure(state=state)
        self.thermo_step_max_slider.configure(state=state)
        self._updating_thermo_step_controls = False
        self._refresh_thermo_step_label()

    def _thermo_step_bounds_for_simulations(self, simulations) -> tuple[float, float] | None:
        bounds = []
        for simulation in simulations:
            if simulation.thermo_df is None or "Step" not in simulation.thermo_df.columns:
                continue
            steps = simulation.thermo_df["Step"].dropna()
            if steps.empty:
                continue
            bounds.append((float(steps.min()), float(steps.max())))
        if not bounds:
            return None
        return min(bound[0] for bound in bounds), max(bound[1] for bound in bounds)

    def _on_thermo_step_slider(self, changed: str) -> None:
        if self._updating_thermo_step_controls:
            return
        lower = self.thermo_step_min.get()
        upper = self.thermo_step_max.get()
        if lower > upper:
            if changed == "min":
                self.thermo_step_max.set(lower)
            else:
                self.thermo_step_min.set(upper)
        self._refresh_thermo_step_label()

    def _refresh_thermo_step_label(self) -> None:
        if self._thermo_step_bounds is None:
            self.thermo_step_label.set("No step data")
            return
        lower, upper = self._thermo_step_range() or self._thermo_step_bounds
        self.thermo_step_label.set(
            f"{self._format_step_value(lower)} to {self._format_step_value(upper)}"
        )

    def _update_thermo_y_controls(self, preserve: bool) -> None:
        bounds = self._thermo_y_bounds_for_simulations(
            self._selected_thermo_simulations(),
            self.thermo_parameter.get(),
        )
        if bounds is None:
            bounds = self._thermo_y_bounds_for_simulations(self._thermo_simulations, self.thermo_parameter.get())
        if bounds is None:
            if not preserve:
                self.thermo_y_min.set("")
                self.thermo_y_max.set("")
            return

        if preserve and self.thermo_y_min.get().strip() and self.thermo_y_max.get().strip():
            try:
                current_lower = float(self.thermo_y_min.get())
                current_upper = float(self.thermo_y_max.get())
            except ValueError:
                current_lower, current_upper = bounds
            lower, upper = sorted((current_lower, current_upper))
        else:
            lower, upper = bounds

        self.thermo_y_min.set(self._format_step_value(lower))
        self.thermo_y_max.set(self._format_step_value(upper))

    def _thermo_y_bounds_for_simulations(
        self,
        simulations,
        parameter: str,
    ) -> tuple[float, float] | None:
        bounds = []
        if not parameter:
            return None
        for simulation in simulations:
            if simulation.thermo_df is None or parameter not in simulation.thermo_df.columns:
                continue
            values = simulation.thermo_df[parameter].dropna()
            if values.empty:
                continue
            bounds.append((float(values.min()), float(values.max())))
        if not bounds:
            return None
        return min(bound[0] for bound in bounds), max(bound[1] for bound in bounds)

    @staticmethod
    def _format_step_value(value: float) -> str:
        if float(value).is_integer():
            return str(int(value))
        return f"{value:g}"

    def _selected_reaction_path_from_table(self) -> str:
        selected = self.reaction_table.selection()
        item_id = selected[0] if selected else self.reaction_table.focus()
        if not item_id:
            return ""
        return self.reaction_table.set(item_id, "reaction")

    def _sync_reaction_path_copy_field(self, _event=None) -> None:
        self.reaction_path_copy_value.set(self._selected_reaction_path_from_table())

    def _copy_selected_reaction_path(self, _event=None) -> str:
        reaction = self._selected_reaction_path_from_table()
        if reaction:
            self.root.clipboard_clear()
            self.root.clipboard_append(reaction)
            self.root.update_idletasks()
            self.reaction_path_copy_value.set(reaction)
        return "break"

    def _toggle_species_selection(self) -> None:
        if self.species_list.size() == 0:
            return
        if len(self.species_list.curselection()) == self.species_list.size():
            self.species_list.selection_clear(0, "end")
        else:
            self.species_list.select_set(0, "end")
        self._update_species_toggle_label()

    def _update_species_toggle_label(self) -> None:
        if not hasattr(self, "species_toggle_button"):
            return
        if self.species_list.size() == 0:
            self.species_toggle_button.configure(text="Select all species")
        elif len(self.species_list.curselection()) == self.species_list.size():
            self.species_toggle_button.configure(text="Deselect all species")
        else:
            self.species_toggle_button.configure(text="Select all species")

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

        self._molecule_photo = None
        plt.close("all")

        try:
            self.root.quit()
            self.root.destroy()
        except tk.TclError:
            pass

    def _refresh_formula_options(self) -> None:
        simulation = self._selected_smiles_simulation()
        formulas = formulas_for_simulation(simulation.chem_formulas) if simulation and simulation.chem_formulas else []
        self.smiles_formula_combo.configure(values=formulas)
        self.smiles_formula.set(formulas[0] if formulas else "")
        self._refresh_smiles_options()

    def _refresh_smiles_options(self) -> None:
        simulation = self._selected_smiles_simulation()
        formula = self.smiles_formula.get()
        values = []
        if simulation and simulation.chem_formulas and simulation.smiles and formula:
            values = smiles_for_formula(simulation.chem_formulas, simulation.smiles, formula)
        self.smiles_combo.configure(values=values)
        self.smiles_value.set(values[0] if values else "")

    def _selected_smiles_simulation(self):
        value = self.smiles_simulation.get()
        if not value:
            return None
        return self.project.simulation(int(value))

    def _replace_canvas(self, attr_name: str, parent: ttk.Frame, figure) -> None:
        old_canvas = getattr(self, attr_name)
        if old_canvas is not None:
            self._destroy_canvas(old_canvas)
        canvas = FigureCanvasTkAgg(figure, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        setattr(self, attr_name, canvas)

    def _destroy_canvas(self, canvas: FigureCanvasTkAgg) -> None:
        try:
            plt.close(canvas.figure)
            canvas.get_tk_widget().destroy()
        except tk.TclError:
            pass

    def _bind_thermo_mousewheel(self, _event) -> None:
        self.root.bind_all("<MouseWheel>", self._on_thermo_mousewheel)
        self.root.bind_all("<Button-4>", self._on_thermo_mousewheel)
        self.root.bind_all("<Button-5>", self._on_thermo_mousewheel)

    def _unbind_thermo_mousewheel(self, _event) -> None:
        self.root.unbind_all("<MouseWheel>")
        self.root.unbind_all("<Button-4>")
        self.root.unbind_all("<Button-5>")

    def _on_thermo_mousewheel(self, event) -> None:
        if getattr(event, "num", None) == 4:
            delta = -1
        elif getattr(event, "num", None) == 5:
            delta = 1
        else:
            delta = -1 * int(event.delta / 120)
        self._thermo_scroll_canvas.yview_scroll(delta, "units")


def launch_gui(project: LammpalyzeProject) -> None:
    """Launch the lammpalyze GUI."""

    gui = LammpalyzeGUI(project)
    try:
        gui.run()
    finally:
        gui.close()
