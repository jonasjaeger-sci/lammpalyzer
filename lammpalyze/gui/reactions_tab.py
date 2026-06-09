"""Reaction table and OVITO visualization tabs for the Tkinter GUI."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from lammpalyze.ovito import OvitoNotAvailableError, create_reaction_scene, launch_ovito_scene, normalize_reaction_path
from lammpalyze.reactions import format_connected_reaction_pathways


class ReactionTabMixin:
    """Build and manage reaction-count and reaction-visualization tabs."""

    def _build_reaction_table_tab(self, parent: ttk.Frame) -> None:
        """Create the reaction-path count table and copy controls."""

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

    def _build_connected_pathways_tab(self, parent: ttk.Frame) -> None:
        """Create the connected reaction pathway table and outline."""

        controls = ttk.Frame(parent)
        controls.pack(fill="x", padx=8, pady=8)
        ttk.Label(controls, text="Notation").pack(side="left", padx=(0, 8))
        self.connected_pathway_notation = tk.StringVar(value="formula")
        ttk.Radiobutton(
            controls,
            text="Formula",
            variable=self.connected_pathway_notation,
            value="formula",
            command=self._refresh_connected_pathways,
        ).pack(side="left")
        ttk.Radiobutton(
            controls,
            text="SMILES",
            variable=self.connected_pathway_notation,
            value="smiles",
            command=self._refresh_connected_pathways,
        ).pack(side="left", padx=(8, 0))

        panes = ttk.PanedWindow(parent, orient="vertical")
        panes.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        table_frame = ttk.Frame(panes)
        outline_frame = ttk.Frame(panes)
        panes.add(table_frame, weight=3)
        panes.add(outline_frame, weight=2)

        columns = ("step", "parents", "depth", "reactants", "arrow", "products", "count", "simulations")
        self.connected_pathway_table = ttk.Treeview(table_frame, columns=columns, show="headings")
        self.connected_pathway_table.heading("step", text="Pathway")
        self.connected_pathway_table.heading("parents", text="After")
        self.connected_pathway_table.heading("depth", text="Depth")
        self.connected_pathway_table.heading("reactants", text="Reactants")
        self.connected_pathway_table.heading("arrow", text="")
        self.connected_pathway_table.heading("products", text="Products")
        self.connected_pathway_table.heading("count", text="Count")
        self.connected_pathway_table.heading("simulations", text="Simulations")
        self.connected_pathway_table.column("step", width=85, minwidth=70, anchor="center", stretch=False)
        self.connected_pathway_table.column("parents", width=95, minwidth=70, anchor="center", stretch=False)
        self.connected_pathway_table.column("depth", width=70, minwidth=60, anchor="e", stretch=False)
        self.connected_pathway_table.column("reactants", width=360, minwidth=220, anchor="w", stretch=True)
        self.connected_pathway_table.column("arrow", width=55, minwidth=45, anchor="center", stretch=False)
        self.connected_pathway_table.column("products", width=360, minwidth=220, anchor="w", stretch=True)
        self.connected_pathway_table.column("count", width=80, minwidth=70, anchor="e", stretch=False)
        self.connected_pathway_table.column("simulations", width=140, minwidth=100, anchor="w", stretch=False)

        y_scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.connected_pathway_table.yview)
        x_scrollbar = ttk.Scrollbar(table_frame, orient="horizontal", command=self.connected_pathway_table.xview)
        self.connected_pathway_table.configure(yscrollcommand=y_scrollbar.set, xscrollcommand=x_scrollbar.set)
        self.connected_pathway_table.grid(row=0, column=0, sticky="nsew")
        y_scrollbar.grid(row=0, column=1, sticky="ns")
        x_scrollbar.grid(row=1, column=0, sticky="ew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)
        self.connected_pathway_cell_value = tk.StringVar()
        self._connected_pathway_cell = ("", "")
        self.connected_pathway_table.bind("<ButtonRelease-1>", self._sync_connected_pathway_cell)
        self.connected_pathway_table.bind("<Control-c>", self._copy_connected_pathway_cell)
        self.connected_pathway_table.bind("<Control-C>", self._copy_connected_pathway_cell)

        copy_frame = ttk.Frame(table_frame)
        copy_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Entry(
            copy_frame,
            textvariable=self.connected_pathway_cell_value,
            state="readonly",
        ).pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(copy_frame, text="Copy cell", command=self._copy_connected_pathway_cell).pack(side="right")

        self.connected_pathway_outline = tk.Text(outline_frame, wrap="none", height=10, state="disabled")
        outline_y_scrollbar = ttk.Scrollbar(
            outline_frame,
            orient="vertical",
            command=self.connected_pathway_outline.yview,
        )
        outline_x_scrollbar = ttk.Scrollbar(
            outline_frame,
            orient="horizontal",
            command=self.connected_pathway_outline.xview,
        )
        self.connected_pathway_outline.configure(
            yscrollcommand=outline_y_scrollbar.set,
            xscrollcommand=outline_x_scrollbar.set,
        )
        self.connected_pathway_outline.grid(row=0, column=0, sticky="nsew")
        outline_y_scrollbar.grid(row=0, column=1, sticky="ns")
        outline_x_scrollbar.grid(row=1, column=0, sticky="ew")
        outline_frame.rowconfigure(0, weight=1)
        outline_frame.columnconfigure(0, weight=1)

        self._refresh_connected_pathways()

    def _build_reaction_tab(self, parent: ttk.Frame) -> None:
        """Create controls for opening reaction occurrences in OVITO."""

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

    def _open_reaction_in_ovito(self) -> None:
        """Open the first matching reaction occurrence in OVITO."""

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

    def _selected_reaction_path_from_table(self) -> str:
        """Return the reaction path from the selected or focused table row."""

        selected = self.reaction_table.selection()
        item_id = selected[0] if selected else self.reaction_table.focus()
        if not item_id:
            return ""
        return self.reaction_table.set(item_id, "reaction")

    def _sync_reaction_path_copy_field(self, _event=None) -> None:
        """Copy the selected table reaction into the read-only text field."""

        self.reaction_path_copy_value.set(self._selected_reaction_path_from_table())

    def _copy_selected_reaction_path(self, _event=None) -> str:
        """Copy the selected reaction path to the system clipboard."""

        reaction = self._selected_reaction_path_from_table()
        if reaction:
            self.root.clipboard_clear()
            self.root.clipboard_append(reaction)
            self.root.update_idletasks()
            self.reaction_path_copy_value.set(reaction)
        return "break"

    def _refresh_connected_pathways(self) -> None:
        """Refresh connected pathway table and text outline for the notation."""

        notation = self.connected_pathway_notation.get()
        pathways = self.project.connected_reaction_pathways(notation=notation)
        for item_id in self.connected_pathway_table.get_children():
            self.connected_pathway_table.delete(item_id)

        for pathway in pathways:
            for step in pathway.steps:
                simulations = ", ".join(str(index) for index in step.simulations)
                self.connected_pathway_table.insert(
                    "",
                    "end",
                    values=(
                        step.label,
                        ", ".join(step.parents),
                        step.depth,
                        step.source,
                        step.arrow,
                        step.target,
                        step.count,
                        simulations,
                    ),
                )

        self.connected_pathway_outline.configure(state="normal")
        self.connected_pathway_outline.delete("1.0", "end")
        self.connected_pathway_outline.insert("1.0", format_connected_reaction_pathways(pathways))
        self.connected_pathway_outline.configure(state="disabled")

    def _sync_connected_pathway_cell(self, event=None) -> None:
        """Store the clicked connected-pathway table cell for copying."""

        if event is None:
            item_id = self.connected_pathway_table.focus()
            column_id = "#1"
        else:
            item_id = self.connected_pathway_table.identify_row(event.y)
            column_id = self.connected_pathway_table.identify_column(event.x)
        columns = self.connected_pathway_table["columns"]
        try:
            column = columns[int(column_id.removeprefix("#")) - 1]
        except (IndexError, ValueError):
            return
        if not item_id:
            return
        value = self.connected_pathway_table.set(item_id, column)
        self._connected_pathway_cell = (item_id, column)
        self.connected_pathway_cell_value.set(value)

    def _copy_connected_pathway_cell(self, _event=None) -> str:
        """Copy the selected connected-pathway table cell."""

        item_id, column = self._connected_pathway_cell
        if not item_id or not column:
            self._sync_connected_pathway_cell()
            item_id, column = self._connected_pathway_cell
        if item_id and column:
            value = self.connected_pathway_table.set(item_id, column)
            self.root.clipboard_clear()
            self.root.clipboard_append(value)
            self.root.update_idletasks()
            self.connected_pathway_cell_value.set(value)
        return "break"
