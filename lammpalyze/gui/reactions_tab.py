"""Reaction table and OVITO visualization tabs for the Tkinter GUI."""

from __future__ import annotations

import csv
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from lammpalyze.gui.molecule_tab import MOLECULE_THUMBNAIL_SIZE
from lammpalyze.ovito import OvitoNotAvailableError, create_reaction_scene, launch_ovito_scene, normalize_reaction_path
from lammpalyze.reactions import format_connected_reaction_pathways
from lammpalyze.smiles import reaction_smiles_groups, reaction_smiles_path


class ReactionTabMixin:
    """Build and manage reaction-count and reaction-visualization tabs."""

    def _build_reaction_table_tab(self, parent: ttk.Frame) -> None:
        """Create the reaction-path count table and copy controls."""

        panes = ttk.PanedWindow(parent, orient="vertical")
        panes.pack(fill="both", expand=True, padx=8, pady=8)
        table_frame = ttk.Frame(panes)
        preview_frame = ttk.Frame(panes)
        panes.add(table_frame, weight=3)
        panes.add(preview_frame, weight=2)

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

        copy_frame = ttk.Frame(table_frame)
        copy_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Label(copy_frame, text="Selected reaction path").pack(anchor="w")
        self.reaction_path_copy_value = tk.StringVar()
        self.reaction_path_copy_entry = ttk.Entry(
            copy_frame,
            textvariable=self.reaction_path_copy_value,
            state="readonly",
        )
        self.reaction_path_copy_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(copy_frame, text="Copy", command=self._copy_selected_reaction_path).pack(side="right")

        self.reaction_path_canvas, self.reaction_path_gallery = self._build_scrollable_molecule_gallery(
            preview_frame
        )
        self.reaction_path_canvas.pack(side="left", fill="both", expand=True)
        reaction_path_scrollbar = ttk.Scrollbar(
            preview_frame,
            orient="vertical",
            command=self.reaction_path_canvas.yview,
        )
        reaction_path_scrollbar.pack(side="right", fill="y")
        self.reaction_path_canvas.configure(yscrollcommand=reaction_path_scrollbar.set)

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
        ttk.Label(controls, text="Minimum total occurrences").pack(side="left", padx=(20, 8))
        self.connected_pathway_min_count = tk.StringVar(value="1")
        threshold_input = ttk.Spinbox(
            controls,
            from_=1,
            to=1_000_000,
            increment=1,
            textvariable=self.connected_pathway_min_count,
            width=8,
            command=self._refresh_connected_pathways,
        )
        threshold_input.pack(side="left")
        threshold_input.bind("<Return>", self._refresh_connected_pathways)
        threshold_input.bind("<FocusOut>", self._refresh_connected_pathways)
        ttk.Button(
            controls,
            text="Export CSV",
            command=self._export_connected_pathways_csv,
        ).pack(side="right")

        panes = ttk.PanedWindow(parent, orient="vertical")
        panes.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        table_frame = ttk.Frame(panes)
        outline_frame = ttk.Frame(panes)
        panes.add(table_frame, weight=3)
        panes.add(outline_frame, weight=2)

        self.connected_pathway_simulation_columns = [
            f"simulation_{index}" for index in self._reaction_simulation_indices
        ]
        columns = (
            "step",
            "parents",
            "depth",
            "reactants",
            "arrow",
            "products",
            "count",
            *self.connected_pathway_simulation_columns,
            "simulations",
        )
        self.connected_pathway_table = ttk.Treeview(table_frame, columns=columns, show="headings")
        self.connected_pathway_table.heading("step", text="Pathway")
        self.connected_pathway_table.heading("parents", text="After")
        self.connected_pathway_table.heading("depth", text="Depth")
        self.connected_pathway_table.heading("reactants", text="Reactants")
        self.connected_pathway_table.heading("arrow", text="")
        self.connected_pathway_table.heading("products", text="Products")
        self.connected_pathway_table.heading("count", text="Count")
        for column, index in zip(
            self.connected_pathway_simulation_columns,
            self._reaction_simulation_indices,
            strict=False,
        ):
            self.connected_pathway_table.heading(column, text=f"Simulation {index}")
        self.connected_pathway_table.heading("simulations", text="Simulations")
        self.connected_pathway_table.column("step", width=85, minwidth=70, anchor="center", stretch=False)
        self.connected_pathway_table.column("parents", width=95, minwidth=70, anchor="center", stretch=False)
        self.connected_pathway_table.column("depth", width=70, minwidth=60, anchor="e", stretch=False)
        self.connected_pathway_table.column("reactants", width=360, minwidth=220, anchor="w", stretch=True)
        self.connected_pathway_table.column("arrow", width=55, minwidth=45, anchor="center", stretch=False)
        self.connected_pathway_table.column("products", width=360, minwidth=220, anchor="w", stretch=True)
        self.connected_pathway_table.column("count", width=80, minwidth=70, anchor="e", stretch=False)
        for column in self.connected_pathway_simulation_columns:
            self.connected_pathway_table.column(column, width=105, minwidth=90, anchor="e", stretch=False)
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

        reaction = self._selected_reaction_path_from_table()
        self.reaction_path_copy_value.set(reaction)
        if hasattr(self, "reaction_path_gallery"):
            self._render_reaction_path_gallery(reaction)

    def _copy_selected_reaction_path(self, _event=None) -> str:
        """Copy the selected reaction path to the system clipboard."""

        reaction = self._selected_reaction_path_from_table()
        if reaction:
            self.root.clipboard_clear()
            self.root.clipboard_append(reaction)
            self.root.update_idletasks()
            self.reaction_path_copy_value.set(reaction)
        return "break"

    def _refresh_connected_pathways(self, _event=None) -> None:
        """Refresh connected pathway table and text outline for the notation."""

        notation = self.connected_pathway_notation.get()
        min_count = self._connected_pathway_threshold()
        pathways = self.project.connected_reaction_pathways(notation=notation, min_count=min_count)
        for item_id in self.connected_pathway_table.get_children():
            self.connected_pathway_table.delete(item_id)

        for pathway in pathways:
            for step in pathway.steps:
                simulations = ", ".join(str(index) for index in step.simulations)
                per_simulation_counts = dict(step.counts_by_simulation)
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
                        *[
                            per_simulation_counts.get(index, 0)
                            for index in self._reaction_simulation_indices
                        ],
                        simulations,
                    ),
                )

        self.connected_pathway_outline.configure(state="normal")
        self.connected_pathway_outline.delete("1.0", "end")
        self.connected_pathway_outline.insert("1.0", format_connected_reaction_pathways(pathways))
        self.connected_pathway_outline.configure(state="disabled")
        self._connected_pathway_cell = ("", "")
        self.connected_pathway_cell_value.set("")

    def _connected_pathway_threshold(self) -> int:
        """Return the selected minimum pathway count, correcting invalid values."""

        try:
            value = int(self.connected_pathway_min_count.get())
        except ValueError:
            value = 1
        value = max(1, value)
        self.connected_pathway_min_count.set(str(value))
        return value

    def _export_connected_pathways_csv(self) -> None:
        """Export the currently visible connected pathway rows to CSV."""

        filename = filedialog.asksaveasfilename(
            title="Export connected pathways CSV",
            initialfile="connected_pathways.csv",
            defaultextension=".csv",
            filetypes=(("CSV file", "*.csv"), ("All files", "*.*")),
        )
        if not filename:
            return

        output_path = Path(filename)
        columns = self.connected_pathway_table["columns"]
        headings = [self.connected_pathway_table.heading(column)["text"] for column in columns]
        try:
            with output_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(headings)
                for item_id in self.connected_pathway_table.get_children():
                    writer.writerow([self.connected_pathway_table.set(item_id, column) for column in columns])
        except OSError as exc:  # pragma: no cover - GUI feedback.
            messagebox.showerror("Export failed", str(exc))
            return

        messagebox.showinfo("CSV exported", f"Exported CSV to {output_path}")

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

    def _render_reaction_path_gallery(self, reaction: str) -> None:
        """Render structures for the selected reaction path."""

        for child in self.reaction_path_gallery.winfo_children():
            child.destroy()
        self._reaction_path_gallery_photos = []
        self._reaction_path_gallery_vars = []
        if not reaction:
            return

        try:
            reactants, products = reaction_smiles_groups(normalize_reaction_path(reaction))
            arrow = self._reaction_path_arrow(reactants, products)
            self._render_reaction_side_by_side(reactants, products, arrow)
        except Exception as exc:  # pragma: no cover - GUI feedback.
            ttk.Label(
                self.reaction_path_gallery,
                text=f"Could not visualize reaction path: {exc}",
                wraplength=620,
                justify="left",
            ).pack(anchor="nw", padx=8, pady=8)

    def _render_reaction_side_by_side(
        self,
        reactants: list[str],
        products: list[str],
        arrow: str,
    ) -> None:
        """Render selected reaction molecules as reactants -> products."""

        reactant_frame = ttk.Frame(self.reaction_path_gallery)
        arrow_frame = ttk.Frame(self.reaction_path_gallery)
        product_frame = ttk.Frame(self.reaction_path_gallery)
        reactant_frame.grid(row=0, column=0, sticky="nsew", padx=(4, 10), pady=4)
        arrow_frame.grid(row=0, column=1, sticky="ns", padx=4, pady=4)
        product_frame.grid(row=0, column=2, sticky="nsew", padx=(10, 4), pady=4)
        self.reaction_path_gallery.columnconfigure(0, weight=1, uniform="reaction_side")
        self.reaction_path_gallery.columnconfigure(1, weight=0)
        self.reaction_path_gallery.columnconfigure(2, weight=1, uniform="reaction_side")
        self.reaction_path_gallery.rowconfigure(0, weight=1)

        ttk.Label(arrow_frame, text=arrow, anchor="center").pack(expand=True, fill="both", pady=96)
        self._render_smiles_gallery(
            reactant_frame,
            reactants,
            photo_attribute="_reaction_path_reactant_photos",
            variable_attribute="_reaction_path_reactant_vars",
            image_size=MOLECULE_THUMBNAIL_SIZE,
            canvas=self.reaction_path_canvas,
            columns=1,
            title="Reactants",
        )
        self._render_smiles_gallery(
            product_frame,
            products,
            photo_attribute="_reaction_path_product_photos",
            variable_attribute="_reaction_path_product_vars",
            image_size=MOLECULE_THUMBNAIL_SIZE,
            canvas=self.reaction_path_canvas,
            columns=1,
            title="Products",
        )
        self._reaction_path_gallery_photos = [
            *self._reaction_path_reactant_photos,
            *self._reaction_path_product_photos,
        ]
        self._reaction_path_gallery_vars = [
            *self._reaction_path_reactant_vars,
            *self._reaction_path_product_vars,
        ]

    def _reaction_path_arrow(self, reactants: list[str], products: list[str]) -> str:
        """Return a reversible arrow when the reverse reaction path exists."""

        available_reactions = {normalize_reaction_path(path.reaction) for path in self._reaction_paths}
        reverse_reaction = reaction_smiles_path(reactants=products, products=reactants)
        return "<->" if reverse_reaction in available_reactions else "->"
