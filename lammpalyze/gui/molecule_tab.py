"""Molecule-visualization tab for the Tkinter GUI."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from lammpalyze.gui.helpers import (
    MOLECULE_RESIZE_DEBOUNCE_MS,
    PNG_FILETYPES,
    image_output_path,
    molecule_observation_summary,
    molecule_render_size,
)
from lammpalyze.smiles import formulas_for_simulation, molecule_image, molecule_photo_image, smiles_for_formula

MOLECULE_THUMBNAIL_SIZE = (260, 200)
MOLECULE_GALLERY_MIN_COLUMN_WIDTH = 300


class MoleculeTabMixin:
    """Build and manage the molecule-visualization tab."""

    def _build_smiles_tab(self, parent: ttk.Frame) -> None:
        """Create controls and output area for molecule rendering."""

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
        self.smiles_formula_combo = ttk.Combobox(
            controls,
            textvariable=self.smiles_formula,
            state="normal",
        )
        self.smiles_formula_combo.pack(fill="x", pady=(0, 12))
        self.smiles_formula_combo.bind("<<ComboboxSelected>>", lambda _event: self._refresh_smiles_options())
        self.smiles_formula_combo.bind("<Return>", lambda _event: self._refresh_smiles_options())

        ttk.Label(controls, text="SMILES").pack(anchor="w")
        self.smiles_combo = ttk.Combobox(
            controls,
            textvariable=self.smiles_value,
            state="normal",
            width=42,
        )
        self.smiles_combo.pack(fill="x", pady=(0, 12))
        self.smiles_combo.bind("<<ComboboxSelected>>", lambda _event: self._refresh_smiles_metadata())
        self.smiles_combo.bind("<Return>", lambda _event: self._refresh_smiles_metadata())

        self.smiles_metadata = tk.StringVar()
        ttk.Label(
            controls,
            textvariable=self.smiles_metadata,
            wraplength=280,
            justify="left",
        ).pack(anchor="w", pady=(0, 12))

        ttk.Button(controls, text="Generate", command=self._generate_molecule).pack(fill="x")
        ttk.Button(controls, text="Generate all structures", command=self._generate_all_molecules).pack(
            fill="x",
            pady=(8, 0),
        )
        ttk.Button(controls, text="Export PNG", command=self._save_molecule_image).pack(fill="x", pady=(8, 0))

        self.molecule_canvas, self.molecule_gallery = self._build_scrollable_molecule_gallery(output)
        self.molecule_canvas.pack(side="left", fill="both", expand=True)
        molecule_scrollbar = ttk.Scrollbar(output, orient="vertical", command=self.molecule_canvas.yview)
        molecule_scrollbar.pack(side="right", fill="y")
        self.molecule_canvas.configure(yscrollcommand=molecule_scrollbar.set)
        self.molecule_canvas.bind("<Configure>", self._schedule_molecule_resize, add="+")
        self._refresh_formula_options()

    def _generate_molecule(self) -> None:
        """Render the selected SMILES string in the molecule tab."""

        try:
            smiles = self.smiles_value.get()
            if not smiles:
                raise ValueError("Select a SMILES string.")
            self._molecule_smiles = smiles
            self._molecule_gallery_mode = "single"
            self._render_molecule_image()
        except Exception as exc:  # pragma: no cover - GUI feedback.
            messagebox.showerror("SMILES visualization failed", str(exc))

    def _generate_all_molecules(self) -> None:
        """Render every observed structure for the selected species."""

        try:
            smiles_values = list(self.smiles_combo["values"])
            if not smiles_values:
                raise ValueError("Select a species with at least one SMILES string.")
            self._molecule_smiles = None
            self._molecule_gallery_mode = "all"
            self._render_smiles_gallery(
                self.molecule_gallery,
                smiles_values,
                photo_attribute="_molecule_gallery_photos",
                variable_attribute="_molecule_gallery_vars",
                image_size=MOLECULE_THUMBNAIL_SIZE,
                canvas=self.molecule_canvas,
            )
        except Exception as exc:  # pragma: no cover - GUI feedback.
            messagebox.showerror("SMILES visualization failed", str(exc))

    def _schedule_molecule_resize(self, _event=None) -> None:
        """Debounce molecule image resizing after output-area changes."""

        if self._molecule_gallery_mode != "single" or not self._molecule_smiles:
            return
        if self._molecule_resize_job is not None:
            self.root.after_cancel(self._molecule_resize_job)
        self._molecule_resize_job = self.root.after(MOLECULE_RESIZE_DEBOUNCE_MS, self._render_molecule_image)

    def _render_molecule_image(self) -> None:
        """Render the current molecule image at the available display size."""

        self._molecule_resize_job = None
        if not self._molecule_smiles:
            return

        image_size = molecule_render_size(
            self.molecule_canvas.winfo_width(),
            self.molecule_canvas.winfo_height(),
        )
        self._molecule_image_size = image_size
        self._render_smiles_gallery(
            self.molecule_gallery,
            [self._molecule_smiles],
            photo_attribute="_molecule_gallery_photos",
            variable_attribute="_molecule_gallery_vars",
            image_size=image_size,
            canvas=self.molecule_canvas,
            columns=1,
        )

    def _save_molecule_image(self) -> None:
        """Save the current molecule rendering to an image file."""

        if not self._molecule_smiles:
            messagebox.showerror("Save failed", "Generate a molecule image before saving.")
            return

        filename = self._ask_image_output_path(
            "Export molecule image",
            "molecule_visualization.png",
            filetypes=PNG_FILETYPES,
        )
        if not filename:
            return
        output_path = image_output_path(filename).with_suffix(".png")
        image_size = self._molecule_image_size or molecule_render_size(
            self.molecule_canvas.winfo_width(),
            self.molecule_canvas.winfo_height(),
        )
        molecule_image(self._molecule_smiles, size=image_size).save(output_path, format="PNG")
        messagebox.showinfo("PNG exported", f"Exported PNG to {output_path}")

    def _refresh_formula_options(self) -> None:
        """Refresh formula options for the selected SMILES simulation."""

        simulation = self._selected_smiles_simulation()
        formulas = formulas_for_simulation(simulation.chem_formulas) if simulation and simulation.chem_formulas else []
        self.smiles_formula_combo.configure(values=formulas)
        self.smiles_formula.set(formulas[0] if formulas else "")
        self._refresh_smiles_options()

    def _refresh_smiles_options(self) -> None:
        """Refresh SMILES options for the selected formula."""

        simulation = self._selected_smiles_simulation()
        formula = self.smiles_formula.get()
        values = []
        if simulation and simulation.chem_formulas and simulation.smiles and formula:
            values = smiles_for_formula(simulation.chem_formulas, simulation.smiles, formula)
        self.smiles_combo.configure(values=values)
        self.smiles_value.set(values[0] if values else "")
        self._refresh_smiles_metadata()

    def _refresh_smiles_metadata(self) -> None:
        """Show charge, ion-candidate, and quality information for a SMILES."""

        simulation = self._selected_smiles_simulation()
        summary = molecule_observation_summary(simulation, self.smiles_value.get()) if simulation else ""
        self.smiles_metadata.set(summary)

    def _selected_smiles_simulation(self):
        """Return the simulation selected in the SMILES tab, if any."""

        value = self.smiles_simulation.get()
        if not value:
            return None
        return self.project.simulation(int(value))

    def _build_scrollable_molecule_gallery(self, parent: ttk.Frame) -> tuple[tk.Canvas, ttk.Frame]:
        """Create a scrollable frame for molecule image galleries."""

        canvas = tk.Canvas(parent, highlightthickness=0)
        gallery = ttk.Frame(canvas)
        window_id = canvas.create_window((0, 0), window=gallery, anchor="nw")

        def sync_scroll_region(_event=None) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def sync_gallery_width(event) -> None:
            canvas.itemconfigure(window_id, width=event.width)

        gallery.bind("<Configure>", sync_scroll_region)
        canvas.bind("<Configure>", sync_gallery_width, add="+")
        return canvas, gallery

    def _render_smiles_gallery(
        self,
        gallery: ttk.Frame,
        smiles_values: list[str],
        *,
        photo_attribute: str,
        variable_attribute: str,
        image_size: tuple[int, int],
        canvas: tk.Canvas,
        columns: int | None = None,
        title: str | None = None,
    ) -> None:
        """Render copyable SMILES tiles into ``gallery``."""

        groups = [(title, smiles_values)] if title else [(None, smiles_values)]
        self._render_grouped_smiles_gallery(
            gallery,
            groups,
            photo_attribute=photo_attribute,
            variable_attribute=variable_attribute,
            image_size=image_size,
            canvas=canvas,
            columns=columns,
        )

    def _render_grouped_smiles_gallery(
        self,
        gallery: ttk.Frame,
        groups: list[tuple[str | None, list[str]]],
        *,
        photo_attribute: str,
        variable_attribute: str,
        image_size: tuple[int, int],
        canvas: tk.Canvas,
        columns: int | None = None,
    ) -> None:
        """Render one or more labeled groups of copyable SMILES tiles."""

        for child in gallery.winfo_children():
            child.destroy()

        photos = []
        variables = []

        if columns is None:
            width = max(canvas.winfo_width(), MOLECULE_GALLERY_MIN_COLUMN_WIDTH)
            columns = max(1, width // MOLECULE_GALLERY_MIN_COLUMN_WIDTH)

        current_row = 0
        for title, smiles_values in groups:
            if title:
                ttk.Label(gallery, text=title).grid(
                    row=current_row,
                    column=0,
                    columnspan=columns,
                    sticky="w",
                    padx=8,
                    pady=(8, 2),
                )
                current_row += 1
            for index, smiles in enumerate(smiles_values):
                row = current_row + index // columns
                column = index % columns
                tile = ttk.Frame(gallery, padding=6)
                tile.grid(row=row, column=column, sticky="nsew", padx=4, pady=4)
                gallery.columnconfigure(column, weight=1, uniform="molecule_gallery")

                photo = molecule_photo_image(smiles, size=image_size)
                photos.append(photo)
                ttk.Label(tile, image=photo, anchor="center").pack(fill="x")

                value = tk.StringVar(value=smiles)
                variables.append(value)
                ttk.Entry(tile, textvariable=value, state="readonly").pack(fill="x", pady=(6, 4))
                ttk.Button(tile, text="Copy SMILES", command=lambda text=smiles: self._copy_text(text)).pack(fill="x")
            current_row += max(1, (len(smiles_values) + columns - 1) // columns)

        setattr(self, photo_attribute, photos)
        setattr(self, variable_attribute, variables)
        gallery.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox("all"))

    def _copy_text(self, text: str) -> None:
        """Copy text to the system clipboard."""

        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update_idletasks()
