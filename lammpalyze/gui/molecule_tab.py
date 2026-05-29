"""Molecule-visualization tab for the Tkinter GUI."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from lammpalyze.gui.helpers import (
    MOLECULE_RESIZE_DEBOUNCE_MS,
    RASTER_IMAGE_FILETYPES,
    image_output_path,
    molecule_render_size,
)
from lammpalyze.smiles import formulas_for_simulation, molecule_image, molecule_photo_image, smiles_for_formula


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
        self.smiles_formula_combo = ttk.Combobox(controls, textvariable=self.smiles_formula, state="readonly")
        self.smiles_formula_combo.pack(fill="x", pady=(0, 12))
        self.smiles_formula_combo.bind("<<ComboboxSelected>>", lambda _event: self._refresh_smiles_options())

        ttk.Label(controls, text="SMILES").pack(anchor="w")
        self.smiles_combo = ttk.Combobox(controls, textvariable=self.smiles_value, state="readonly", width=42)
        self.smiles_combo.pack(fill="x", pady=(0, 12))

        ttk.Button(controls, text="Generate", command=self._generate_molecule).pack(fill="x")
        ttk.Button(controls, text="Save image", command=self._save_molecule_image).pack(fill="x", pady=(8, 0))

        self.molecule_label = ttk.Label(output, anchor="center")
        self.molecule_label.pack(fill="both", expand=True)
        output.bind("<Configure>", self._schedule_molecule_resize)
        self._refresh_formula_options()

    def _generate_molecule(self) -> None:
        """Render the selected SMILES string in the molecule tab."""

        try:
            smiles = self.smiles_value.get()
            if not smiles:
                raise ValueError("Select a SMILES string.")
            self._molecule_smiles = smiles
            self._render_molecule_image()
        except Exception as exc:  # pragma: no cover - GUI feedback.
            messagebox.showerror("SMILES visualization failed", str(exc))

    def _schedule_molecule_resize(self, _event=None) -> None:
        """Debounce molecule image resizing after output-area changes."""

        if not self._molecule_smiles:
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
            self.molecule_label.winfo_width(),
            self.molecule_label.winfo_height(),
        )
        self._molecule_image_size = image_size
        self._molecule_photo = molecule_photo_image(self._molecule_smiles, size=image_size)
        self.molecule_label.configure(image=self._molecule_photo)

    def _save_molecule_image(self) -> None:
        """Save the current molecule rendering to an image file."""

        if not self._molecule_smiles:
            messagebox.showerror("Save failed", "Generate a molecule image before saving.")
            return

        filename = self._ask_image_output_path(
            "Save molecule image",
            "molecule_visualization.png",
            filetypes=RASTER_IMAGE_FILETYPES,
        )
        if not filename:
            return
        output_path = image_output_path(filename)
        image_size = self._molecule_image_size or molecule_render_size(
            self.molecule_label.winfo_width(),
            self.molecule_label.winfo_height(),
        )
        molecule_image(self._molecule_smiles, size=image_size).save(output_path)
        messagebox.showinfo("Image saved", f"Saved image to {output_path}")

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

    def _selected_smiles_simulation(self):
        """Return the simulation selected in the SMILES tab, if any."""

        value = self.smiles_simulation.get()
        if not value:
            return None
        return self.project.simulation(int(value))
