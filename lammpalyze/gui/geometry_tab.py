"""Trajectory geometry tab for the Tkinter GUI."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from lammpalyze.geometry import (
    MoleculeMembershipFilter,
    atom_id_groups,
    compute_distances,
    compute_geometry,
    compute_intramolecular_distances,
    distance_pairs,
    parse_atom_ids,
    parse_distance_selections,
    parse_intramolecular_groups,
    parse_molecule_descriptors,
)
from lammpalyze.geometry_plotting import plot_geometry


_DISTANCE_SELECTION_KINDS = {
    "Atom": "atom",
    "COM (atom IDs)": "com_atoms",
    "COM (molecule IDs)": "com_molecule",
    "Plane (3 atom IDs)": "plane",
}
_INTRAMOLECULAR_KINDS = {
    "Intramolecule (atom IDs)": "atoms",
    "Intramolecule (molecule IDs)": "molecules",
}


class GeometryTabMixin:
    """Build and manage trajectory-derived distance and angle plots."""

    def _build_geometry_tab(self, parent: ttk.Frame) -> None:
        """Create geometry selectors and a plot area."""

        controls, self._geometry_plot_area = self._computed_tab_layout(parent)
        self._geometry_simulations = [
            simulation
            for simulation in self.project.simulations
            if simulation.trajectory_path is not None
        ]

        ttk.Label(controls, text="Simulations").pack(anchor="w")
        self.geometry_simulation_list = tk.Listbox(
            controls,
            selectmode="multiple",
            exportselection=False,
            height=6,
        )
        for simulation in self._geometry_simulations:
            self.geometry_simulation_list.insert("end", f"Simulation {simulation.index}")
        if self.geometry_simulation_list.size():
            self.geometry_simulation_list.select_set(0, "end")
        self.geometry_simulation_list.pack(fill="x", pady=(0, 12))

        self.geometry_kind = tk.StringVar(value="Distance")
        ttk.Label(controls, text="Measurement").pack(anchor="w")
        kind_box = ttk.Combobox(
            controls,
            textvariable=self.geometry_kind,
            values=["Distance", "Angle"],
            state="readonly",
        )
        kind_box.bind("<<ComboboxSelected>>", lambda _event: self._update_geometry_fields())
        kind_box.pack(fill="x", pady=(0, 12))

        self.geometry_selection_1 = tk.StringVar(value="Atom")
        self.geometry_selection_2 = tk.StringVar(value="Atom")
        ttk.Label(controls, text="First distance selection").pack(anchor="w")
        self.geometry_selection_1_box = ttk.Combobox(
            controls,
            textvariable=self.geometry_selection_1,
            values=[*_DISTANCE_SELECTION_KINDS, *_INTRAMOLECULAR_KINDS],
            state="readonly",
        )
        self.geometry_selection_1_box.bind(
            "<<ComboboxSelected>>", lambda _event: self._update_geometry_fields()
        )
        self.geometry_selection_1_box.pack(fill="x", pady=(0, 8))
        ttk.Label(controls, text="Second distance selection").pack(anchor="w")
        self.geometry_selection_2_box = ttk.Combobox(
            controls,
            textvariable=self.geometry_selection_2,
            values=[*_DISTANCE_SELECTION_KINDS, *_INTRAMOLECULAR_KINDS],
            state="readonly",
        )
        self.geometry_selection_2_box.bind(
            "<<ComboboxSelected>>", lambda _event: self._update_geometry_fields()
        )
        self.geometry_selection_2_box.pack(fill="x", pady=(0, 12))

        self.geometry_atom_1 = tk.StringVar()
        self.geometry_atom_2 = tk.StringVar()
        self.geometry_atom_3 = tk.StringVar()
        self.geometry_atom_1_label = ttk.Label(controls, text="Atom 1 ID(s)")
        self.geometry_atom_1_label.pack(anchor="w")
        ttk.Entry(controls, textvariable=self.geometry_atom_1).pack(fill="x", pady=(0, 8))
        self.geometry_atom_2_label = ttk.Label(controls, text="Atom 2 ID(s)")
        self.geometry_atom_2_label.pack(anchor="w")
        self.geometry_atom_2_entry = ttk.Entry(controls, textvariable=self.geometry_atom_2)
        self.geometry_atom_2_entry.pack(fill="x", pady=(0, 8))
        self.geometry_atom_3_label = ttk.Label(controls, text="Atom 3 ID(s)")
        self.geometry_atom_3_label.pack(anchor="w")
        self.geometry_atom_3_entry = ttk.Entry(controls, textvariable=self.geometry_atom_3)
        self.geometry_atom_3_entry.pack(fill="x", pady=(0, 4))
        self.geometry_hint = ttk.Label(
            controls,
            text="Equal-length lists are paired by position; unequal lists use all combinations.",
            wraplength=270,
            justify="left",
        )
        self.geometry_hint.pack(anchor="w", pady=(0, 12))

        self.geometry_membership_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            controls,
            text="Only while all measurement atoms are part of",
            variable=self.geometry_membership_enabled,
            command=self._update_geometry_membership_state,
        ).pack(anchor="w", pady=(0, 6))
        self.geometry_membership_notation = tk.StringVar(value="Auto-detect")
        self.geometry_membership_notation_box = ttk.Combobox(
            controls,
            textvariable=self.geometry_membership_notation,
            values=["Auto-detect", "Chemical formula", "SMILES"],
            state="disabled",
        )
        self.geometry_membership_notation_box.pack(fill="x", pady=(0, 6))
        self.geometry_membership_descriptors = tk.StringVar()
        self.geometry_membership_descriptors_entry = ttk.Entry(
            controls,
            textvariable=self.geometry_membership_descriptors,
            state="disabled",
        )
        self.geometry_membership_descriptors_entry.pack(fill="x", pady=(0, 4))
        ttk.Label(
            controls,
            text='One value or a string list, e.g. ["C3H4LiO3", "C2H4O2"].',
            wraplength=270,
            justify="left",
        ).pack(anchor="w", pady=(0, 12))

        self.geometry_molecule_atoms = tk.StringVar()
        ttk.Label(controls, text="Chemical-state atom ID(s) (optional)").pack(anchor="w")
        ttk.Entry(controls, textvariable=self.geometry_molecule_atoms).pack(
            fill="x",
            pady=(0, 8),
        )
        self.geometry_molecule_notation = tk.StringVar(value="Chemical formula")
        ttk.Label(controls, text="Chemical-state notation").pack(anchor="w")
        ttk.Combobox(
            controls,
            textvariable=self.geometry_molecule_notation,
            values=["Chemical formula", "SMILES"],
            state="readonly",
        ).pack(fill="x", pady=(0, 12))

        self._build_computed_plot_options(controls, "geometry")
        ttk.Button(controls, text="Plot", command=self._plot_geometry).pack(fill="x")
        ttk.Button(controls, text="Export PNG", command=self._save_geometry_plot).pack(
            fill="x", pady=(8, 0)
        )
        self._update_geometry_fields()

    def _update_geometry_membership_state(self) -> None:
        """Enable molecule-descriptor controls only when filtering is requested."""

        state = "readonly" if self.geometry_membership_enabled.get() else "disabled"
        self.geometry_membership_notation_box.configure(state=state)
        self.geometry_membership_descriptors_entry.configure(
            state="normal" if self.geometry_membership_enabled.get() else "disabled"
        )

    def _update_geometry_fields(self) -> None:
        """Update endpoint controls and field guidance for the selected measurement."""

        is_angle = self.geometry_kind.get() == "Angle"
        if not is_angle and self.geometry_selection_2.get() in _INTRAMOLECULAR_KINDS:
            self.geometry_selection_1.set(self.geometry_selection_2.get())
            self.geometry_selection_2.set("Atom")
        intramolecular = self.geometry_selection_1.get() in _INTRAMOLECULAR_KINDS
        self.geometry_selection_1_box.configure(state="disabled" if is_angle else "readonly")
        self.geometry_selection_2_box.configure(
            state="disabled" if is_angle or intramolecular else "readonly"
        )
        self.geometry_atom_2_entry.configure(
            state="disabled" if not is_angle and intramolecular else "normal"
        )
        self.geometry_atom_3_entry.configure(state="normal" if is_angle else "disabled")
        if is_angle:
            self.geometry_atom_1_label.configure(text="Atom 1 ID(s)")
            self.geometry_atom_2_label.configure(text="Atom 2 ID(s) (angle vertex)")
            self.geometry_atom_3_label.configure(text="Atom 3 ID(s)")
            self.geometry_hint.configure(
                text="Angle lists are paired by position; all three lists must be the same length."
            )
            return

        self.geometry_atom_1_label.configure(
            text=self._geometry_distance_field_label(self.geometry_selection_1.get(), "First")
        )
        self.geometry_atom_2_label.configure(
            text=(
                "Second selection (not used in intramolecule mode)"
                if intramolecular
                else self._geometry_distance_field_label(
                    self.geometry_selection_2.get(), "Second"
                )
            )
        )
        self.geometry_atom_3_label.configure(text="Atom 3 ID(s) (angles only)")
        self.geometry_hint.configure(
            text=(
                "Use [1,3,4] for one atom group or [[1,3,4],[7,8,9]] for several groups."
                if intramolecular
                else "Equal-length selections are paired by position; unequal selections use "
                "all combinations. Nested lists define multiple COMs or planes."
            )
        )

    @staticmethod
    def _geometry_distance_field_label(selection: str, ordinal: str) -> str:
        """Return a concise input label for one distance selection type."""

        labels = {
            "Atom": "atom ID(s)",
            "COM (atom IDs)": "COM atom IDs",
            "COM (molecule IDs)": "COM molecule ID(s)",
            "Plane (3 atom IDs)": "plane atom IDs (three per plane)",
            "Intramolecule (atom IDs)": "intramolecular atom group(s)",
            "Intramolecule (molecule IDs)": "intramolecular molecule ID(s)",
        }
        return f"{ordinal} {labels[selection]}"

    def _plot_geometry(self) -> None:
        """Calculate and render the selected trajectory geometry."""

        try:
            simulations = [
                self._geometry_simulations[index]
                for index in self.geometry_simulation_list.curselection()
            ]
            if not simulations:
                raise ValueError("Select at least one simulation.")
            kind = self.geometry_kind.get().lower()
            membership_filter = self._geometry_membership_filter()
            if kind == "angle":
                columns = [
                    parse_atom_ids(self.geometry_atom_1.get()),
                    parse_atom_ids(self.geometry_atom_2.get()),
                    parse_atom_ids(self.geometry_atom_3.get()),
                ]
                groups = atom_id_groups(*columns)
            options = self._computed_plot_options("geometry")
            if kind == "angle":
                results = compute_geometry(
                    simulations,
                    kind,
                    groups,
                    timestep_range=options["step_range"],
                    membership_filter=membership_filter,
                )
            elif self.geometry_selection_1.get() in _INTRAMOLECULAR_KINDS:
                intramolecular_kind = _INTRAMOLECULAR_KINDS[self.geometry_selection_1.get()]
                groups = parse_intramolecular_groups(
                    self.geometry_atom_1.get(), intramolecular_kind
                )
                results = compute_intramolecular_distances(
                    simulations,
                    groups,
                    intramolecular_kind,
                    timestep_range=options["step_range"],
                    membership_filter=membership_filter,
                )
            else:
                first_kind = _DISTANCE_SELECTION_KINDS[self.geometry_selection_1.get()]
                second_kind = _DISTANCE_SELECTION_KINDS[self.geometry_selection_2.get()]
                pairs = distance_pairs(
                    parse_distance_selections(self.geometry_atom_1.get(), first_kind),
                    parse_distance_selections(self.geometry_atom_2.get(), second_kind),
                )
                results = compute_distances(
                    simulations,
                    pairs,
                    timestep_range=options["step_range"],
                    membership_filter=membership_filter,
                )
            molecule_atoms = self._geometry_molecule_atom_ids()
            figure = plot_geometry(
                results,
                kind,
                simulations=simulations,
                molecule_atom_ids=molecule_atoms,
                molecule_notation=(
                    "formula"
                    if self.geometry_molecule_notation.get() == "Chemical formula"
                    else "smiles"
                ),
                plot_settings=self._plot_settings(),
                **options,
            )
            self._replace_canvas("_geometry_canvas", self._geometry_plot_area, figure)
        except Exception as exc:  # pragma: no cover - GUI feedback.
            messagebox.showerror("Geometry plotting failed", str(exc))

    def _geometry_molecule_atom_ids(self) -> list[int] | None:
        """Return optional atom IDs for chemical-state overlay."""

        value = self.geometry_molecule_atoms.get().strip()
        return parse_atom_ids(value) if value else None

    def _geometry_membership_filter(self) -> MoleculeMembershipFilter | None:
        """Return the optional same-component molecule descriptor filter."""

        if not self.geometry_membership_enabled.get():
            return None
        notation = {
            "Auto-detect": "auto",
            "Chemical formula": "formula",
            "SMILES": "smiles",
        }[self.geometry_membership_notation.get()]
        descriptors = parse_molecule_descriptors(
            self.geometry_membership_descriptors.get()
        )
        return MoleculeMembershipFilter(tuple(descriptors), notation)

    def _save_geometry_plot(self) -> None:
        """Export the displayed trajectory geometry plot as PNG."""

        kind = self.geometry_kind.get().lower() or "geometry"
        self._export_canvas_figure_png(
            self._geometry_canvas,
            "Export trajectory geometry",
            f"trajectory_{kind}.png",
        )
