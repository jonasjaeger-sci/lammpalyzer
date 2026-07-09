"""Trajectory distance-and-angle tab for the Tkinter GUI."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from lammpalyze.geometry import atom_id_groups, compute_geometry, parse_atom_ids
from lammpalyze.geometry_plotting import plot_geometry


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

        self.geometry_atom_1 = tk.StringVar()
        self.geometry_atom_2 = tk.StringVar()
        self.geometry_atom_3 = tk.StringVar()
        ttk.Label(controls, text="Atom 1 ID(s)").pack(anchor="w")
        ttk.Entry(controls, textvariable=self.geometry_atom_1).pack(fill="x", pady=(0, 8))
        self.geometry_atom_2_label = ttk.Label(controls, text="Atom 2 ID(s)")
        self.geometry_atom_2_label.pack(anchor="w")
        ttk.Entry(controls, textvariable=self.geometry_atom_2).pack(fill="x", pady=(0, 8))
        self.geometry_atom_3_label = ttk.Label(controls, text="Atom 3 ID(s)")
        self.geometry_atom_3_label.pack(anchor="w")
        self.geometry_atom_3_entry = ttk.Entry(controls, textvariable=self.geometry_atom_3)
        self.geometry_atom_3_entry.pack(fill="x", pady=(0, 4))
        self.geometry_hint = ttk.Label(
            controls,
            text="Lists are paired by position, e.g. [1, 4] and [2, 5].",
            wraplength=270,
            justify="left",
        )
        self.geometry_hint.pack(anchor="w", pady=(0, 12))

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

    def _update_geometry_fields(self) -> None:
        """Enable the third atom only for angle measurements."""

        is_angle = self.geometry_kind.get() == "Angle"
        self.geometry_atom_3_entry.configure(state="normal" if is_angle else "disabled")
        self.geometry_atom_3_label.configure(
            text="Atom 3 ID(s)" if is_angle else "Atom 3 ID(s) (angles only)"
        )
        self.geometry_atom_2_label.configure(
            text="Atom 2 ID(s) (angle vertex)" if is_angle else "Atom 2 ID(s)"
        )

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
            columns = [
                parse_atom_ids(self.geometry_atom_1.get()),
                parse_atom_ids(self.geometry_atom_2.get()),
            ]
            if kind == "angle":
                columns.append(parse_atom_ids(self.geometry_atom_3.get()))
            groups = atom_id_groups(*columns)
            options = self._computed_plot_options("geometry")
            results = compute_geometry(
                simulations,
                kind,
                groups,
                timestep_range=options["step_range"],
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
                **options,
            )
            self._replace_canvas("_geometry_canvas", self._geometry_plot_area, figure)
        except Exception as exc:  # pragma: no cover - GUI feedback.
            messagebox.showerror("Geometry plotting failed", str(exc))

    def _geometry_molecule_atom_ids(self) -> list[int] | None:
        """Return optional atom IDs for chemical-state overlay."""

        value = self.geometry_molecule_atoms.get().strip()
        return parse_atom_ids(value) if value else None

    def _save_geometry_plot(self) -> None:
        """Export the displayed trajectory geometry plot as PNG."""

        kind = self.geometry_kind.get().lower() or "geometry"
        self._export_canvas_figure_png(
            self._geometry_canvas,
            "Export trajectory geometry",
            f"trajectory_{kind}.png",
        )
