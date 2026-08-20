"""System-state Snapshot tab for the Tkinter GUI."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from lammpalyze.snapshot import (
    atom_snapshot_table,
    is_snapshot_notation_column,
    molecule_snapshot_table,
)

ATOM_VIEW = "Atoms"
MOLECULE_VIEW = "Molecules"


class SnapshotTabMixin:
    """Build and manage atom and molecule snapshot tables."""

    def _build_snapshot_tab(self, parent: ttk.Frame) -> None:
        """Create snapshot selectors and a scrollable state table."""

        container = ttk.Frame(parent)
        container.pack(fill="both", expand=True, padx=12, pady=12)

        controls = ttk.LabelFrame(container, text="Snapshot selection")
        controls.pack(fill="x", pady=(0, 10))
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(3, weight=1)

        self._snapshot_simulations = [
            simulation for simulation in self.project.simulations if simulation.has_bond_data
        ]
        simulation_labels = [
            f"Simulation {simulation.index}" for simulation in self._snapshot_simulations
        ]

        ttk.Label(controls, text="Simulation").grid(
            row=0, column=0, sticky="w", padx=(10, 6), pady=(10, 6)
        )
        self.snapshot_simulation = tk.StringVar()
        self.snapshot_simulation_box = ttk.Combobox(
            controls,
            textvariable=self.snapshot_simulation,
            values=simulation_labels,
            state="readonly" if simulation_labels else "disabled",
        )
        self.snapshot_simulation_box.grid(
            row=0, column=1, sticky="ew", padx=(0, 14), pady=(10, 6)
        )
        self.snapshot_simulation_box.bind(
            "<<ComboboxSelected>>", self._snapshot_simulation_changed
        )

        ttk.Label(controls, text="View").grid(
            row=0, column=2, sticky="w", padx=(0, 6), pady=(10, 6)
        )
        self.snapshot_view = tk.StringVar(value=ATOM_VIEW)
        self.snapshot_view_box = ttk.Combobox(
            controls,
            textvariable=self.snapshot_view,
            values=(ATOM_VIEW, MOLECULE_VIEW),
            state="readonly" if simulation_labels else "disabled",
        )
        self.snapshot_view_box.grid(
            row=0, column=3, sticky="ew", padx=(0, 10), pady=(10, 6)
        )
        self.snapshot_view_box.bind("<<ComboboxSelected>>", self._refresh_snapshot)

        ttk.Label(controls, text="Timestep").grid(
            row=1, column=0, sticky="w", padx=(10, 6), pady=(6, 10)
        )
        self.snapshot_timestep = tk.StringVar()
        self.snapshot_timestep_entry = ttk.Entry(
            controls,
            textvariable=self.snapshot_timestep,
            state="normal" if simulation_labels else "disabled",
        )
        self.snapshot_timestep_entry.grid(
            row=1, column=1, sticky="ew", padx=(0, 14), pady=(6, 10)
        )
        self.snapshot_timestep_entry.bind("<Return>", self._refresh_snapshot)

        ttk.Label(controls, text="Notation").grid(
            row=1, column=2, sticky="w", padx=(0, 6), pady=(6, 10)
        )
        notation_frame = ttk.Frame(controls)
        notation_frame.grid(row=1, column=3, sticky="ew", padx=(0, 10), pady=(6, 10))
        self.snapshot_notation = tk.StringVar(value="formula")
        ttk.Radiobutton(
            notation_frame,
            text="Formula",
            variable=self.snapshot_notation,
            value="formula",
            command=self._refresh_snapshot,
        ).pack(side="left")
        ttk.Radiobutton(
            notation_frame,
            text="SMILES",
            variable=self.snapshot_notation,
            value="smiles",
            command=self._refresh_snapshot,
        ).pack(side="left", padx=(12, 0))
        ttk.Button(
            notation_frame,
            text="Refresh",
            command=self._refresh_snapshot,
        ).pack(side="right")

        table_frame = ttk.LabelFrame(container, text="System state")
        table_frame.pack(fill="both", expand=True)
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)
        self.snapshot_table = ttk.Treeview(table_frame, show="headings")
        y_scrollbar = ttk.Scrollbar(
            table_frame, orient="vertical", command=self.snapshot_table.yview
        )
        x_scrollbar = ttk.Scrollbar(
            table_frame, orient="horizontal", command=self.snapshot_table.xview
        )
        self.snapshot_table.configure(
            yscrollcommand=y_scrollbar.set,
            xscrollcommand=x_scrollbar.set,
        )
        self.snapshot_table.grid(row=0, column=0, sticky="nsew")
        y_scrollbar.grid(row=0, column=1, sticky="ns")
        x_scrollbar.grid(row=1, column=0, sticky="ew")
        self._snapshot_notation_cell = ("", "")
        self.snapshot_table.bind("<ButtonRelease-1>", self._select_snapshot_notation_cell)
        self.snapshot_table.bind("<Double-1>", self._copy_snapshot_notation_from_event)
        self.snapshot_table.bind("<Control-c>", self._copy_snapshot_notation)
        self.snapshot_table.bind("<Control-C>", self._copy_snapshot_notation)

        copy_frame = ttk.Frame(container)
        copy_frame.pack(fill="x", pady=(8, 0))
        ttk.Label(copy_frame, text="Selected notation").pack(side="left", padx=(0, 6))
        self.snapshot_selected_notation = tk.StringVar()
        ttk.Entry(
            copy_frame,
            textvariable=self.snapshot_selected_notation,
            state="readonly",
        ).pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(
            copy_frame,
            text="Copy notation",
            command=self._copy_snapshot_notation,
        ).pack(side="right")

        self.snapshot_status = tk.StringVar(
            value=(
                "Select a simulation snapshot. Calculated mol_id values are zero-based "
                "connected-component indexes."
                if simulation_labels
                else "No simulations with bond-derived molecule data are available."
            )
        )
        ttk.Label(
            container,
            textvariable=self.snapshot_status,
            justify="left",
            wraplength=1000,
        ).pack(fill="x", pady=(8, 0))

        if simulation_labels:
            self.snapshot_simulation_box.current(0)
            self._snapshot_simulation_changed()

    def _selected_snapshot_simulation(self):
        """Return the simulation selected for snapshot display."""

        selection = self.snapshot_simulation_box.current()
        if selection < 0:
            raise ValueError("Select a simulation.")
        return self._snapshot_simulations[selection]

    def _snapshot_simulation_changed(self, _event=None) -> None:
        """Populate available bond-analysis timesteps for one simulation."""

        try:
            simulation = self._selected_snapshot_simulation()
            timesteps = sorted(simulation.smiles_id or {})
            if not timesteps:
                raise ValueError(
                    f"Simulation {simulation.index} has no bond-derived molecule timesteps."
                )
            self.snapshot_timestep_entry.configure(state="normal")
            trajectory_timesteps = (
                set(simulation.trajectory_timesteps())
                if simulation.trajectory_path is not None
                else set()
            )
            aligned_timesteps = [
                timestep for timestep in timesteps if timestep in trajectory_timesteps
            ]
            if aligned_timesteps:
                self.snapshot_timestep.set(str(aligned_timesteps[0]))
            else:
                self.snapshot_timestep.set(str(timesteps[0]))
                self.snapshot_view.set(MOLECULE_VIEW)
            self._refresh_snapshot(_event or "automatic")
        except Exception as exc:  # pragma: no cover - GUI feedback.
            self.snapshot_status.set(str(exc))
            self.snapshot_timestep_entry.configure(state="disabled")
            self._clear_snapshot_table()

    def _refresh_snapshot(self, _event=None) -> None:
        """Build and display the selected atom or molecule snapshot."""

        try:
            simulation = self._selected_snapshot_simulation()
            timestep_text = self.snapshot_timestep.get().strip()
            if not timestep_text:
                raise ValueError("Select a snapshot timestep.")
            timestep = int(timestep_text)
            notation = self.snapshot_notation.get()
            if self.snapshot_view.get() == ATOM_VIEW:
                table = atom_snapshot_table(simulation, timestep, notation)
                status = (
                    f"Showing {len(table.rows)} atoms across molecule observations "
                    f"{', '.join(str(value) for value in table.observation_timesteps)}. "
                    "Calculated mol_id is the zero-based component index at the selected timestep."
                )
            else:
                table = molecule_snapshot_table(simulation, timestep, notation)
                status = (
                    f"Showing {len(table.rows)} calculated molecules at timestep {timestep}. "
                    "Calculated mol_id is the zero-based connected-component index."
                )
            self._display_snapshot_table(table)
            self.snapshot_status.set(status)
        except Exception as exc:  # pragma: no cover - GUI feedback.
            self.snapshot_status.set(str(exc))
            self._clear_snapshot_table()
            if _event is None:
                messagebox.showerror("Snapshot generation failed", str(exc))

    def _display_snapshot_table(self, table) -> None:
        """Render GUI-independent snapshot data in the tree view."""

        self.snapshot_table.delete(*self.snapshot_table.get_children())
        column_keys = [column.key for column in table.columns]
        self.snapshot_table.configure(columns=column_keys)
        for column in table.columns:
            self.snapshot_table.heading(column.key, text=column.heading)
            width = self._snapshot_column_width(column.key)
            self.snapshot_table.column(
                column.key,
                width=width,
                minwidth=min(width, 80),
                anchor="w" if column.key in {"molecule", "atoms"} else "center",
                stretch=column.key in {"molecule", "atoms"},
            )
        for row in table.rows:
            self.snapshot_table.insert("", "end", values=row)
        self._snapshot_notation_cell = ("", "")
        self.snapshot_selected_notation.set("")

    def _clear_snapshot_table(self) -> None:
        """Remove all snapshot rows and dynamic columns."""

        self.snapshot_table.delete(*self.snapshot_table.get_children())
        self.snapshot_table.configure(columns=())
        self._snapshot_notation_cell = ("", "")
        self.snapshot_selected_notation.set("")

    def _select_snapshot_notation_cell(self, event) -> None:
        """Remember a clicked formula or SMILES cell for copying."""

        item_id = self.snapshot_table.identify_row(event.y)
        column_id = self.snapshot_table.identify_column(event.x)
        columns = self.snapshot_table["columns"]
        try:
            column = columns[int(column_id.removeprefix("#")) - 1]
        except (IndexError, ValueError):
            return
        if not item_id or not is_snapshot_notation_column(column):
            self._snapshot_notation_cell = ("", "")
            self.snapshot_selected_notation.set("")
            return
        value = self.snapshot_table.set(item_id, column)
        self._snapshot_notation_cell = (item_id, column)
        self.snapshot_selected_notation.set(value)

    def _copy_snapshot_notation_from_event(self, event) -> str:
        """Select and copy a double-clicked notation cell."""

        self._select_snapshot_notation_cell(event)
        return self._copy_snapshot_notation(event)

    def _copy_snapshot_notation(self, _event=None) -> str:
        """Copy the selected Snapshot formula or SMILES to the clipboard."""

        item_id, column = self._snapshot_notation_cell
        if not item_id or not column:
            if _event is None:
                messagebox.showerror(
                    "Copy failed",
                    "Click a formula or SMILES cell in the Snapshot table first.",
                )
            return "break"
        value = self.snapshot_table.set(item_id, column)
        if not value or value == "—":
            if _event is None:
                messagebox.showerror("Copy failed", "The selected cell has no notation value.")
            return "break"
        self.root.clipboard_clear()
        self.root.clipboard_append(value)
        self.root.update_idletasks()
        self.snapshot_selected_notation.set(value)
        self.snapshot_status.set(f"Copied notation to clipboard: {value}")
        return "break"

    def _snapshot_column_width(self, column_key: str) -> int:
        """Return a practical initial width for a snapshot column."""

        if column_key == "atoms":
            return 480
        if column_key == "molecule":
            return 260 if self.snapshot_notation.get() == "smiles" else 170
        if column_key.startswith("state_"):
            return 230 if self.snapshot_notation.get() == "smiles" else 145
        return 125
