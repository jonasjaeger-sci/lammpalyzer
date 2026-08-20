"""Atomic index-list generator tab for the Tkinter GUI."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from lammpalyze.atomic_indices import (
    atomic_ids_from_frame,
    format_atomic_id_list,
    parse_atomic_index_selection,
    parse_repeat_count,
)
from lammpalyze.parsers import iter_lammpstrj_frames, trajectory_atom_columns

ATOM_TYPE_MODE = "By atom type"
MOLECULE_ID_MODE = "By molecule ID"


class AtomicIndicesTabMixin:
    """Build and manage atom-ID list generation from trajectory metadata."""

    def _build_atomic_indices_tab(self, parent: ttk.Frame) -> None:
        """Create trajectory selection, matching, repetition, and copy controls."""

        container = ttk.Frame(parent)
        container.pack(fill="both", expand=True, padx=16, pady=16)
        ttk.Label(
            container,
            text=(
                "Generate atom-ID lists from the first trajectory frame. Atom types come from "
                "the type column; molecule IDs come only from a trajectory mol column."
            ),
            justify="left",
            wraplength=800,
        ).pack(anchor="w", pady=(0, 12))

        controls = ttk.LabelFrame(container, text="Atomic index selection")
        controls.pack(fill="x")
        controls.columnconfigure(1, weight=1)

        self._atomic_index_simulations = [
            simulation
            for simulation in self.project.simulations
            if simulation.trajectory_path is not None
        ]
        self._atomic_index_first_frames = {}
        self._atomic_index_mol_availability = {}
        simulation_labels = [
            f"Simulation {simulation.index}"
            for simulation in self._atomic_index_simulations
        ]

        ttk.Label(controls, text="Simulation").grid(
            row=0, column=0, sticky="w", padx=(10, 8), pady=(10, 6)
        )
        self.atomic_index_simulation = tk.StringVar()
        self.atomic_index_simulation_box = ttk.Combobox(
            controls,
            textvariable=self.atomic_index_simulation,
            values=simulation_labels,
            state="readonly" if simulation_labels else "disabled",
        )
        self.atomic_index_simulation_box.grid(
            row=0, column=1, sticky="ew", padx=(0, 10), pady=(10, 6)
        )
        self.atomic_index_simulation_box.bind(
            "<<ComboboxSelected>>",
            self._atomic_index_simulation_changed,
        )

        ttk.Label(controls, text="Select atoms").grid(
            row=1, column=0, sticky="w", padx=(10, 8), pady=6
        )
        self.atomic_index_mode = tk.StringVar(value=ATOM_TYPE_MODE)
        self.atomic_index_mode_box = ttk.Combobox(
            controls,
            textvariable=self.atomic_index_mode,
            values=(ATOM_TYPE_MODE,),
            state="readonly" if simulation_labels else "disabled",
        )
        self.atomic_index_mode_box.grid(row=1, column=1, sticky="ew", padx=(0, 10), pady=6)
        self.atomic_index_mode_box.bind(
            "<<ComboboxSelected>>",
            self._atomic_index_mode_changed,
        )

        self.atomic_index_values_label = ttk.Label(controls, text="Atom types")
        self.atomic_index_values_label.grid(
            row=2, column=0, sticky="w", padx=(10, 8), pady=6
        )
        self.atomic_index_values = tk.StringVar()
        ttk.Entry(controls, textvariable=self.atomic_index_values).grid(
            row=2, column=1, sticky="ew", padx=(0, 10), pady=6
        )
        ttk.Label(
            controls,
            text="Use commas or spaces; * is an inclusive range, e.g. 1,3,4*7.",
            justify="left",
        ).grid(row=3, column=1, sticky="w", padx=(0, 10), pady=(0, 8))

        repeat_frame = ttk.Frame(controls)
        repeat_frame.grid(row=4, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 8))
        self.atomic_index_repeat_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            repeat_frame,
            text="Repeat each atom ID",
            variable=self.atomic_index_repeat_enabled,
            command=self._update_atomic_index_repeat_state,
        ).pack(side="left")
        ttk.Label(repeat_frame, text="Count").pack(side="left", padx=(20, 6))
        self.atomic_index_repeat = tk.StringVar(value="3")
        self.atomic_index_repeat_entry = ttk.Entry(
            repeat_frame,
            textvariable=self.atomic_index_repeat,
            width=8,
            state="disabled",
        )
        self.atomic_index_repeat_entry.pack(side="left")

        self.atomic_index_availability = tk.StringVar()
        ttk.Label(
            controls,
            textvariable=self.atomic_index_availability,
            justify="left",
            wraplength=780,
        ).grid(row=5, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 8))

        self.atomic_index_generate_button = ttk.Button(
            controls,
            text="Generate atom-ID list",
            command=self._generate_atomic_index_list,
            state="normal" if simulation_labels else "disabled",
        )
        self.atomic_index_generate_button.grid(
            row=6, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10)
        )

        output_frame = ttk.LabelFrame(container, text="Generated atom IDs")
        output_frame.pack(fill="both", expand=True, pady=(12, 0))
        output_frame.rowconfigure(0, weight=1)
        output_frame.columnconfigure(0, weight=1)
        self.atomic_index_output = tk.Text(output_frame, height=10, wrap="word")
        output_scrollbar = ttk.Scrollbar(
            output_frame,
            orient="vertical",
            command=self.atomic_index_output.yview,
        )
        self.atomic_index_output.configure(yscrollcommand=output_scrollbar.set)
        self.atomic_index_output.grid(row=0, column=0, sticky="nsew", padx=(8, 0), pady=8)
        output_scrollbar.grid(row=0, column=1, sticky="ns", padx=(0, 8), pady=8)

        output_actions = ttk.Frame(output_frame)
        output_actions.grid(row=1, column=0, columnspan=2, sticky="ew", padx=8, pady=(0, 8))
        self.atomic_index_status = tk.StringVar(
            value="Generate a list, edit it if needed, then copy it."
        )
        ttk.Label(output_actions, textvariable=self.atomic_index_status).pack(side="left")
        ttk.Button(
            output_actions,
            text="Copy list",
            command=self._copy_atomic_index_list,
        ).pack(side="right")

        if simulation_labels:
            self.atomic_index_simulation_box.current(0)
            self._atomic_index_simulation_changed()
        else:
            self.atomic_index_availability.set("No configured trajectory files are available.")

    def _selected_atomic_index_simulation(self):
        """Return the trajectory-backed simulation selected in the combobox."""

        selection = self.atomic_index_simulation_box.current()
        if selection < 0:
            raise ValueError("Select a simulation.")
        return self._atomic_index_simulations[selection]

    def _atomic_index_simulation_changed(self, _event=None) -> None:
        """Update molecule-ID availability for the selected trajectory."""

        try:
            simulation = self._selected_atomic_index_simulation()
            if simulation.index not in self._atomic_index_mol_availability:
                columns = trajectory_atom_columns(simulation.trajectory_path)
                self._atomic_index_mol_availability[simulation.index] = "mol" in columns
            has_mol = self._atomic_index_mol_availability[simulation.index]
            modes = (ATOM_TYPE_MODE, MOLECULE_ID_MODE) if has_mol else (ATOM_TYPE_MODE,)
            self.atomic_index_mode_box.configure(values=modes)
            if self.atomic_index_mode.get() not in modes:
                self.atomic_index_mode.set(ATOM_TYPE_MODE)
            availability = (
                "Molecule-ID selection is available: the first atom table contains mol."
                if has_mol
                else "Molecule-ID selection is unavailable: the first atom table has no mol column."
            )
            self.atomic_index_availability.set(availability)
            self.atomic_index_generate_button.configure(state="normal")
            self._atomic_index_mode_changed()
        except Exception as exc:  # pragma: no cover - GUI feedback.
            self.atomic_index_availability.set(str(exc))
            self.atomic_index_generate_button.configure(state="disabled")

    def _atomic_index_mode_changed(self, _event=None) -> None:
        """Update the selector label for atom-type or molecule-ID mode."""

        label = "Molecule IDs" if self.atomic_index_mode.get() == MOLECULE_ID_MODE else "Atom types"
        self.atomic_index_values_label.configure(text=label)

    def _update_atomic_index_repeat_state(self) -> None:
        """Enable the repetition count only when repetition is requested."""

        state = "normal" if self.atomic_index_repeat_enabled.get() else "disabled"
        self.atomic_index_repeat_entry.configure(state=state)

    def _generate_atomic_index_list(self) -> None:
        """Generate and display atom IDs from the selected first frame."""

        try:
            simulation = self._selected_atomic_index_simulation()
            mode = (
                "molecule_id"
                if self.atomic_index_mode.get() == MOLECULE_ID_MODE
                else "atom_type"
            )
            selected_values = parse_atomic_index_selection(
                self.atomic_index_values.get(),
                allow_zero=mode == "molecule_id",
            )
            repeat = (
                parse_repeat_count(self.atomic_index_repeat.get())
                if self.atomic_index_repeat_enabled.get()
                else 1
            )
            if simulation.index not in self._atomic_index_first_frames:
                frames = iter_lammpstrj_frames(simulation.trajectory_path)
                try:
                    self._atomic_index_first_frames[simulation.index] = next(frames)
                except StopIteration as exc:
                    raise ValueError(
                        f"Simulation {simulation.index} has no trajectory frames."
                    ) from exc
                finally:
                    frames.close()
            frame = self._atomic_index_first_frames[simulation.index]
            atom_ids = atomic_ids_from_frame(
                frame,
                mode,
                selected_values,
                repeat=repeat,
            )
            output = format_atomic_id_list(atom_ids)
            self.atomic_index_output.delete("1.0", "end")
            self.atomic_index_output.insert("1.0", output)
            unique_count = len(atom_ids) // repeat
            self.atomic_index_status.set(
                f"Matched {unique_count} atom(s) at timestep {frame.timestep}; "
                f"generated {len(atom_ids)} list entries."
            )
        except Exception as exc:  # pragma: no cover - GUI feedback.
            messagebox.showerror("Atomic index generation failed", str(exc))

    def _copy_atomic_index_list(self) -> None:
        """Copy the generated bracketed atom-ID list to the clipboard."""

        output = self.atomic_index_output.get("1.0", "end-1c").strip()
        if not output:
            messagebox.showerror("Copy failed", "Generate an atom-ID list before copying.")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(output)
        self.root.update_idletasks()
        self.atomic_index_status.set("Copied the atom-ID list to the clipboard.")
