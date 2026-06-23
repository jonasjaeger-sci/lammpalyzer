"""Atomic partial-charge evolution tab for the Tkinter GUI."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from lammpalyze.plotting import plot_charge_evolution


class ChargeTabMixin:
    """Build and manage plots of ReaxFF atomic partial charges."""

    def _build_charge_tab(self, parent: ttk.Frame) -> None:
        """Create charge selection controls and a Matplotlib output area."""

        controls = ttk.Frame(parent)
        controls.pack(side="left", fill="y", padx=8, pady=8)
        self._charge_plot_area = ttk.Frame(parent)
        self._charge_plot_area.pack(side="right", fill="both", expand=True, padx=8, pady=8)

        self._charge_simulations = [
            simulation for simulation in self.project.simulations if simulation.charge_statistics
        ]
        ttk.Label(controls, text="Simulations").pack(anchor="w")
        self.charge_sim_list = tk.Listbox(
            controls,
            selectmode="multiple",
            exportselection=False,
            height=6,
        )
        for simulation in self._charge_simulations:
            self.charge_sim_list.insert("end", f"Simulation {simulation.index}")
        if self.charge_sim_list.size():
            self.charge_sim_list.select_set(0, "end")
        self.charge_sim_list.pack(fill="x", pady=(0, 12))

        available_elements = sorted(
            {
                element
                for simulation in self._charge_simulations
                for summaries in (simulation.charge_statistics or {}).values()
                for element in summaries
            }
        )
        ttk.Label(controls, text="Elements").pack(anchor="w")
        self.charge_element_list = tk.Listbox(
            controls,
            selectmode="multiple",
            exportselection=False,
            height=min(8, max(3, len(available_elements))),
        )
        for element in available_elements:
            self.charge_element_list.insert("end", element)
        if self.charge_element_list.size():
            self.charge_element_list.select_set(0, "end")
        self.charge_element_list.pack(fill="x", pady=(0, 12))

        self.charge_uncertainty = tk.StringVar(value="Standard-deviation band")
        ttk.Label(controls, text="Uncertainty display").pack(anchor="w")
        ttk.Combobox(
            controls,
            textvariable=self.charge_uncertainty,
            values=["Standard-deviation band", "Error bars", "None"],
            state="readonly",
        ).pack(fill="x", pady=(0, 12))

        self.charge_step_start = tk.StringVar()
        self.charge_step_end = tk.StringVar()
        ttk.Label(controls, text="Timestep start (optional)").pack(anchor="w")
        ttk.Entry(controls, textvariable=self.charge_step_start).pack(fill="x", pady=(0, 8))
        ttk.Label(controls, text="Timestep end (optional)").pack(anchor="w")
        ttk.Entry(controls, textvariable=self.charge_step_end).pack(fill="x", pady=(0, 12))

        self.charge_theme = tk.StringVar(value="Dark")
        ttk.Label(controls, text="Background").pack(anchor="w")
        ttk.Combobox(
            controls,
            textvariable=self.charge_theme,
            values=["Dark", "Bright"],
            state="readonly",
        ).pack(fill="x", pady=(0, 12))

        ttk.Button(controls, text="Plot", command=self._plot_charges).pack(fill="x")
        ttk.Button(controls, text="Export PNG", command=self._save_charge_plot).pack(
            fill="x",
            pady=(8, 0),
        )
        ttk.Label(
            controls,
            text="Mean and population standard deviation are calculated across all atoms of "
            "an element within each bond-file frame.",
            wraplength=230,
            justify="left",
        ).pack(anchor="w", pady=(16, 0))

    def _plot_charges(self) -> None:
        """Plot selected per-element partial-charge statistics."""

        try:
            simulations = [
                self._charge_simulations[index]
                for index in self.charge_sim_list.curselection()
            ]
            if not simulations:
                raise ValueError("Select at least one simulation.")
            elements = [self.charge_element_list.get(index) for index in self.charge_element_list.curselection()]
            uncertainty = {
                "Standard-deviation band": "band",
                "Error bars": "errorbar",
                "None": "none",
            }[self.charge_uncertainty.get()]
            figure = plot_charge_evolution(
                simulations,
                elements,
                uncertainty=uncertainty,
                step_range=self._charge_step_range(),
                theme=self.charge_theme.get(),
            )
            self._replace_canvas("_charge_canvas", self._charge_plot_area, figure)
        except Exception as exc:  # pragma: no cover - GUI feedback.
            messagebox.showerror("Charge plotting failed", str(exc))

    def _charge_step_range(self) -> tuple[float, float] | None:
        """Return the optional charge-plot timestep range."""

        start = self.charge_step_start.get().strip()
        end = self.charge_step_end.get().strip()
        if not start and not end:
            return None
        if not start or not end:
            raise ValueError("Enter both charge timestep bounds, or leave both empty.")
        return float(start), float(end)

    def _save_charge_plot(self) -> None:
        """Export the displayed charge plot as a PNG image."""

        self._export_canvas_figure_png(
            self._charge_canvas,
            "Export charge evolution",
            "atomic_charges.png",
        )
