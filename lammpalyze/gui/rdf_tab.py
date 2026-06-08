"""Radial-distribution tab for the Tkinter GUI."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from lammpalyze.analysis import LoadedSimulation
from lammpalyze.gui.helpers import parse_reference_lines
from lammpalyze.parsers import list_lammpstrj_timesteps
from lammpalyze.plotting import plot_rdf
from lammpalyze.rdf import compute_rdf


class RdfTabMixin:
    """Build and manage the radial-distribution tab."""

    def _build_rdf_tab(self, parent: ttk.Frame) -> None:
        """Create controls and output area for RDF plotting."""

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

        self.rdf_theme = tk.StringVar(value="Dark")
        ttk.Label(controls, text="Background").pack(anchor="w")
        ttk.Combobox(
            controls,
            textvariable=self.rdf_theme,
            values=["Dark", "Bright"],
            state="readonly",
        ).pack(fill="x", pady=(0, 12))

        self.rdf_gradient_enabled = tk.BooleanVar(value=False)
        self.rdf_gradient_start = tk.StringVar(value="#f9c74f")
        self.rdf_gradient_end = tk.StringVar(value="#7209b7")
        ttk.Checkbutton(
            controls,
            text="Use gradient colors",
            variable=self.rdf_gradient_enabled,
        ).pack(anchor="w", pady=(0, 4))
        ttk.Label(controls, text="Gradient start").pack(anchor="w")
        ttk.Entry(controls, textvariable=self.rdf_gradient_start).pack(fill="x", pady=(0, 4))
        ttk.Label(controls, text="Gradient end").pack(anchor="w")
        ttk.Entry(controls, textvariable=self.rdf_gradient_end).pack(fill="x", pady=(0, 12))

        ttk.Label(controls, text="Vertical lines").pack(anchor="w")
        self.rdf_vertical_lines = tk.StringVar()
        ttk.Entry(controls, textvariable=self.rdf_vertical_lines).pack(fill="x", pady=(0, 8))
        ttk.Label(controls, text="Horizontal lines").pack(anchor="w")
        self.rdf_horizontal_lines = tk.StringVar()
        ttk.Entry(controls, textvariable=self.rdf_horizontal_lines).pack(fill="x", pady=(0, 12))

        ttk.Button(controls, text="Plot", command=self._plot_rdf).pack(fill="x")
        ttk.Button(controls, text="Export PNG", command=self._save_rdf_plot).pack(fill="x", pady=(8, 0))

        self.rdf_status = ttk.Label(plot_area, text="", wraplength=620, justify="left")
        self.rdf_status.pack(anchor="nw", padx=8, pady=8)
        self._set_rdf_last_timesteps()

    def _plot_rdf(self) -> None:
        """Compute and plot RDF curves from the selected GUI values."""

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
            figure = plot_rdf(
                results,
                element_a,
                element_b,
                reference_lines=self._rdf_reference_lines(),
                theme=self.rdf_theme.get(),
                gradient_colors=self._rdf_gradient_colors(),
            )
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

    def _save_rdf_plot(self) -> None:
        """Save the current RDF plot to an image file."""

        self._export_canvas_figure_png(self._rdf_canvas, "Export RDF plot", "radial_distribution.png")

    def _selected_rdf_simulations(self):
        """Return trajectory-capable simulations selected in the RDF listbox."""

        return [self._rdf_simulations[index] for index in self.rdf_sim_list.curselection()]

    def _rdf_reference_lines(self) -> tuple[list[float], list[float]]:
        """Return vertical and horizontal reference lines for the RDF plot."""

        return (
            parse_reference_lines(self.rdf_vertical_lines.get()),
            parse_reference_lines(self.rdf_horizontal_lines.get()),
        )

    def _rdf_gradient_colors(self) -> tuple[str, str] | None:
        """Return the optional RDF line-color gradient."""

        if not self.rdf_gradient_enabled.get():
            return None
        start = self.rdf_gradient_start.get().strip()
        end = self.rdf_gradient_end.get().strip()
        if not start or not end:
            raise ValueError("Enter both RDF gradient start and end colors.")
        return start, end

    def _set_rdf_last_timesteps(self) -> None:
        """Populate RDF timestep entries with the last five selected timesteps."""

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
        """Return cached trajectory timesteps for one simulation."""

        if simulation.index not in self._rdf_timesteps_by_simulation:
            if simulation.trajectory_path is None:
                self._rdf_timesteps_by_simulation[simulation.index] = []
            else:
                self._rdf_timesteps_by_simulation[simulation.index] = list_lammpstrj_timesteps(
                    simulation.trajectory_path
                )
        return self._rdf_timesteps_by_simulation[simulation.index]
