"""Structural-relaxation tab for the Tkinter GUI."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from lammpalyze.analysis import LoadedSimulation
from lammpalyze.gui.helpers import LEGEND_PLACEMENTS
from lammpalyze.structure import compute_structural_relaxation
from lammpalyze.structure_plotting import plot_structural_relaxation


class StructuralRelaxationTabMixin:
    """Build and manage S(q) and incoherent-scattering plots."""

    def _build_structure_tab(self, parent: ttk.Frame) -> None:
        """Create controls and output area for structural relaxation."""

        controls_container = ttk.Frame(parent)
        controls_container.pack(side="left", fill="y", padx=8, pady=8)
        controls_canvas = tk.Canvas(controls_container, highlightthickness=0, width=275)
        controls_scrollbar = ttk.Scrollbar(
            controls_container,
            orient="vertical",
            command=controls_canvas.yview,
        )
        controls = ttk.Frame(controls_canvas)
        controls_window = controls_canvas.create_window((0, 0), window=controls, anchor="nw")
        controls_canvas.configure(yscrollcommand=controls_scrollbar.set)
        controls.bind(
            "<Configure>",
            lambda _event: controls_canvas.configure(scrollregion=controls_canvas.bbox("all")),
        )
        controls_canvas.bind(
            "<Configure>",
            lambda event: controls_canvas.itemconfigure(controls_window, width=event.width),
        )
        controls_canvas.pack(side="left", fill="y", expand=True)
        controls_scrollbar.pack(side="right", fill="y")

        plot_container = ttk.Frame(parent)
        plot_container.pack(side="right", fill="both", expand=True, padx=8, pady=8)
        self._structure_scroll_canvas = tk.Canvas(
            plot_container,
            highlightthickness=0,
            background="#0b1020",
        )
        plot_scrollbar = ttk.Scrollbar(
            plot_container,
            orient="vertical",
            command=self._structure_scroll_canvas.yview,
        )
        self._structure_plot_area = ttk.Frame(self._structure_scroll_canvas)
        plot_window = self._structure_scroll_canvas.create_window(
            (0, 0),
            window=self._structure_plot_area,
            anchor="nw",
        )
        self._structure_scroll_canvas.configure(yscrollcommand=plot_scrollbar.set)
        self._structure_plot_area.bind(
            "<Configure>",
            lambda _event: self._structure_scroll_canvas.configure(
                scrollregion=self._structure_scroll_canvas.bbox("all")
            ),
        )
        self._structure_scroll_canvas.bind(
            "<Configure>",
            lambda event: self._structure_scroll_canvas.itemconfigure(plot_window, width=event.width),
        )
        self._structure_scroll_canvas.pack(side="left", fill="both", expand=True)
        plot_scrollbar.pack(side="right", fill="y")

        ttk.Label(controls, text="Simulations").pack(anchor="w")
        self.structure_sim_list = tk.Listbox(
            controls,
            selectmode="multiple",
            exportselection=False,
            height=6,
        )
        self._structure_simulations = [
            simulation
            for simulation in self.project.simulations
            if simulation.trajectory_path is not None
        ]
        for simulation in self._structure_simulations:
            self.structure_sim_list.insert("end", f"Simulation {simulation.index}")
        if self.structure_sim_list.size():
            self.structure_sim_list.select_set(0, "end")
        self.structure_sim_list.bind("<<ListboxSelect>>", lambda _event: self._set_structure_start())
        self.structure_sim_list.pack(fill="x", pady=(0, 12))

        element_values = ["All", *self.project.config.element_list]
        self.structure_element = tk.StringVar(value="All")
        ttk.Label(controls, text="Atoms").pack(anchor="w")
        ttk.Combobox(
            controls,
            textvariable=self.structure_element,
            values=element_values,
            state="readonly",
        ).pack(fill="x", pady=(0, 12))

        self.structure_start_timestep = tk.StringVar()
        self.structure_frame_count = tk.StringVar(value="100")
        self.structure_origin_count = tk.StringVar(value="10")
        self.structure_max_q_index = tk.StringVar(value="8")
        self.structure_block_count = tk.StringVar(value="5")

        ttk.Label(controls, text="Production start timestep").pack(anchor="w")
        ttk.Entry(controls, textvariable=self.structure_start_timestep).pack(fill="x", pady=(0, 8))
        ttk.Label(controls, text="Frames").pack(anchor="w")
        ttk.Entry(controls, textvariable=self.structure_frame_count).pack(fill="x", pady=(0, 8))
        ttk.Label(controls, text="Time origins").pack(anchor="w")
        ttk.Entry(controls, textvariable=self.structure_origin_count).pack(fill="x", pady=(0, 8))
        ttk.Label(controls, text="Max q index").pack(anchor="w")
        ttk.Entry(controls, textvariable=self.structure_max_q_index).pack(fill="x", pady=(0, 8))
        ttk.Label(controls, text="Blocks").pack(anchor="w")
        ttk.Entry(controls, textvariable=self.structure_block_count).pack(fill="x", pady=(0, 12))

        self.structure_theme = tk.StringVar(value="Dark")
        ttk.Label(controls, text="Background").pack(anchor="w")
        ttk.Combobox(
            controls,
            textvariable=self.structure_theme,
            values=["Dark", "Bright"],
            state="readonly",
        ).pack(fill="x", pady=(0, 12))

        self.structure_legend_location = tk.StringVar(value="Best")
        ttk.Label(controls, text="Legend placement").pack(anchor="w")
        ttk.Combobox(
            controls,
            textvariable=self.structure_legend_location,
            values=LEGEND_PLACEMENTS,
            state="readonly",
        ).pack(fill="x", pady=(0, 12))

        ttk.Button(controls, text="Plot", command=self._plot_structure).pack(fill="x")
        ttk.Button(
            controls,
            text="Export PNG",
            command=self._save_structure_plot,
        ).pack(fill="x", pady=(8, 0))

        self.structure_status = ttk.Label(
            self._structure_plot_area,
            text="",
            wraplength=620,
            justify="left",
        )
        self.structure_status.pack(anchor="nw", padx=8, pady=8)
        self._set_structure_start()

    def _plot_structure(self) -> None:
        """Compute and plot S(q) and F_s(q,t) from selected GUI values."""

        try:
            simulations = self._selected_structure_simulations()
            if not simulations:
                raise ValueError("Select at least one simulation.")
            element = self._selected_structure_element()
            results = compute_structural_relaxation(
                simulations,
                start_timestep=int(self.structure_start_timestep.get()),
                frame_count=int(self.structure_frame_count.get()),
                time_origin_count=int(self.structure_origin_count.get()),
                max_q_index=int(self.structure_max_q_index.get()),
                block_count=int(self.structure_block_count.get()),
                element=element,
            )
            figures = plot_structural_relaxation(
                results,
                element_label=element if element is not None else "All atoms",
                legend_location=self.structure_legend_location.get(),
                theme=self.structure_theme.get(),
            )
            for canvas in self._structure_canvases:
                self._destroy_canvas(canvas)
            self._structure_canvases = []
            for figure in figures:
                canvas = self._create_figure_canvas(figure, self._structure_plot_area)
                canvas.get_tk_widget().pack(fill="x", expand=False, pady=(0, 12))
                self._structure_canvases.append(canvas)
            self._structure_scroll_canvas.configure(
                background="#f8fafc" if self.structure_theme.get() == "Bright" else "#0b1020"
            )
            self._structure_scroll_canvas.yview_moveto(0)
            self.structure_status.configure(text=self._structure_status_text(results))
        except Exception as exc:  # pragma: no cover - GUI feedback.
            messagebox.showerror("Structural relaxation failed", str(exc))

    def _save_structure_plot(self) -> None:
        """Save the current structural-relaxation plots to image files."""

        self._export_canvas_figures_png(
            self._structure_canvases,
            "Export structural-relaxation plots",
            "structural_relaxation.png",
            ["static_structure_factor", "incoherent_scattering"],
        )

    def _selected_structure_simulations(self) -> list[LoadedSimulation]:
        """Return trajectory-capable simulations selected in the listbox."""

        return [self._structure_simulations[index] for index in self.structure_sim_list.curselection()]

    def _selected_structure_element(self) -> str | None:
        """Return the selected atom element or ``None`` for all atoms."""

        element = self.structure_element.get().strip()
        return None if element == "All" else element

    def _set_structure_start(self) -> None:
        """Populate the production start entry with the first selected timestep."""

        timesteps = sorted(
            {
                timestep
                for simulation in self._selected_structure_simulations()
                for timestep in self._structure_timesteps(simulation)
            }
        )
        self.structure_start_timestep.set(str(timesteps[0]) if timesteps else "")

    def _structure_timesteps(self, simulation: LoadedSimulation) -> list[int]:
        """Return cached trajectory timesteps for one simulation."""

        if simulation.index not in self._structure_timesteps_by_simulation:
            self._structure_timesteps_by_simulation[simulation.index] = (
                simulation.trajectory_timesteps()
            )
        return self._structure_timesteps_by_simulation[simulation.index]

    @staticmethod
    def _structure_status_text(results) -> str:
        """Return a compact summary of frames, origins, and peak q values."""

        parts = []
        for result in results:
            static = result.static_structure_factor
            incoherent = result.incoherent_scattering
            parts.append(
                f"Simulation {result.simulation_index}: "
                f"{len(static.timesteps)} frame(s), "
                f"{len(incoherent.origin_timesteps)} origin(s), "
                f"peak q={static.peak_q:.6g}"
            )
        return "\n".join(parts)
