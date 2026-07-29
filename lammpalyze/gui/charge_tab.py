"""Trajectory-backed atomic data tab for the Tkinter GUI."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib.pyplot as plt

from lammpalyze.atomic import (
    atomic_property_label,
    available_atomic_properties,
    collect_atomic_series,
    collect_element_atomic_series,
    parse_atom_ids,
)
from lammpalyze.plotting import plot_collected_atomic_series


class ChargeTabMixin:
    """Build and manage plots of flexible per-atom trajectory data."""

    def _build_charge_tab(self, parent: ttk.Frame) -> None:
        """Create atomic-data controls and a Matplotlib output area."""

        controls = ttk.Frame(parent)
        controls.pack(side="left", fill="y", padx=8, pady=8)
        plot_container = ttk.Frame(parent)
        plot_container.pack(side="right", fill="both", expand=True, padx=8, pady=8)
        self._charge_scroll_canvas = tk.Canvas(
            plot_container,
            highlightthickness=0,
            background="#0b1020",
        )
        charge_scrollbar = ttk.Scrollbar(
            plot_container,
            orient="vertical",
            command=self._charge_scroll_canvas.yview,
        )
        self._charge_plot_area = ttk.Frame(self._charge_scroll_canvas)
        self._charge_plot_window = self._charge_scroll_canvas.create_window(
            (0, 0),
            window=self._charge_plot_area,
            anchor="nw",
        )
        self._charge_scroll_canvas.configure(yscrollcommand=charge_scrollbar.set)
        self._charge_plot_area.bind(
            "<Configure>",
            lambda _event: self._charge_scroll_canvas.configure(
                scrollregion=self._charge_scroll_canvas.bbox("all")
            ),
        )
        self._charge_scroll_canvas.bind(
            "<Configure>",
            lambda event: self._charge_scroll_canvas.itemconfigure(
                self._charge_plot_window,
                width=event.width,
            ),
        )
        self._charge_scroll_canvas.bind("<Enter>", self._bind_charge_mousewheel)
        self._charge_scroll_canvas.bind("<Leave>", self._unbind_charge_mousewheel)
        self._charge_scroll_canvas.pack(side="left", fill="both", expand=True)
        charge_scrollbar.pack(side="right", fill="y")

        self._charge_simulations = [
            simulation
            for simulation in self.project.simulations
            if simulation.trajectory_path is not None
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

        properties = available_atomic_properties(self._charge_simulations)
        self.atomic_property = tk.StringVar(value=properties[0] if properties else "")
        ttk.Label(controls, text="Property").pack(anchor="w")
        ttk.Combobox(
            controls,
            textvariable=self.atomic_property,
            values=properties,
            state="readonly",
        ).pack(fill="x", pady=(0, 12))

        self.atomic_selection_mode = tk.StringVar(value="Elements")
        ttk.Label(controls, text="Selection").pack(anchor="w")
        selection_combo = ttk.Combobox(
            controls,
            textvariable=self.atomic_selection_mode,
            values=["Elements", "Atom IDs"],
            state="readonly",
        )
        selection_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._update_atomic_selection_controls(),
        )
        selection_combo.pack(fill="x", pady=(0, 12))

        available_elements = sorted(set(self.project.config.element_list))
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

        self.atomic_atom_ids = tk.StringVar()
        ttk.Label(controls, text="Atom IDs").pack(anchor="w")
        self.atomic_atom_ids_entry = ttk.Entry(controls, textvariable=self.atomic_atom_ids)
        self.atomic_atom_ids_entry.pack(fill="x", pady=(0, 12))

        self.charge_uncertainty = tk.StringVar(value="Standard-deviation band")
        ttk.Label(controls, text="Uncertainty display").pack(anchor="w")
        self.atomic_uncertainty_combo = ttk.Combobox(
            controls,
            textvariable=self.charge_uncertainty,
            values=["Standard-deviation band", "Error bars", "None"],
            state="readonly",
        )
        self.atomic_uncertainty_combo.pack(fill="x", pady=(0, 12))

        self.atomic_show_individual_atoms = tk.BooleanVar(value=False)
        self.atomic_individual_check = ttk.Checkbutton(
            controls,
            text="Plot individual atoms",
            variable=self.atomic_show_individual_atoms,
        )
        self.atomic_individual_check.pack(anchor="w", pady=(0, 12))

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

        self.atomic_plot_button = ttk.Button(controls, text="Plot", command=self._plot_charges)
        self.atomic_plot_button.pack(fill="x")
        ttk.Button(controls, text="Export PNG", command=self._save_charge_plot).pack(
            fill="x",
            pady=(8, 0),
        )
        self.atomic_status = tk.StringVar()
        ttk.Label(controls, textvariable=self.atomic_status, wraplength=230).pack(anchor="w", pady=(8, 0))
        self.atomic_progress = ttk.Progressbar(
            controls,
            mode="indeterminate",
            length=180,
        )
        self.atomic_progress.stop()
        self._atomic_plot_queue: queue.Queue = queue.Queue()
        self._atomic_plot_token = None
        self._atomic_plot_thread: threading.Thread | None = None
        self._atomic_plot_after_job: str | None = None
        self._update_atomic_selection_controls()

    def _update_atomic_selection_controls(self) -> None:
        """Enable only controls relevant to the active atom selection mode."""

        element_mode = self.atomic_selection_mode.get() == "Elements"
        self.charge_element_list.configure(state="normal" if element_mode else "disabled")
        self.atomic_atom_ids_entry.configure(state="disabled" if element_mode else "normal")
        self.atomic_uncertainty_combo.configure(state="readonly" if element_mode else "disabled")
        self.atomic_individual_check.configure(state="normal" if element_mode else "disabled")

    def _plot_charges(self) -> None:
        """Plot the selected trajectory atom property."""

        try:
            if self._atomic_plot_thread is not None and self._atomic_plot_thread.is_alive():
                return
            simulations = [
                self._charge_simulations[index]
                for index in self.charge_sim_list.curselection()
            ]
            if not simulations:
                raise ValueError("Select at least one simulation.")
            property_name = self.atomic_property.get()
            if not property_name:
                raise ValueError("Select an atomic property.")
            elements = None
            atom_ids = None
            if self.atomic_selection_mode.get() == "Elements":
                elements = [
                    self.charge_element_list.get(index)
                    for index in self.charge_element_list.curselection()
                ]
                if not elements:
                    raise ValueError("Select at least one element.")
            else:
                atom_ids = parse_atom_ids(self.atomic_atom_ids.get())
            uncertainty = {
                "Standard-deviation band": "band",
                "Error bars": "errorbar",
                "None": "none",
            }[self.charge_uncertainty.get()]
            self._start_atomic_plot_worker(
                simulations=simulations,
                property_name=property_name,
                elements=elements,
                atom_ids=atom_ids,
                include_individual_element_atoms=bool(elements)
                and self.atomic_show_individual_atoms.get(),
                uncertainty=uncertainty,
                step_range=self._charge_step_range(),
                theme=self.charge_theme.get(),
                plot_settings=self._plot_settings(),
            )
        except Exception as exc:  # pragma: no cover - GUI feedback.
            messagebox.showerror("Atomic data plotting failed", str(exc))

    def _start_atomic_plot_worker(
        self,
        *,
        simulations,
        property_name: str,
        elements: list[str] | None,
        atom_ids: list[int] | None,
        include_individual_element_atoms: bool,
        uncertainty: str,
        step_range: tuple[float, float] | None,
        theme: str,
        plot_settings,
    ) -> None:
        """Collect atomic data in a worker so Tk remains responsive."""

        token = object()
        self._atomic_plot_token = token
        self._atomic_plot_queue = queue.Queue()
        self.atomic_plot_button.configure(state="disabled")
        self.atomic_status.set("Reading trajectory data...")
        self.atomic_progress.pack(fill="x", pady=(6, 0))
        self.atomic_progress.start(12)
        options = {
            "simulations": simulations,
            "property_name": property_name,
            "elements": elements,
            "atom_ids": atom_ids,
            "include_individual_element_atoms": include_individual_element_atoms,
            "uncertainty": uncertainty,
            "step_range": step_range,
            "theme": theme,
            "plot_settings": plot_settings,
        }
        thread = threading.Thread(
            target=_collect_atomic_plot_data_worker,
            args=(self._atomic_plot_queue, token, options),
            daemon=True,
        )
        self._atomic_plot_thread = thread
        thread.start()
        self._atomic_plot_after_job = self.root.after(100, self._poll_atomic_plot_queue)

    def _poll_atomic_plot_queue(self) -> None:
        """Handle completed worker results on the Tk thread."""

        if self._closed:
            return
        try:
            status, token, payload = self._atomic_plot_queue.get_nowait()
        except queue.Empty:
            if self._atomic_plot_thread is not None and self._atomic_plot_thread.is_alive():
                self._atomic_plot_after_job = self.root.after(100, self._poll_atomic_plot_queue)
            else:
                self._atomic_plot_after_job = None
            return

        self._atomic_plot_after_job = None
        if token is not self._atomic_plot_token:
            if status == "success":
                self._close_figures_from_payload(payload)
            return

        self._atomic_plot_token = None
        self._atomic_plot_thread = None
        self.atomic_plot_button.configure(state="normal")
        self.atomic_status.set("")
        self.atomic_progress.stop()
        self.atomic_progress.pack_forget()

        if status == "error":
            messagebox.showerror("Atomic data plotting failed", str(payload))
            return

        figures = []
        try:
            figures = self._atomic_figures_from_payload(payload)
            for canvas in self._charge_canvases:
                self._destroy_canvas(canvas)
            self._charge_canvases = []
            for figure in figures:
                canvas = self._create_figure_canvas(figure, self._charge_plot_area)
                canvas.get_tk_widget().pack(fill="x", expand=False, pady=(0, 8))
                self._charge_canvases.append(canvas)
            self._charge_scroll_canvas.yview_moveto(0)
        except Exception as exc:  # pragma: no cover - GUI feedback.
            for figure in figures:
                plt.close(figure)
            messagebox.showerror("Atomic data plotting failed", str(exc))

    def _atomic_figures_from_payload(self, payload) -> list:
        """Create Matplotlib figures from worker-collected atomic series."""

        kind, series, options = payload
        if kind == "series":
            return [
                plot_collected_atomic_series(
                    series,
                    options["property_name"],
                    uncertainty=options["uncertainty"],
                    show_uncertainty=bool(options["elements"]),
                    step_range=options["step_range"],
                    theme=options["theme"],
                    plot_settings=options["plot_settings"],
                )
            ]

        property_label = atomic_property_label(options["property_name"])
        element_label = ", ".join(options["elements"])
        return [
            plot_collected_atomic_series(
                series.aggregate,
                options["property_name"],
                uncertainty=options["uncertainty"],
                show_uncertainty=True,
                step_range=options["step_range"],
                theme=options["theme"],
                plot_settings=options["plot_settings"],
            ),
            plot_collected_atomic_series(
                series.individual,
                options["property_name"],
                uncertainty="none",
                show_uncertainty=False,
                step_range=options["step_range"],
                theme=options["theme"],
                title=f"Individual {element_label} atoms: {property_label}",
                individual_legend=True,
                plot_settings=options["plot_settings"],
            ),
        ]

    @staticmethod
    def _close_figures_from_payload(payload) -> None:
        """Close figures if a stale payload ever contains rendered figures."""

        if not isinstance(payload, list):
            return
        for figure in payload:
            plt.close(figure)

    def _close_atomic_plot_worker(self) -> None:
        """Detach any running atomic worker from Tk-owned GUI state."""

        self._atomic_plot_token = None
        self._atomic_plot_thread = None
        if self._atomic_plot_after_job is not None:
            try:
                self.root.after_cancel(self._atomic_plot_after_job)
            except tk.TclError:
                pass
            self._atomic_plot_after_job = None
        try:
            self.atomic_progress.stop()
            self.atomic_progress.pack_forget()
            self.atomic_status.set("")
            self.atomic_plot_button.configure(state="normal")
        except tk.TclError:
            pass

    def _charge_step_range(self) -> tuple[float, float] | None:
        """Return the optional atomic-data timestep range."""

        start = self.charge_step_start.get().strip()
        end = self.charge_step_end.get().strip()
        if not start and not end:
            return None
        if not start or not end:
            raise ValueError("Enter both timestep bounds, or leave both empty.")
        return float(start), float(end)

    def _save_charge_plot(self) -> None:
        """Export the displayed atomic-data plot as a PNG image."""

        self._export_canvas_figures_png(
            self._charge_canvases,
            "Export atomic data",
            "atomic_data.png",
            ["element_mean", "individual_atoms"],
        )

    def _bind_charge_mousewheel(self, _event) -> None:
        """Bind mouse-wheel scrolling while the pointer is over atomic plots."""

        self.root.bind_all("<MouseWheel>", self._on_charge_mousewheel)
        self.root.bind_all("<Button-4>", self._on_charge_mousewheel)
        self.root.bind_all("<Button-5>", self._on_charge_mousewheel)

    def _unbind_charge_mousewheel(self, _event) -> None:
        """Remove mouse-wheel bindings for atomic plot scrolling."""

        self.root.unbind_all("<MouseWheel>")
        self.root.unbind_all("<Button-4>")
        self.root.unbind_all("<Button-5>")

    def _on_charge_mousewheel(self, event) -> None:
        """Scroll the atomic plot canvas from mouse-wheel events."""

        if getattr(event, "num", None) == 4:
            delta = -1
        elif getattr(event, "num", None) == 5:
            delta = 1
        else:
            delta = -1 * int(event.delta / 120)
        self._charge_scroll_canvas.yview_scroll(delta, "units")


def _collect_atomic_plot_data_worker(result_queue: queue.Queue, token, options: dict) -> None:
    """Collect atomic plot data without touching Tk-owned GUI objects."""

    try:
        if options["elements"] and options["include_individual_element_atoms"]:
            series = collect_element_atomic_series(
                options["simulations"],
                options["property_name"],
                options["elements"],
                step_range=options["step_range"],
                max_individual_series=200,
            )
            payload = ("element_with_individual", series, options)
        else:
            series = collect_atomic_series(
                options["simulations"],
                options["property_name"],
                elements=options["elements"],
                atom_ids=options["atom_ids"],
                step_range=options["step_range"],
            )
            payload = ("series", series, options)
        result_queue.put(("success", token, payload))
    except Exception as exc:  # pragma: no cover - GUI feedback.
        result_queue.put(("error", token, exc))
