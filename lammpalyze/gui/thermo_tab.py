"""Thermodynamic-data tab for the Tkinter GUI."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from lammpalyze.gui.helpers import THERMO_DEFAULTS
from lammpalyze.plotting import plot_thermo


class ThermoTabMixin:
    """Build and manage the thermodynamic-data tab."""

    def _build_thermo_tab(self, parent: ttk.Frame) -> None:
        """Create controls and scrollable output area for thermo plots."""

        controls = ttk.Frame(parent)
        controls.pack(side="left", fill="y", padx=8, pady=8)
        plot_area = ttk.Frame(parent)
        plot_area.pack(side="right", fill="both", expand=True, padx=8, pady=8)
        self._thermo_scroll_canvas = tk.Canvas(plot_area, highlightthickness=0, background="#0b1020")
        thermo_scrollbar = ttk.Scrollbar(
            plot_area,
            orient="vertical",
            command=self._thermo_scroll_canvas.yview,
        )
        self._thermo_plot_area = ttk.Frame(self._thermo_scroll_canvas)
        self._thermo_window = self._thermo_scroll_canvas.create_window(
            (0, 0),
            window=self._thermo_plot_area,
            anchor="nw",
        )
        self._thermo_scroll_canvas.configure(yscrollcommand=thermo_scrollbar.set)
        self._thermo_plot_area.bind(
            "<Configure>",
            lambda _event: self._thermo_scroll_canvas.configure(
                scrollregion=self._thermo_scroll_canvas.bbox("all")
            ),
        )
        self._thermo_scroll_canvas.bind(
            "<Configure>",
            lambda event: self._thermo_scroll_canvas.itemconfigure(self._thermo_window, width=event.width),
        )
        self._thermo_scroll_canvas.bind("<Enter>", self._bind_thermo_mousewheel)
        self._thermo_scroll_canvas.bind("<Leave>", self._unbind_thermo_mousewheel)
        self._thermo_scroll_canvas.pack(side="left", fill="both", expand=True)
        thermo_scrollbar.pack(side="right", fill="y")

        ttk.Label(controls, text="Simulations").pack(anchor="w")
        self.thermo_sim_list = tk.Listbox(controls, selectmode="multiple", exportselection=False, height=6)
        self._thermo_simulations = [
            simulation
            for simulation in self.project.simulations
            if simulation.thermo_df is not None
        ]
        for simulation in self.project.simulations:
            if simulation.thermo_df is not None:
                self.thermo_sim_list.insert("end", f"Simulation {simulation.index}")
        if self.thermo_sim_list.size():
            self.thermo_sim_list.select_set(0, "end")
        self.thermo_sim_list.bind(
            "<<ListboxSelect>>",
            lambda _event: self._update_thermo_range_controls(preserve=True),
        )
        self.thermo_sim_list.pack(fill="x", pady=(0, 12))

        ttk.Label(controls, text="Legend labels").pack(anchor="w")
        self.thermo_label_vars: dict[int, tk.StringVar] = {}
        labels_frame = ttk.Frame(controls)
        labels_frame.pack(fill="x", pady=(0, 12))
        for simulation in self._thermo_simulations:
            ttk.Label(labels_frame, text=f"Simulation {simulation.index}").pack(anchor="w")
            label_var = tk.StringVar()
            self.thermo_label_vars[simulation.index] = label_var
            ttk.Entry(labels_frame, textvariable=label_var).pack(fill="x", pady=(0, 6))

        available = sorted(
            {
                column
                for simulation in self.project.simulations
                if simulation.thermo_df is not None
                for column in simulation.thermo_df.columns
                if column != "Step"
            }
        )
        values = [value for value in THERMO_DEFAULTS if value in available] or available
        self.thermo_parameter = tk.StringVar(value=values[0] if values else "")
        ttk.Label(controls, text="Parameter").pack(anchor="w")
        self.thermo_parameter_combo = ttk.Combobox(
            controls,
            textvariable=self.thermo_parameter,
            values=values,
            state="readonly",
        )
        self.thermo_parameter_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._update_thermo_range_controls(preserve=False),
        )
        self.thermo_parameter_combo.pack(fill="x", pady=(0, 12))
        ttk.Label(controls, text="Step range").pack(anchor="w")
        self._thermo_step_bounds: tuple[float, float] | None = None
        self._updating_thermo_step_controls = False
        self.thermo_step_min = tk.DoubleVar()
        self.thermo_step_max = tk.DoubleVar()
        self.thermo_step_label = tk.StringVar()
        ttk.Label(controls, textvariable=self.thermo_step_label).pack(anchor="w", pady=(0, 4))
        self.thermo_step_min_slider = ttk.Scale(
            controls,
            orient="horizontal",
            variable=self.thermo_step_min,
            command=lambda _value: self._on_thermo_step_slider("min"),
        )
        self.thermo_step_min_slider.pack(fill="x", pady=(0, 4))
        self.thermo_step_max_slider = ttk.Scale(
            controls,
            orient="horizontal",
            variable=self.thermo_step_max,
            command=lambda _value: self._on_thermo_step_slider("max"),
        )
        self.thermo_step_max_slider.pack(fill="x", pady=(0, 8))
        ttk.Button(
            controls,
            text="Full step range",
            command=lambda: self._update_thermo_step_controls(preserve=False),
        ).pack(fill="x", pady=(0, 12))

        ttk.Label(controls, text="Y-axis range").pack(anchor="w")
        self.thermo_y_min = tk.StringVar()
        self.thermo_y_max = tk.StringVar()
        ttk.Label(controls, text="Minimum").pack(anchor="w")
        ttk.Entry(controls, textvariable=self.thermo_y_min).pack(fill="x", pady=(0, 4))
        ttk.Label(controls, text="Maximum").pack(anchor="w")
        ttk.Entry(controls, textvariable=self.thermo_y_max).pack(fill="x", pady=(0, 8))
        ttk.Button(
            controls,
            text="Full y range",
            command=lambda: self._update_thermo_y_controls(preserve=False),
        ).pack(fill="x", pady=(0, 12))
        self._update_thermo_step_controls(preserve=False)
        self._update_thermo_y_controls(preserve=False)
        ttk.Button(controls, text="Plot", command=self._plot_thermo).pack(fill="x")
        ttk.Button(controls, text="Save plots", command=self._save_thermo_plots).pack(fill="x", pady=(8, 0))

    def _plot_thermo(self) -> None:
        """Plot the selected thermo parameter for selected simulations."""

        try:
            parameter = self.thermo_parameter.get()
            if not parameter:
                raise ValueError("Select a thermodynamic parameter.")
            simulations = self._selected_thermo_simulations()
            if not simulations:
                raise ValueError("Select at least one simulation.")
            for canvas in self._thermo_canvases:
                self._destroy_canvas(canvas)
            self._thermo_canvases = []
            legend_labels = self._thermo_legend_labels()
            figures = plot_thermo(
                simulations,
                parameter,
                legend_labels=legend_labels,
                step_range=self._thermo_step_range(),
                y_range=self._thermo_y_range(),
            )
            for figure in figures:
                canvas = FigureCanvasTkAgg(figure, master=self._thermo_plot_area)
                canvas.draw()
                canvas.get_tk_widget().pack(fill="x", expand=False, pady=(0, 12))
                self._thermo_canvases.append(canvas)
            self._thermo_scroll_canvas.yview_moveto(0)
        except Exception as exc:  # pragma: no cover - GUI feedback.
            messagebox.showerror("Thermo plotting failed", str(exc))

    def _save_thermo_plots(self) -> None:
        """Save the current thermodynamic plots to image files."""

        parameter = self.thermo_parameter.get() or "thermo"
        initialfile = f"thermodynamic_data_{parameter}.png"
        self._save_canvas_figures(
            self._thermo_canvases,
            "Save thermodynamic plots",
            initialfile,
            ["selected_simulations", "average"],
        )

    def _selected_thermo_simulations(self):
        """Return thermo-capable simulations selected in the thermo listbox."""

        return [self._thermo_simulations[index] for index in self.thermo_sim_list.curselection()]

    def _thermo_legend_labels(self) -> dict[int, str]:
        """Return custom thermo legend labels keyed by simulation index."""

        return {index: label_var.get() for index, label_var in self.thermo_label_vars.items()}

    def _thermo_step_range(self) -> tuple[float, float] | None:
        """Return the selected thermo step range, or ``None`` if unavailable."""

        if self._thermo_step_bounds is None:
            return None
        return tuple(sorted((self.thermo_step_min.get(), self.thermo_step_max.get())))

    def _thermo_y_range(self) -> tuple[float, float] | None:
        """Return the selected thermo y-axis range, or ``None`` for auto limits."""

        minimum = self.thermo_y_min.get().strip()
        maximum = self.thermo_y_max.get().strip()
        if not minimum and not maximum:
            return None
        if not minimum or not maximum:
            raise ValueError("Enter both y-axis minimum and maximum, or reset to the full y range.")
        return tuple(sorted((float(minimum), float(maximum))))

    def _update_thermo_range_controls(self, preserve: bool) -> None:
        """Refresh step and y-axis range controls for the thermo tab."""

        self._update_thermo_step_controls(preserve=preserve)
        self._update_thermo_y_controls(preserve=preserve)

    def _update_thermo_step_controls(self, preserve: bool) -> None:
        """Refresh thermo step sliders, optionally preserving their values."""

        bounds = self._thermo_step_bounds_for_simulations(self._selected_thermo_simulations())
        if bounds is None:
            bounds = self._thermo_step_bounds_for_simulations(self._thermo_simulations)
        previous_bounds = self._thermo_step_bounds
        self._thermo_step_bounds = bounds
        if bounds is None:
            self.thermo_step_label.set("No step data")
            self.thermo_step_min_slider.configure(state="disabled")
            self.thermo_step_max_slider.configure(state="disabled")
            return

        lower, upper = bounds
        if preserve and previous_bounds is not None:
            current_lower = self.thermo_step_min.get()
            current_upper = self.thermo_step_max.get()
            lower_value = min(max(current_lower, lower), upper)
            upper_value = min(max(current_upper, lower), upper)
            if lower_value > upper_value or (lower_value == upper_value and lower != upper):
                lower_value, upper_value = lower, upper
        else:
            lower_value, upper_value = lower, upper

        self._updating_thermo_step_controls = True
        self.thermo_step_min_slider.configure(from_=lower, to=upper)
        self.thermo_step_max_slider.configure(from_=lower, to=upper)
        self.thermo_step_min.set(lower_value)
        self.thermo_step_max.set(upper_value)
        state = "normal" if lower != upper else "disabled"
        self.thermo_step_min_slider.configure(state=state)
        self.thermo_step_max_slider.configure(state=state)
        self._updating_thermo_step_controls = False
        self._refresh_thermo_step_label()

    def _thermo_step_bounds_for_simulations(self, simulations) -> tuple[float, float] | None:
        """Return min and max Step values across simulations with thermo data."""

        bounds = []
        for simulation in simulations:
            if simulation.thermo_df is None or "Step" not in simulation.thermo_df.columns:
                continue
            steps = simulation.thermo_df["Step"].dropna()
            if steps.empty:
                continue
            bounds.append((float(steps.min()), float(steps.max())))
        if not bounds:
            return None
        return min(bound[0] for bound in bounds), max(bound[1] for bound in bounds)

    def _on_thermo_step_slider(self, changed: str) -> None:
        """Keep thermo step sliders ordered after one slider changes."""

        if self._updating_thermo_step_controls:
            return
        lower = self.thermo_step_min.get()
        upper = self.thermo_step_max.get()
        if lower > upper:
            if changed == "min":
                self.thermo_step_max.set(lower)
            else:
                self.thermo_step_min.set(upper)
        self._refresh_thermo_step_label()

    def _refresh_thermo_step_label(self) -> None:
        """Update the text label that displays the selected step range."""

        if self._thermo_step_bounds is None:
            self.thermo_step_label.set("No step data")
            return
        lower, upper = self._thermo_step_range() or self._thermo_step_bounds
        self.thermo_step_label.set(
            f"{self._format_step_value(lower)} to {self._format_step_value(upper)}"
        )

    def _update_thermo_y_controls(self, preserve: bool) -> None:
        """Refresh y-axis range entries for the selected thermo parameter."""

        bounds = self._thermo_y_bounds_for_simulations(
            self._selected_thermo_simulations(),
            self.thermo_parameter.get(),
        )
        if bounds is None:
            bounds = self._thermo_y_bounds_for_simulations(self._thermo_simulations, self.thermo_parameter.get())
        if bounds is None:
            if not preserve:
                self.thermo_y_min.set("")
                self.thermo_y_max.set("")
            return

        if preserve and self.thermo_y_min.get().strip() and self.thermo_y_max.get().strip():
            try:
                current_lower = float(self.thermo_y_min.get())
                current_upper = float(self.thermo_y_max.get())
            except ValueError:
                current_lower, current_upper = bounds
            lower, upper = sorted((current_lower, current_upper))
        else:
            lower, upper = bounds

        self.thermo_y_min.set(self._format_step_value(lower))
        self.thermo_y_max.set(self._format_step_value(upper))

    def _thermo_y_bounds_for_simulations(
        self,
        simulations,
        parameter: str,
    ) -> tuple[float, float] | None:
        """Return min and max values for a thermo parameter across simulations."""

        bounds = []
        if not parameter:
            return None
        for simulation in simulations:
            if simulation.thermo_df is None or parameter not in simulation.thermo_df.columns:
                continue
            values = simulation.thermo_df[parameter].dropna()
            if values.empty:
                continue
            bounds.append((float(values.min()), float(values.max())))
        if not bounds:
            return None
        return min(bound[0] for bound in bounds), max(bound[1] for bound in bounds)

    @staticmethod
    def _format_step_value(value: float) -> str:
        """Format whole-number floats without a decimal part."""

        if float(value).is_integer():
            return str(int(value))
        return f"{value:g}"

    def _bind_thermo_mousewheel(self, _event) -> None:
        """Bind global mouse-wheel scrolling while the pointer is over thermo plots."""

        self.root.bind_all("<MouseWheel>", self._on_thermo_mousewheel)
        self.root.bind_all("<Button-4>", self._on_thermo_mousewheel)
        self.root.bind_all("<Button-5>", self._on_thermo_mousewheel)

    def _unbind_thermo_mousewheel(self, _event) -> None:
        """Remove global mouse-wheel bindings for thermo plot scrolling."""

        self.root.unbind_all("<MouseWheel>")
        self.root.unbind_all("<Button-4>")
        self.root.unbind_all("<Button-5>")

    def _on_thermo_mousewheel(self, event) -> None:
        """Scroll the thermo plot canvas from mouse-wheel events."""

        if getattr(event, "num", None) == 4:
            delta = -1
        elif getattr(event, "num", None) == 5:
            delta = 1
        else:
            delta = -1 * int(event.delta / 120)
        self._thermo_scroll_canvas.yview_scroll(delta, "units")
