"""Thermodynamic-data tab for the Tkinter GUI."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from lammpalyze.gui.helpers import THERMO_DEFAULTS, parse_reference_lines, parse_simulation_groups
from lammpalyze.plotting import plot_thermo


class ThermoTabMixin:
    """Build and manage the thermodynamic-data tab."""

    def _build_thermo_tab(self, parent: ttk.Frame) -> None:
        """Create controls and scrollable output area for thermo plots."""

        controls_container = ttk.Frame(parent)
        controls_container.pack(side="left", fill="y", padx=8, pady=8)
        self._thermo_controls_canvas = tk.Canvas(controls_container, highlightthickness=0, width=260)
        thermo_controls_scrollbar = ttk.Scrollbar(
            controls_container,
            orient="vertical",
            command=self._thermo_controls_canvas.yview,
        )
        controls = ttk.Frame(self._thermo_controls_canvas)
        self._thermo_controls_window = self._thermo_controls_canvas.create_window(
            (0, 0),
            window=controls,
            anchor="nw",
        )
        self._thermo_controls_canvas.configure(yscrollcommand=thermo_controls_scrollbar.set)
        controls.bind(
            "<Configure>",
            lambda _event: self._thermo_controls_canvas.configure(
                scrollregion=self._thermo_controls_canvas.bbox("all")
            ),
        )
        self._thermo_controls_canvas.bind(
            "<Configure>",
            lambda event: self._thermo_controls_canvas.itemconfigure(
                self._thermo_controls_window,
                width=event.width,
            ),
        )
        self._thermo_controls_canvas.bind("<Enter>", self._bind_thermo_controls_mousewheel)
        self._thermo_controls_canvas.bind("<Leave>", self._unbind_thermo_controls_mousewheel)
        controls.bind("<Enter>", self._bind_thermo_controls_mousewheel)
        controls.bind("<Leave>", self._unbind_thermo_controls_mousewheel)
        self._thermo_controls_canvas.pack(side="left", fill="y", expand=True)
        thermo_controls_scrollbar.pack(side="right", fill="y")

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

        ttk.Label(controls, text="Average groups (1,3; 2,4)").pack(anchor="w")
        self.thermo_average_groups = tk.StringVar()
        ttk.Entry(controls, textvariable=self.thermo_average_groups).pack(fill="x", pady=(0, 12))
        ttk.Label(controls, text="Average labels").pack(anchor="w")
        self.thermo_average_labels = tk.StringVar()
        ttk.Entry(controls, textvariable=self.thermo_average_labels).pack(fill="x", pady=(0, 12))

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

        self.thermo_theme = tk.StringVar(value="Dark")
        ttk.Label(controls, text="Background").pack(anchor="w")
        ttk.Combobox(
            controls,
            textvariable=self.thermo_theme,
            values=["Dark", "Bright"],
            state="readonly",
        ).pack(fill="x", pady=(0, 12))

        self.thermo_gradient_enabled = tk.BooleanVar(value=False)
        self.thermo_gradient_start = tk.StringVar(value="#f9c74f")
        self.thermo_gradient_end = tk.StringVar(value="#7209b7")
        ttk.Checkbutton(
            controls,
            text="Use gradient colors",
            variable=self.thermo_gradient_enabled,
        ).pack(anchor="w", pady=(0, 4))
        ttk.Label(controls, text="Gradient start").pack(anchor="w")
        ttk.Entry(controls, textvariable=self.thermo_gradient_start).pack(fill="x", pady=(0, 4))
        ttk.Label(controls, text="Gradient end").pack(anchor="w")
        ttk.Entry(controls, textvariable=self.thermo_gradient_end).pack(fill="x", pady=(0, 12))

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

        self.thermo_running_average_enabled = tk.BooleanVar(value=False)
        self.thermo_running_average_points = tk.StringVar(value="10")
        ttk.Checkbutton(
            controls,
            text="Running average in first plot",
            variable=self.thermo_running_average_enabled,
        ).pack(anchor="w", pady=(0, 4))
        ttk.Label(controls, text="Average points").pack(anchor="w")
        ttk.Spinbox(
            controls,
            from_=1,
            to=1000000,
            textvariable=self.thermo_running_average_points,
            width=10,
        ).pack(fill="x", pady=(0, 12))

        ttk.Label(controls, text="Vertical lines").pack(anchor="w")
        self.thermo_vertical_lines = tk.StringVar()
        ttk.Entry(controls, textvariable=self.thermo_vertical_lines).pack(fill="x", pady=(0, 8))
        ttk.Label(controls, text="Horizontal lines").pack(anchor="w")
        self.thermo_horizontal_lines = tk.StringVar()
        ttk.Entry(controls, textvariable=self.thermo_horizontal_lines).pack(fill="x", pady=(0, 12))

        self._update_thermo_step_controls(preserve=False)
        self._update_thermo_y_controls(preserve=False)
        ttk.Button(controls, text="Plot", command=self._plot_thermo).pack(fill="x")
        ttk.Button(controls, text="Export PNGs", command=self._save_thermo_plots).pack(fill="x", pady=(8, 0))

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
                running_average_points=self._thermo_running_average_points(),
                reference_lines=self._thermo_reference_lines(),
                average_groups=self._thermo_average_groups(simulations),
                average_group_labels=self._thermo_average_group_labels(),
                theme=self.thermo_theme.get(),
                gradient_colors=self._thermo_gradient_colors(),
            )
            for figure in figures:
                canvas = self._create_figure_canvas(figure, self._thermo_plot_area)
                canvas.get_tk_widget().pack(fill="x", expand=False, pady=(0, 12))
                self._thermo_canvases.append(canvas)
            self._thermo_scroll_canvas.yview_moveto(0)
        except Exception as exc:  # pragma: no cover - GUI feedback.
            messagebox.showerror("Thermo plotting failed", str(exc))

    def _save_thermo_plots(self) -> None:
        """Save the current thermodynamic plots to image files."""

        parameter = self.thermo_parameter.get() or "thermo"
        initialfile = f"thermodynamic_data_{parameter}.png"
        self._export_canvas_figures_png(
            self._thermo_canvases,
            "Export thermodynamic plots",
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

    def _thermo_running_average_points(self) -> int | None:
        """Return the running-average window size for the first thermo plot."""

        if not self.thermo_running_average_enabled.get():
            return None
        points = int(self.thermo_running_average_points.get())
        if points < 1:
            raise ValueError("Running-average points must be at least 1.")
        return points

    def _thermo_reference_lines(self) -> tuple[list[float], list[float]]:
        """Return vertical and horizontal reference lines for thermo plots."""

        return (
            parse_reference_lines(self.thermo_vertical_lines.get()),
            parse_reference_lines(self.thermo_horizontal_lines.get()),
        )

    def _thermo_average_groups(self, selected_simulations) -> list[list[int]] | None:
        """Return simulation-index groups for the thermo average plot."""

        groups = parse_simulation_groups(self.thermo_average_groups.get())
        if not groups:
            return None
        selected_indices = {simulation.index for simulation in selected_simulations}
        unknown_indices = sorted({index for group in groups for index in group} - selected_indices)
        if unknown_indices:
            missing = ", ".join(str(index) for index in unknown_indices)
            raise ValueError(f"Average groups include unselected simulations: {missing}.")
        return groups

    def _thermo_average_group_labels(self) -> list[str] | None:
        """Return optional labels for thermodynamic average groups."""

        labels = [label.strip() for label in self.thermo_average_labels.get().split(";")]
        if not any(labels):
            return None
        return labels

    def _thermo_gradient_colors(self) -> tuple[str, str] | None:
        """Return the optional thermo line-color gradient."""

        if not self.thermo_gradient_enabled.get():
            return None
        start = self.thermo_gradient_start.get().strip()
        end = self.thermo_gradient_end.get().strip()
        if not start or not end:
            raise ValueError("Enter both gradient start and end colors.")
        return start, end

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

    def _bind_thermo_controls_mousewheel(self, _event) -> None:
        """Bind global mouse-wheel scrolling while the pointer is over thermo controls."""

        self.root.bind_all("<MouseWheel>", self._on_thermo_controls_mousewheel)
        self.root.bind_all("<Button-4>", self._on_thermo_controls_mousewheel)
        self.root.bind_all("<Button-5>", self._on_thermo_controls_mousewheel)

    def _unbind_thermo_controls_mousewheel(self, _event) -> None:
        """Remove global mouse-wheel bindings for thermo controls scrolling."""

        self.root.unbind_all("<MouseWheel>")
        self.root.unbind_all("<Button-4>")
        self.root.unbind_all("<Button-5>")

    def _on_thermo_controls_mousewheel(self, event) -> None:
        """Scroll the thermo controls canvas from mouse-wheel events."""

        if getattr(event, "num", None) == 4:
            delta = -1
        elif getattr(event, "num", None) == 5:
            delta = 1
        else:
            delta = -1 * int(event.delta / 120)
        self._thermo_controls_canvas.yview_scroll(delta, "units")
