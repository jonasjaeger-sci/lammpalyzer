"""Pairwise local-dump and computed MSD tabs for the Tkinter GUI."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from lammpalyze.gui.helpers import LEGEND_PLACEMENTS, parse_reference_lines, parse_simulation_groups
from lammpalyze.parsers import msd_data_columns, pairwise_data_columns
from lammpalyze.plotting import plot_msd, plot_pairwise


class ComputedDataTabMixin:
    """Build and manage the pairwise-data and mean-square-displacement tabs."""

    def _build_pairwise_tab(self, parent: ttk.Frame) -> None:
        """Create selectors for local-dump data columns and particle pairs."""

        controls, self._pairwise_plot_area = self._computed_tab_layout(parent)
        self._pairwise_simulations = [
            simulation for simulation in self.project.simulations if simulation.pairwise_df is not None
        ]
        parameters = sorted(
            {
                column
                for simulation in self._pairwise_simulations
                for column in pairwise_data_columns(simulation.pairwise_df)
            }
        )
        self.pairwise_parameter = tk.StringVar(value=parameters[0] if parameters else "")
        ttk.Label(controls, text="Data column").pack(anchor="w")
        ttk.Combobox(
            controls,
            textvariable=self.pairwise_parameter,
            values=parameters,
            state="readonly",
        ).pack(fill="x", pady=(0, 12))

        ttk.Label(controls, text="Particle pairs").pack(anchor="w")
        pair_container = ttk.Frame(controls)
        pair_container.pack(fill="both", pady=(0, 12))
        self.pairwise_series_list = tk.Listbox(
            pair_container,
            selectmode="multiple",
            exportselection=False,
            height=14,
            width=32,
        )
        pair_scrollbar = ttk.Scrollbar(
            pair_container,
            orient="vertical",
            command=self.pairwise_series_list.yview,
        )
        self.pairwise_series_list.configure(yscrollcommand=pair_scrollbar.set)
        self._pairwise_series: list[tuple[int, str]] = []
        for simulation in self._pairwise_simulations:
            pairs = sorted(
                simulation.pairwise_df["Pair"].dropna().astype(str).unique(),
                key=_pair_sort_key,
            )
            for pair in pairs:
                self._pairwise_series.append((simulation.index, pair))
                self.pairwise_series_list.insert("end", f"Dump{simulation.index} - {pair}")
        if self.pairwise_series_list.size():
            self.pairwise_series_list.select_set(0, "end")
        self.pairwise_series_list.pack(side="left", fill="both", expand=True)
        pair_scrollbar.pack(side="right", fill="y")
        self._build_list_selection_buttons(controls, self.pairwise_series_list)

        self._pairwise_atoms: dict[str, tuple[int, int]] = {}
        for simulation in self._pairwise_simulations:
            if not simulation.smiles_id:
                continue
            atom_ids = sorted(
                set(simulation.pairwise_df["Particle 1"].dropna().astype(int))
                | set(simulation.pairwise_df["Particle 2"].dropna().astype(int))
            )
            for atom_id in atom_ids:
                label = f"Simulation {simulation.index} - atom {atom_id}"
                self._pairwise_atoms[label] = (simulation.index, atom_id)
        atom_values = ["None", *self._pairwise_atoms]
        self.pairwise_atom = tk.StringVar(value="None")
        ttk.Label(controls, text="Molecule track atom").pack(anchor="w")
        ttk.Combobox(
            controls,
            textvariable=self.pairwise_atom,
            values=atom_values,
            state="readonly",
        ).pack(fill="x", pady=(0, 8))
        self.pairwise_molecule_notation = tk.StringVar(value="Chemical formula")
        ttk.Label(controls, text="Molecule notation").pack(anchor="w")
        ttk.Combobox(
            controls,
            textvariable=self.pairwise_molecule_notation,
            values=["Chemical formula", "SMILES"],
            state="readonly",
        ).pack(fill="x", pady=(0, 12))

        self._build_computed_plot_options(controls, "pairwise")
        ttk.Button(controls, text="Plot", command=self._plot_pairwise).pack(fill="x")
        ttk.Button(controls, text="Export PNG", command=self._save_pairwise_plot).pack(
            fill="x", pady=(8, 0)
        )

    def _build_msd_tab(self, parent: ttk.Frame) -> None:
        """Create an independently selectable list of every MSD file column."""

        controls, self._msd_plot_area = self._computed_tab_layout(parent, scroll_plot=True)
        self._msd_simulations = [
            simulation for simulation in self.project.simulations if simulation.msd_df is not None
        ]
        ttk.Label(controls, text="MSD data series").pack(anchor="w")
        series_container = ttk.Frame(controls)
        series_container.pack(fill="both", pady=(0, 12))
        self.msd_series_list = tk.Listbox(
            series_container,
            selectmode="multiple",
            exportselection=False,
            height=16,
            width=34,
        )
        series_scrollbar = ttk.Scrollbar(
            series_container,
            orient="vertical",
            command=self.msd_series_list.yview,
        )
        self.msd_series_list.configure(yscrollcommand=series_scrollbar.set)
        self._msd_series: list[tuple[int, str]] = []
        for simulation in self._msd_simulations:
            for column in msd_data_columns(simulation.msd_df):
                self._msd_series.append((simulation.index, column))
                self.msd_series_list.insert("end", f"MSD{simulation.index} - {column}")
        if self.msd_series_list.size():
            self.msd_series_list.select_set(0, "end")
        self.msd_series_list.pack(side="left", fill="both", expand=True)
        series_scrollbar.pack(side="right", fill="y")
        self._build_list_selection_buttons(controls, self.msd_series_list)

        self.msd_average_groups = tk.StringVar()
        ttk.Label(controls, text="Average groups (e.g. 1,2; 3,4)").pack(anchor="w")
        ttk.Entry(controls, textvariable=self.msd_average_groups).pack(fill="x", pady=(0, 8))
        self.msd_average_labels = tk.StringVar()
        ttk.Label(controls, text="Average group labels (semicolon-separated)").pack(anchor="w")
        ttk.Entry(controls, textvariable=self.msd_average_labels).pack(fill="x", pady=(0, 12))

        self._build_computed_plot_options(controls, "msd")
        ttk.Button(controls, text="Plot", command=self._plot_msd).pack(fill="x")
        ttk.Button(controls, text="Export PNG", command=self._save_msd_plot).pack(
            fill="x", pady=(8, 0)
        )

    def _computed_tab_layout(
        self,
        parent: ttk.Frame,
        *,
        scroll_plot: bool = False,
    ) -> tuple[ttk.Frame, ttk.Frame]:
        """Build the thermo-like left-control/right-plot tab layout."""

        controls_container = ttk.Frame(parent)
        controls_container.pack(side="left", fill="y", padx=8, pady=8)
        controls_canvas = tk.Canvas(controls_container, highlightthickness=0, width=290)
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
        if not scroll_plot:
            return controls, plot_container

        self._msd_scroll_canvas = tk.Canvas(
            plot_container,
            highlightthickness=0,
            background="#0b1020",
        )
        plot_scrollbar = ttk.Scrollbar(
            plot_container,
            orient="vertical",
            command=self._msd_scroll_canvas.yview,
        )
        plot_area = ttk.Frame(self._msd_scroll_canvas)
        plot_window = self._msd_scroll_canvas.create_window(
            (0, 0),
            window=plot_area,
            anchor="nw",
        )
        self._msd_scroll_canvas.configure(yscrollcommand=plot_scrollbar.set)
        plot_area.bind(
            "<Configure>",
            lambda _event: self._msd_scroll_canvas.configure(
                scrollregion=self._msd_scroll_canvas.bbox("all")
            ),
        )
        self._msd_scroll_canvas.bind(
            "<Configure>",
            lambda event: self._msd_scroll_canvas.itemconfigure(plot_window, width=event.width),
        )
        self._msd_scroll_canvas.pack(side="left", fill="both", expand=True)
        plot_scrollbar.pack(side="right", fill="y")
        return controls, plot_area

    def _build_computed_plot_options(self, controls: ttk.Frame, prefix: str) -> None:
        """Add shared range, smoothing, reference-line, and theme controls."""

        step_start = tk.StringVar()
        step_end = tk.StringVar()
        y_minimum = tk.StringVar()
        y_maximum = tk.StringVar()
        running_enabled = tk.BooleanVar(value=False)
        running_points = tk.StringVar(value="10")
        vertical_lines = tk.StringVar()
        horizontal_lines = tk.StringVar()
        theme = tk.StringVar(value="Dark")
        legend_location = tk.StringVar(value="Best")
        for suffix, value in (
            ("step_start", step_start),
            ("step_end", step_end),
            ("y_minimum", y_minimum),
            ("y_maximum", y_maximum),
            ("running_enabled", running_enabled),
            ("running_points", running_points),
            ("vertical_lines", vertical_lines),
            ("horizontal_lines", horizontal_lines),
            ("theme", theme),
            ("legend_location", legend_location),
        ):
            setattr(self, f"{prefix}_{suffix}", value)

        ttk.Label(controls, text="Timestep range (optional)").pack(anchor="w")
        ttk.Entry(controls, textvariable=step_start).pack(fill="x", pady=(0, 4))
        ttk.Entry(controls, textvariable=step_end).pack(fill="x", pady=(0, 12))
        ttk.Label(controls, text="Y-axis range (optional)").pack(anchor="w")
        ttk.Entry(controls, textvariable=y_minimum).pack(fill="x", pady=(0, 4))
        ttk.Entry(controls, textvariable=y_maximum).pack(fill="x", pady=(0, 12))
        ttk.Checkbutton(
            controls,
            text="Show running average",
            variable=running_enabled,
        ).pack(anchor="w")
        ttk.Label(controls, text="Average points").pack(anchor="w")
        ttk.Spinbox(
            controls,
            from_=1,
            to=1000000,
            textvariable=running_points,
        ).pack(fill="x", pady=(0, 12))
        ttk.Label(controls, text="Vertical reference lines").pack(anchor="w")
        ttk.Entry(controls, textvariable=vertical_lines).pack(fill="x", pady=(0, 8))
        ttk.Label(controls, text="Horizontal reference lines").pack(anchor="w")
        ttk.Entry(controls, textvariable=horizontal_lines).pack(fill="x", pady=(0, 12))
        ttk.Label(controls, text="Background").pack(anchor="w")
        ttk.Combobox(
            controls,
            textvariable=theme,
            values=["Dark", "Bright"],
            state="readonly",
        ).pack(fill="x", pady=(0, 12))
        ttk.Label(controls, text="Legend placement").pack(anchor="w")
        ttk.Combobox(
            controls,
            textvariable=legend_location,
            values=LEGEND_PLACEMENTS,
            state="readonly",
        ).pack(fill="x", pady=(0, 12))

    @staticmethod
    def _build_list_selection_buttons(controls: ttk.Frame, listbox: tk.Listbox) -> None:
        """Add compact buttons for selecting or clearing every list entry."""

        buttons = ttk.Frame(controls)
        buttons.pack(fill="x", pady=(0, 12))
        ttk.Button(
            buttons,
            text="Select all",
            command=lambda: listbox.select_set(0, "end"),
        ).pack(side="left", fill="x", expand=True)
        ttk.Button(
            buttons,
            text="Deselect all",
            command=lambda: listbox.selection_clear(0, "end"),
        ).pack(side="left", fill="x", expand=True, padx=(6, 0))

    def _plot_pairwise(self) -> None:
        """Render the chosen pairwise parameter for selected particle pairs."""

        try:
            parameter = self.pairwise_parameter.get()
            if not parameter:
                raise ValueError("Select a pairwise data column.")
            selections = [
                self._pairwise_series[index]
                for index in self.pairwise_series_list.curselection()
            ]
            if not selections:
                raise ValueError("Select at least one particle pair.")
            figure = plot_pairwise(
                self._pairwise_simulations,
                parameter,
                selections,
                molecule_atom=self._selected_pairwise_atom(),
                molecule_notation=(
                    "formula"
                    if self.pairwise_molecule_notation.get() == "Chemical formula"
                    else "smiles"
                ),
                **self._computed_plot_options("pairwise"),
            )
            self._replace_canvas("_pairwise_canvas", self._pairwise_plot_area, figure)
        except Exception as exc:  # pragma: no cover - GUI feedback.
            messagebox.showerror("Pairwise plotting failed", str(exc))

    def _plot_msd(self) -> None:
        """Render every independently selected MSD file column."""

        try:
            selections = [self._msd_series[index] for index in self.msd_series_list.curselection()]
            if not selections:
                raise ValueError("Select at least one MSD data series.")
            figures = plot_msd(
                self._msd_simulations,
                selections,
                average_groups=self._msd_average_groups(selections),
                average_group_labels=self._msd_average_group_labels(),
                **self._computed_plot_options("msd"),
            )
            for canvas in self._msd_canvases:
                self._destroy_canvas(canvas)
            self._msd_canvases = []
            for figure in figures:
                canvas = self._create_figure_canvas(figure, self._msd_plot_area)
                canvas.get_tk_widget().pack(fill="x", expand=False, pady=(0, 12))
                self._msd_canvases.append(canvas)
            self._msd_scroll_canvas.yview_moveto(0)
        except Exception as exc:  # pragma: no cover - GUI feedback.
            messagebox.showerror("MSD plotting failed", str(exc))

    def _selected_pairwise_atom(self) -> tuple[int, int] | None:
        """Return the optional simulation/atom selection for molecule tracking."""

        return self._pairwise_atoms.get(self.pairwise_atom.get())

    def _msd_average_groups(self, selections: list[tuple[int, str]]) -> list[list[int]] | None:
        """Return validated simulation-index groups for selected MSD series."""

        groups = parse_simulation_groups(self.msd_average_groups.get())
        if not groups:
            return None
        selected_indices = {simulation_index for simulation_index, _ in selections}
        unknown_indices = sorted({index for group in groups for index in group} - selected_indices)
        if unknown_indices:
            missing = ", ".join(str(index) for index in unknown_indices)
            raise ValueError(f"Average groups include simulations without selected MSD series: {missing}.")
        return groups

    def _msd_average_group_labels(self) -> list[str] | None:
        """Return optional labels for the MSD average groups."""

        labels = [label.strip() for label in self.msd_average_labels.get().split(";")]
        return labels if any(labels) else None

    def _computed_plot_options(self, prefix: str) -> dict:
        """Validate and collect common computed-data plot options."""

        step_range = self._optional_range(prefix, "step_start", "step_end", "timestep")
        y_range = self._optional_range(prefix, "y_minimum", "y_maximum", "y-axis")
        running_points = None
        if getattr(self, f"{prefix}_running_enabled").get():
            running_points = int(getattr(self, f"{prefix}_running_points").get())
            if running_points < 1:
                raise ValueError("Running-average points must be at least 1.")
        return {
            "step_range": step_range,
            "y_range": y_range,
            "running_average_points": running_points,
            "reference_lines": (
                parse_reference_lines(getattr(self, f"{prefix}_vertical_lines").get()),
                parse_reference_lines(getattr(self, f"{prefix}_horizontal_lines").get()),
            ),
            "legend_location": getattr(self, f"{prefix}_legend_location").get(),
            "theme": getattr(self, f"{prefix}_theme").get(),
        }

    def _optional_range(
        self,
        prefix: str,
        minimum_name: str,
        maximum_name: str,
        description: str,
    ) -> tuple[float, float] | None:
        """Read two optional range entries, requiring either both or neither."""

        minimum = getattr(self, f"{prefix}_{minimum_name}").get().strip()
        maximum = getattr(self, f"{prefix}_{maximum_name}").get().strip()
        if not minimum and not maximum:
            return None
        if not minimum or not maximum:
            raise ValueError(f"Enter both {description} bounds, or leave both empty.")
        return tuple(sorted((float(minimum), float(maximum))))

    def _save_pairwise_plot(self) -> None:
        """Export the displayed pairwise plot as PNG."""

        parameter = self.pairwise_parameter.get() or "data"
        self._export_canvas_figure_png(
            self._pairwise_canvas,
            "Export pairwise data",
            f"pairwise_{parameter}.png",
        )

    def _save_msd_plot(self) -> None:
        """Export the displayed MSD plot as PNG."""

        self._export_canvas_figures_png(
            self._msd_canvases,
            "Export mean-square-displacement plots",
            "mean_square_displacement.png",
            ["selected_series", "average"],
        )


def _pair_sort_key(pair: str) -> tuple[int, int, int] | tuple[int, str, str]:
    """Sort numeric ``id-id`` descriptors naturally, with a safe fallback."""

    try:
        first, second = pair.split("-", maxsplit=1)
        return 0, int(first), int(second)
    except (TypeError, ValueError):
        return 1, str(pair), ""
