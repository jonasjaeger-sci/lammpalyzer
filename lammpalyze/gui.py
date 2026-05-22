"""Tkinter graphical interface for lammpalyze projects."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from lammpalyze.analysis import LammpalyzeProject
from lammpalyze.plotting import plot_species, plot_thermo
from lammpalyze.smiles import formulas_for_simulation, molecule_photo_image, smiles_for_formula


THERMO_DEFAULTS = ["Temp", "PotEng", "KinEng", "Press", "Volume", "Density"]


class LammpalyzeGUI:
    """Three-tab GUI for species, thermo, and SMILES visualization."""

    def __init__(self, project: LammpalyzeProject) -> None:
        self.project = project
        self.root = tk.Tk()
        self.root.title("lammpalyze")
        self.root.geometry("1100x760")
        self._species_canvas: FigureCanvasTkAgg | None = None
        self._thermo_canvases: list[FigureCanvasTkAgg] = []
        self._molecule_photo = None
        self._closed = False
        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<Control-q>", lambda _event: self.close())
        self.root.bind("<Control-Q>", lambda _event: self.close())

    def run(self) -> None:
        """Start the Tkinter event loop."""

        self.root.mainloop()

    def _build(self) -> None:
        top_bar = ttk.Frame(self.root)
        top_bar.pack(fill="x", padx=8, pady=(8, 0))
        ttk.Button(top_bar, text="Quit", command=self.close).pack(side="right")

        tabs = ttk.Notebook(self.root)
        tabs.pack(fill="both", expand=True, padx=8, pady=8)

        species_tab = ttk.Frame(tabs)
        thermo_tab = ttk.Frame(tabs)
        smiles_tab = ttk.Frame(tabs)
        tabs.add(species_tab, text="Species analysis")
        tabs.add(thermo_tab, text="Thermodynamic data")
        tabs.add(smiles_tab, text="Molecule visualization")

        self._build_species_tab(species_tab)
        self._build_thermo_tab(thermo_tab)
        self._build_smiles_tab(smiles_tab)

    def _build_species_tab(self, parent: ttk.Frame) -> None:
        controls = ttk.Frame(parent)
        controls.pack(side="left", fill="y", padx=8, pady=8)
        plot_area = ttk.Frame(parent)
        plot_area.pack(side="right", fill="both", expand=True, padx=8, pady=8)
        self._species_plot_area = plot_area

        ttk.Label(controls, text="Simulations").pack(anchor="w")
        self.species_sim_list = tk.Listbox(controls, selectmode="multiple", exportselection=False, height=6)
        for simulation in self.project.simulations:
            if simulation.species_df is not None:
                self.species_sim_list.insert("end", f"Simulation {simulation.index}")
        self.species_sim_list.pack(fill="x", pady=(0, 12))

        ttk.Label(controls, text="Species").pack(anchor="w")
        species = sorted(
            {
                name
                for simulation in self.project.simulations
                if simulation.species
                for name in simulation.species
            }
        )
        self.species_list = tk.Listbox(controls, selectmode="multiple", exportselection=False, height=18)
        for name in species:
            self.species_list.insert("end", name)
        if species:
            self.species_list.select_set(0, "end")
        self.species_list.bind("<<ListboxSelect>>", lambda _event: self._update_species_toggle_label())
        self.species_list.pack(fill="both", expand=True, pady=(0, 12))

        self.species_toggle_button = ttk.Button(
            controls,
            text="Deselect all species",
            command=self._toggle_species_selection,
        )
        self.species_toggle_button.pack(fill="x", pady=(0, 8))
        ttk.Button(controls, text="Plot", command=self._plot_species).pack(fill="x")
        self._update_species_toggle_label()

    def _build_thermo_tab(self, parent: ttk.Frame) -> None:
        controls = ttk.Frame(parent)
        controls.pack(side="left", fill="y", padx=8, pady=8)
        plot_area = ttk.Frame(parent)
        plot_area.pack(side="right", fill="both", expand=True, padx=8, pady=8)
        self._thermo_plot_area = plot_area

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
        ttk.Combobox(controls, textvariable=self.thermo_parameter, values=values, state="readonly").pack(
            fill="x", pady=(0, 12)
        )
        ttk.Button(controls, text="Plot", command=self._plot_thermo).pack(fill="x")

    def _build_smiles_tab(self, parent: ttk.Frame) -> None:
        controls = ttk.Frame(parent)
        controls.pack(side="left", fill="y", padx=8, pady=8)
        output = ttk.Frame(parent)
        output.pack(side="right", fill="both", expand=True, padx=8, pady=8)

        sim_values = [str(sim.index) for sim in self.project.simulations if sim.has_bond_data]
        self.smiles_simulation = tk.StringVar(value=sim_values[0] if sim_values else "")
        self.smiles_formula = tk.StringVar()
        self.smiles_value = tk.StringVar()

        ttk.Label(controls, text="Simulation").pack(anchor="w")
        self.smiles_sim_combo = ttk.Combobox(
            controls,
            textvariable=self.smiles_simulation,
            values=sim_values,
            state="readonly",
        )
        self.smiles_sim_combo.pack(fill="x", pady=(0, 12))
        self.smiles_sim_combo.bind("<<ComboboxSelected>>", lambda _event: self._refresh_formula_options())

        ttk.Label(controls, text="Formula/species").pack(anchor="w")
        self.smiles_formula_combo = ttk.Combobox(controls, textvariable=self.smiles_formula, state="readonly")
        self.smiles_formula_combo.pack(fill="x", pady=(0, 12))
        self.smiles_formula_combo.bind("<<ComboboxSelected>>", lambda _event: self._refresh_smiles_options())

        ttk.Label(controls, text="SMILES").pack(anchor="w")
        self.smiles_combo = ttk.Combobox(controls, textvariable=self.smiles_value, state="readonly", width=42)
        self.smiles_combo.pack(fill="x", pady=(0, 12))

        ttk.Button(controls, text="Generate", command=self._generate_molecule).pack(fill="x")

        self.molecule_label = ttk.Label(output)
        self.molecule_label.pack(fill="both", expand=True)
        self._refresh_formula_options()

    def _plot_species(self) -> None:
        try:
            simulations = self._selected_species_simulations()
            selected_species = [self.species_list.get(index) for index in self.species_list.curselection()]
            if not simulations or not selected_species:
                raise ValueError("Select at least one simulation and one species.")
            figure = plot_species(simulations, selected_species)
            self._replace_canvas("_species_canvas", self._species_plot_area, figure)
        except Exception as exc:  # pragma: no cover - GUI feedback.
            messagebox.showerror("Species plotting failed", str(exc))

    def _plot_thermo(self) -> None:
        try:
            parameter = self.thermo_parameter.get()
            if not parameter:
                raise ValueError("Select a thermodynamic parameter.")
            for canvas in self._thermo_canvases:
                self._destroy_canvas(canvas)
            self._thermo_canvases = []
            figures = plot_thermo(self.project.simulations, parameter)
            for figure in figures:
                canvas = FigureCanvasTkAgg(figure, master=self._thermo_plot_area)
                canvas.draw()
                canvas.get_tk_widget().pack(fill="both", expand=True)
                self._thermo_canvases.append(canvas)
        except Exception as exc:  # pragma: no cover - GUI feedback.
            messagebox.showerror("Thermo plotting failed", str(exc))

    def _generate_molecule(self) -> None:
        try:
            smiles = self.smiles_value.get()
            if not smiles:
                raise ValueError("Select a SMILES string.")
            self._molecule_photo = molecule_photo_image(smiles)
            self.molecule_label.configure(image=self._molecule_photo)
        except Exception as exc:  # pragma: no cover - GUI feedback.
            messagebox.showerror("SMILES visualization failed", str(exc))

    def _selected_species_simulations(self):
        available = [simulation for simulation in self.project.simulations if simulation.species_df is not None]
        return [available[index] for index in self.species_sim_list.curselection()]

    def _toggle_species_selection(self) -> None:
        if self.species_list.size() == 0:
            return
        if len(self.species_list.curselection()) == self.species_list.size():
            self.species_list.selection_clear(0, "end")
        else:
            self.species_list.select_set(0, "end")
        self._update_species_toggle_label()

    def _update_species_toggle_label(self) -> None:
        if not hasattr(self, "species_toggle_button"):
            return
        if self.species_list.size() == 0:
            self.species_toggle_button.configure(text="Select all species")
        elif len(self.species_list.curselection()) == self.species_list.size():
            self.species_toggle_button.configure(text="Deselect all species")
        else:
            self.species_toggle_button.configure(text="Select all species")

    def close(self) -> None:
        """Close the GUI and release Matplotlib/Tk resources promptly."""

        if self._closed:
            return
        self._closed = True

        if self._species_canvas is not None:
            self._destroy_canvas(self._species_canvas)
            self._species_canvas = None

        for canvas in self._thermo_canvases:
            self._destroy_canvas(canvas)
        self._thermo_canvases = []

        self._molecule_photo = None
        plt.close("all")

        try:
            self.root.quit()
            self.root.destroy()
        except tk.TclError:
            pass

    def _refresh_formula_options(self) -> None:
        simulation = self._selected_smiles_simulation()
        formulas = formulas_for_simulation(simulation.chem_formulas) if simulation and simulation.chem_formulas else []
        self.smiles_formula_combo.configure(values=formulas)
        self.smiles_formula.set(formulas[0] if formulas else "")
        self._refresh_smiles_options()

    def _refresh_smiles_options(self) -> None:
        simulation = self._selected_smiles_simulation()
        formula = self.smiles_formula.get()
        values = []
        if simulation and simulation.chem_formulas and simulation.smiles and formula:
            values = smiles_for_formula(simulation.chem_formulas, simulation.smiles, formula)
        self.smiles_combo.configure(values=values)
        self.smiles_value.set(values[0] if values else "")

    def _selected_smiles_simulation(self):
        value = self.smiles_simulation.get()
        if not value:
            return None
        return self.project.simulation(int(value))

    def _replace_canvas(self, attr_name: str, parent: ttk.Frame, figure) -> None:
        old_canvas = getattr(self, attr_name)
        if old_canvas is not None:
            self._destroy_canvas(old_canvas)
        canvas = FigureCanvasTkAgg(figure, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        setattr(self, attr_name, canvas)

    def _destroy_canvas(self, canvas: FigureCanvasTkAgg) -> None:
        try:
            plt.close(canvas.figure)
            canvas.get_tk_widget().destroy()
        except tk.TclError:
            pass


def launch_gui(project: LammpalyzeProject) -> None:
    """Launch the lammpalyze GUI."""

    gui = LammpalyzeGUI(project)
    try:
        gui.run()
    finally:
        gui.close()
