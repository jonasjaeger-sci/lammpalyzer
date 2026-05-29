"""Shared Matplotlib canvas helpers for the Tkinter GUI."""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from lammpalyze.gui.helpers import IMAGE_FILETYPES, image_output_path, suffixed_image_output_path


class CanvasMixin:
    """Helpers for replacing and destroying embedded Matplotlib canvases."""

    def _replace_canvas(self, attr_name: str, parent: ttk.Frame, figure) -> None:
        """Replace one stored Matplotlib canvas with a new figure canvas."""

        old_canvas = getattr(self, attr_name)
        if old_canvas is not None:
            self._destroy_canvas(old_canvas)
        canvas = FigureCanvasTkAgg(figure, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        setattr(self, attr_name, canvas)

    def _destroy_canvas(self, canvas: FigureCanvasTkAgg) -> None:
        """Close and destroy a Matplotlib canvas widget."""

        try:
            plt.close(canvas.figure)
            canvas.get_tk_widget().destroy()
        except tk.TclError:
            pass

    def _ask_image_output_path(self, title: str, initialfile: str, filetypes=IMAGE_FILETYPES) -> str:
        """Ask the user where an image should be saved."""

        return filedialog.asksaveasfilename(
            title=title,
            initialfile=initialfile,
            defaultextension=".png",
            filetypes=filetypes,
        )

    def _save_canvas_figure(self, canvas: FigureCanvasTkAgg | None, title: str, initialfile: str) -> None:
        """Save one displayed Matplotlib canvas to an image file."""

        if canvas is None:
            messagebox.showerror("Save failed", "Generate a plot before saving.")
            return

        filename = self._ask_image_output_path(title, initialfile)
        if not filename:
            return
        output_path = image_output_path(filename)
        canvas.figure.savefig(output_path, bbox_inches="tight")
        messagebox.showinfo("Plot saved", f"Saved plot to {output_path}")

    def _save_canvas_figures(
        self,
        canvases: list[FigureCanvasTkAgg],
        title: str,
        initialfile: str,
        suffixes: list[str],
    ) -> None:
        """Save multiple displayed Matplotlib canvases using one chosen base name."""

        if not canvases:
            messagebox.showerror("Save failed", "Generate a plot before saving.")
            return

        filename = self._ask_image_output_path(title, initialfile)
        if not filename:
            return

        if len(canvases) == 1:
            output_paths = [image_output_path(filename)]
        else:
            output_paths = [
                suffixed_image_output_path(
                    filename,
                    suffixes[index] if index < len(suffixes) else f"plot_{index + 1}",
                )
                for index in range(len(canvases))
            ]
        for canvas, output_path in zip(canvases, output_paths, strict=False):
            canvas.figure.savefig(output_path, bbox_inches="tight")
        messagebox.showinfo("Plots saved", "Saved plot files:\n" + "\n".join(str(path) for path in output_paths))

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
