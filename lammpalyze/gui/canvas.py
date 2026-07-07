"""Shared Matplotlib canvas helpers for the Tkinter GUI."""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from lammpalyze.gui.helpers import IMAGE_FILETYPES, PNG_FILETYPES, image_output_path, suffixed_image_output_path


class CanvasMixin:
    """Helpers for replacing and destroying embedded Matplotlib canvases."""

    def _replace_canvas(self, attr_name: str, parent: ttk.Frame, figure) -> None:
        """Replace one stored Matplotlib canvas with a new figure canvas."""

        old_canvas = getattr(self, attr_name)
        if old_canvas is not None:
            self._destroy_canvas(old_canvas)
        canvas = self._create_figure_canvas(figure, parent)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        setattr(self, attr_name, canvas)

    def _create_figure_canvas(self, figure, parent: ttk.Frame) -> FigureCanvasTkAgg:
        """Create a Tk canvas with shared line-value hover annotations."""

        canvas = FigureCanvasTkAgg(figure, master=parent)
        _connect_line_hover(canvas)
        canvas.draw()
        return canvas

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

    def _export_canvas_figure_png(self, canvas: FigureCanvasTkAgg | None, title: str, initialfile: str) -> None:
        """Export one displayed Matplotlib canvas to a PNG file."""

        if canvas is None:
            messagebox.showerror("Export failed", "Generate a plot before exporting.")
            return

        filename = self._ask_image_output_path(title, initialfile, filetypes=PNG_FILETYPES)
        if not filename:
            return
        output_path = image_output_path(filename).with_suffix(".png")
        canvas.figure.savefig(output_path, bbox_inches="tight", format="png")
        messagebox.showinfo("PNG exported", f"Exported PNG to {output_path}")

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

    def _export_canvas_figures_png(
        self,
        canvases: list[FigureCanvasTkAgg],
        title: str,
        initialfile: str,
        suffixes: list[str],
    ) -> None:
        """Export multiple displayed Matplotlib canvases as PNG files."""

        if not canvases:
            messagebox.showerror("Export failed", "Generate a plot before exporting.")
            return

        filename = self._ask_image_output_path(title, initialfile, filetypes=PNG_FILETYPES)
        if not filename:
            return

        base_path = image_output_path(filename).with_suffix(".png")
        if len(canvases) == 1:
            output_paths = [base_path]
        else:
            output_paths = [
                suffixed_image_output_path(
                    str(base_path),
                    suffixes[index] if index < len(suffixes) else f"plot_{index + 1}",
                ).with_suffix(".png")
                for index in range(len(canvases))
            ]
        for canvas, output_path in zip(canvases, output_paths, strict=False):
            canvas.figure.savefig(output_path, bbox_inches="tight", format="png")
        messagebox.showinfo("PNGs exported", "Exported PNG files:\n" + "\n".join(str(path) for path in output_paths))

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


def _connect_line_hover(canvas: FigureCanvasTkAgg, tolerance: float = 10.0) -> None:
    """Show x/y values for the nearest labelled line point under the mouse."""

    annotations = {}
    state = {"selection": None}

    def hide_annotations() -> bool:
        changed = False
        for annotation in annotations.values():
            if annotation.get_visible():
                annotation.set_visible(False)
                changed = True
        state["selection"] = None
        return changed

    def on_motion(event) -> None:
        if event.x is None or event.y is None:
            if hide_annotations():
                canvas.draw_idle()
            return
        nearest = _nearest_line_point(canvas.figure, event.x, event.y, tolerance)
        if nearest is None:
            if hide_annotations():
                canvas.draw_idle()
            return

        line, point_index, x_value, y_value = nearest
        selection = (id(line), point_index)
        if selection == state["selection"]:
            return
        hide_annotations()
        axis = line.axes
        annotation = annotations.get(axis)
        if annotation is None:
            annotation = axis.annotate(
                "",
                xy=(x_value, y_value),
                xytext=(10, 12),
                textcoords="offset points",
                color="#111827",
                fontsize="small",
                bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#64748b", "alpha": 0.95},
                arrowprops={"arrowstyle": "->", "color": "#64748b"},
                zorder=100,
            )
            annotations[axis] = annotation
        label = line.get_label()
        annotation.xy = (x_value, y_value)
        annotation.set_text(f"{label}\nx = {_format_hover_value(x_value)}\ny = {_format_hover_value(y_value)}")
        annotation.set_visible(True)
        state["selection"] = selection
        canvas.draw_idle()

    canvas.mpl_connect("motion_notify_event", on_motion)
    canvas.mpl_connect("figure_leave_event", lambda _event: canvas.draw_idle() if hide_annotations() else None)


def _nearest_line_point(figure, x_pixel: float, y_pixel: float, tolerance: float = 10.0):
    """Return the closest labelled, finite line point within a pixel radius."""

    best = None
    best_distance_squared = tolerance * tolerance
    for axis in figure.axes:
        if not axis.get_visible() or not axis.bbox.contains(x_pixel, y_pixel):
            continue
        for line in axis.get_lines():
            label = line.get_label()
            if not line.get_visible() or not label or label.startswith("_"):
                continue
            points = np.asarray(line.get_xydata(), dtype=float)
            if points.ndim != 2 or points.shape[1] != 2 or points.size == 0:
                continue
            finite = np.isfinite(points).all(axis=1)
            if not finite.any():
                continue
            finite_indices = np.flatnonzero(finite)
            display_points = line.get_transform().transform(points[finite])
            distances_squared = np.sum(
                (display_points - np.array([x_pixel, y_pixel])) ** 2,
                axis=1,
            )
            local_index = int(np.argmin(distances_squared))
            distance_squared = float(distances_squared[local_index])
            if distance_squared <= best_distance_squared:
                point_index = int(finite_indices[local_index])
                best_distance_squared = distance_squared
                best = (line, point_index, float(points[point_index, 0]), float(points[point_index, 1]))
    return best


def _format_hover_value(value: float) -> str:
    """Format graph coordinates compactly while retaining useful precision."""

    return f"{value:.7g}"
