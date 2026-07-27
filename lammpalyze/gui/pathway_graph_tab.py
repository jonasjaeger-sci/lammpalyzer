"""Connected-pathway graph tab for the Tkinter GUI."""

from __future__ import annotations

import queue
import shutil
import tempfile
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk

import matplotlib.image as mpimg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch

from lammpalyze.analysis import LammpalyzeProject
from lammpalyze.gui.pathway_graph import (
    PathwayGraph,
    PathwayGraphNode,
    build_pathway_graph,
    pathway_graph_choices,
    pathway_graph_image_extent,
)
from lammpalyze.ovito import create_reaction_state_snapshot
from lammpalyze.reactions import ConnectedReactionPathway, ConnectedReactionStep

PATHWAY_GRAPH_DPI = 100
PATHWAY_GRAPH_MIN_SIZE = (1500, 980)
PATHWAY_GRAPH_NODE_SIZE = (6.2, 4.0)
PATHWAY_GRAPH_NODE_SPACING = (7.8, 6.2)
PATHWAY_GRAPH_SNAPSHOT_SIZE = (980, 700)

SnapshotCacheKey = tuple[str, int, str, str]


@dataclass(frozen=True)
class PathwaySnapshotTask:
    """One graph-node snapshot to render outside Tk's main thread."""

    cache_key: SnapshotCacheKey
    step: ConnectedReactionStep
    side: str
    output_dir: Path


@dataclass(frozen=True)
class PathwaySnapshotResult:
    """The result of one background graph-node snapshot render."""

    cache_key: SnapshotCacheKey
    image_file: Path | None
    renderer: str | None
    error: str | None


def remove_pathway_snapshot_directory(path: Path | None) -> None:
    """Remove graph snapshot files created for one GUI session."""

    if path is not None:
        shutil.rmtree(path, ignore_errors=True)


def render_pathway_snapshots_worker(
    result_queue: queue.Queue,
    token: int,
    project: LammpalyzeProject,
    notation: str,
    tasks: list[PathwaySnapshotTask],
    cancel_event: threading.Event,
) -> None:
    """Render pathway snapshots without retaining or touching Tk objects."""

    for task in tasks:
        if cancel_event.is_set():
            break
        try:
            simulation, connected_occurrence = project.first_connected_reaction_occurrence(
                task.step,
                notation=notation,
            )
            side = task.side
            if connected_occurrence.matched_direction == "reverse":
                side = "products" if side == "reactants" else "reactants"
            snapshot = create_reaction_state_snapshot(
                simulation,
                connected_occurrence.occurrence,
                side,
                output_dir=task.output_dir,
                image_size=PATHWAY_GRAPH_SNAPSHOT_SIZE,
            )
            result = PathwaySnapshotResult(
                cache_key=task.cache_key,
                image_file=snapshot.image_file,
                renderer=snapshot.renderer,
                error=None,
            )
        except Exception as exc:
            result = PathwaySnapshotResult(
                cache_key=task.cache_key,
                image_file=None,
                renderer=None,
                error=str(exc),
            )
        result_queue.put(("snapshot", token, result))
    result_queue.put(("complete", token, None))


class PathwayGraphTabMixin:
    """Build and manage the connected-pathway graph tab."""

    def _build_pathway_graph_tab(self, parent: ttk.Frame) -> None:
        """Create a visual top-down graph for connected reaction pathways."""

        controls = ttk.Frame(parent)
        controls.pack(fill="x", padx=8, pady=8)
        ttk.Label(controls, text="Notation").pack(side="left", padx=(0, 8))
        self.pathway_graph_notation = tk.StringVar(value="formula")
        ttk.Radiobutton(
            controls,
            text="Formula",
            variable=self.pathway_graph_notation,
            value="formula",
            command=self._refresh_pathway_graph_choices,
        ).pack(side="left")
        ttk.Radiobutton(
            controls,
            text="SMILES",
            variable=self.pathway_graph_notation,
            value="smiles",
            command=self._refresh_pathway_graph_choices,
        ).pack(side="left", padx=(8, 0))
        ttk.Label(controls, text="Minimum total occurrences").pack(side="left", padx=(20, 8))
        self.pathway_graph_min_count = tk.StringVar(value="1")
        threshold_input = ttk.Spinbox(
            controls,
            from_=1,
            to=1_000_000,
            increment=1,
            textvariable=self.pathway_graph_min_count,
            width=8,
            command=self._refresh_pathway_graph_choices,
        )
        threshold_input.pack(side="left")
        threshold_input.bind("<Return>", self._refresh_pathway_graph_choices)
        threshold_input.bind("<FocusOut>", self._refresh_pathway_graph_choices)

        selector_frame = ttk.Frame(parent)
        selector_frame.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Label(selector_frame, text="Pathway").pack(side="left", padx=(0, 8))
        self.pathway_graph_choice_value = tk.StringVar()
        self.pathway_graph_choice = ttk.Combobox(
            selector_frame,
            textvariable=self.pathway_graph_choice_value,
            state="readonly",
            width=90,
        )
        self.pathway_graph_choice.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.pathway_graph_choice.bind("<<ComboboxSelected>>", self._refresh_pathway_graph)
        ttk.Button(
            selector_frame,
            text="Refresh",
            command=self._refresh_pathway_graph_choices,
        ).pack(side="right")

        self.pathway_graph_status = ttk.Label(parent, text="", anchor="w", justify="left")
        self.pathway_graph_status.pack(fill="x", padx=8, pady=(0, 4))
        self.pathway_graph_progress_value = tk.IntVar(value=0)
        self.pathway_graph_progress = ttk.Progressbar(
            parent,
            mode="determinate",
            variable=self.pathway_graph_progress_value,
            maximum=1,
        )

        self.pathway_graph_frame = ttk.Frame(parent)
        self.pathway_graph_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.pathway_graph_scroll_canvas = tk.Canvas(self.pathway_graph_frame, highlightthickness=0)
        graph_y_scrollbar = ttk.Scrollbar(
            self.pathway_graph_frame,
            orient="vertical",
            command=self.pathway_graph_scroll_canvas.yview,
        )
        graph_x_scrollbar = ttk.Scrollbar(
            self.pathway_graph_frame,
            orient="horizontal",
            command=self.pathway_graph_scroll_canvas.xview,
        )
        self.pathway_graph_scroll_canvas.configure(
            yscrollcommand=graph_y_scrollbar.set,
            xscrollcommand=graph_x_scrollbar.set,
        )
        self.pathway_graph_scroll_canvas.grid(row=0, column=0, sticky="nsew")
        graph_y_scrollbar.grid(row=0, column=1, sticky="ns")
        graph_x_scrollbar.grid(row=1, column=0, sticky="ew")
        self.pathway_graph_frame.rowconfigure(0, weight=1)
        self.pathway_graph_frame.columnconfigure(0, weight=1)

        self.pathway_graph_canvas_frame = ttk.Frame(self.pathway_graph_scroll_canvas)
        self.pathway_graph_canvas_window = self.pathway_graph_scroll_canvas.create_window(
            (0, 0),
            window=self.pathway_graph_canvas_frame,
            anchor="nw",
        )
        self.pathway_graph_canvas_frame.bind("<Configure>", self._update_pathway_graph_scroll_region)
        self.pathway_graph_scroll_canvas.bind("<Configure>", self._sync_pathway_graph_canvas_width)
        self.pathway_graph_scroll_canvas.bind("<MouseWheel>", self._scroll_pathway_graph_vertically)
        self.pathway_graph_scroll_canvas.bind("<Shift-MouseWheel>", self._scroll_pathway_graph_horizontally)
        self.pathway_graph_scroll_canvas.bind("<Button-4>", self._scroll_pathway_graph_vertically)
        self.pathway_graph_scroll_canvas.bind("<Button-5>", self._scroll_pathway_graph_vertically)

        self.pathway_graph_figure = Figure(
            figsize=(
                PATHWAY_GRAPH_MIN_SIZE[0] / PATHWAY_GRAPH_DPI,
                PATHWAY_GRAPH_MIN_SIZE[1] / PATHWAY_GRAPH_DPI,
            ),
            dpi=PATHWAY_GRAPH_DPI,
        )
        self.pathway_graph_axis = self.pathway_graph_figure.add_subplot(111)
        self._pathway_graph_canvas = FigureCanvasTkAgg(
            self.pathway_graph_figure,
            master=self.pathway_graph_canvas_frame,
        )
        self._pathway_graph_canvas.get_tk_widget().pack(anchor="nw")

        self._pathway_graph_choices: list[tuple[int, str, str]] = []
        self._pathway_graph_pathways: list[ConnectedReactionPathway] = []
        self._pathway_graph_image_cache: dict[SnapshotCacheKey, Path] = {}
        self._pathway_graph_renderer_cache: dict[SnapshotCacheKey, str] = {}
        self._pathway_graph_error_cache: dict[SnapshotCacheKey, str] = {}
        self._pathway_graph_snapshot_dir: Path | None = None
        self._pathway_graph_result_queue: queue.Queue = queue.Queue()
        self._pathway_graph_worker_thread: threading.Thread | None = None
        self._pathway_graph_cancel_event: threading.Event | None = None
        self._pathway_graph_poll_job: str | None = None
        self._pathway_graph_worker_token = 0
        self._pathway_graph_completed_snapshots = 0
        self._pathway_graph_total_snapshots = 0
        self._pathway_graph_current: tuple[PathwayGraph, ConnectedReactionPathway] | None = None
        self._refresh_pathway_graph_choices()

    def _refresh_pathway_graph_choices(self, _event=None) -> None:
        """Refresh the selectable pathway branches for the graph tab."""

        self._cancel_pathway_graph_worker()
        self._pathway_graph_error_cache.clear()
        notation = self.pathway_graph_notation.get()
        min_count = self._pathway_graph_threshold()
        self._pathway_graph_pathways = self.project.connected_reaction_pathways(
            notation=notation,
            min_count=min_count,
        )
        choices = []
        for pathway in self._pathway_graph_pathways:
            for label, display in pathway_graph_choices(pathway):
                choices.append((pathway.index, label, display))

        self._pathway_graph_choices = choices
        values = [display for _, _, display in choices]
        self.pathway_graph_choice.configure(values=values)
        current = self.pathway_graph_choice_value.get()
        if current not in values:
            self.pathway_graph_choice_value.set("")
        self._refresh_pathway_graph()

    def _refresh_pathway_graph(self, _event=None) -> None:
        """Redraw the selected pathway graph and queue missing snapshots."""

        self._cancel_pathway_graph_worker()
        self.pathway_graph_axis.clear()
        selected = self.pathway_graph_choice_value.get()
        selected_choice = next(
            (choice for choice in self._pathway_graph_choices if choice[2] == selected),
            None,
        )
        if selected_choice is None:
            message = (
                "Select a pathway to visualize."
                if self._pathway_graph_choices
                else "No connected reaction pathways available."
            )
            self._pathway_graph_current = None
            self._draw_empty_pathway_graph(message)
            return

        pathway_index, root_label, _display = selected_choice
        pathway = next(
            pathway
            for pathway in self._pathway_graph_pathways
            if pathway.index == pathway_index
        )
        graph = build_pathway_graph(pathway, root_label=root_label)
        self._pathway_graph_current = (graph, pathway)
        self._draw_pathway_graph(graph, pathway, queue_missing=True)

    def _draw_empty_pathway_graph(self, message: str) -> None:
        """Draw a centered empty-state message in the graph area."""

        axis = self.pathway_graph_axis
        axis.set_axis_off()
        axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes)
        self.pathway_graph_status.configure(text=message)
        self._hide_pathway_graph_progress()
        self._pathway_graph_canvas.draw_idle()

    def _draw_pathway_graph(
        self,
        graph: PathwayGraph,
        pathway: ConnectedReactionPathway,
        *,
        queue_missing: bool,
    ) -> None:
        """Draw graph nodes and queue uncached state snapshots."""

        axis = self.pathway_graph_axis
        axis.clear()
        axis.set_axis_off()
        axis.set_aspect("equal", adjustable="box")
        if not graph.nodes:
            self._draw_empty_pathway_graph("No steps found for the selected pathway.")
            return

        positions = self._pathway_graph_positions(graph)
        self._resize_pathway_graph_figure(positions)
        steps_by_label = {step.label: step for step in pathway.steps}
        render_messages = []
        tasks = []
        queued_keys = set()
        node_width, node_height = PATHWAY_GRAPH_NODE_SIZE
        for edge in graph.edges:
            x1, y1 = positions[edge.source_key]
            x2, y2 = positions[edge.target_key]
            arrow_style = "<->" if edge.arrow == "<->" else "->"
            axis.annotate(
                "",
                xy=(x2, y2 + node_height * 0.48),
                xytext=(x1, y1 - node_height * 0.48),
                arrowprops={
                    "arrowstyle": arrow_style,
                    "lw": 2.2,
                    "color": "#303030",
                    "mutation_scale": 22,
                    "shrinkA": 8,
                    "shrinkB": 8,
                },
            )
            axis.text(
                0.5 * (x1 + x2),
                0.5 * (y1 + y2),
                f"{edge.step_label}  n={edge.count}",
                ha="center",
                va="center",
                fontsize=10,
                bbox={"boxstyle": "round,pad=0.28", "fc": "white", "ec": "#d0d0d0", "alpha": 0.94},
            )

        for node in graph.nodes:
            cache_key = self._pathway_graph_cache_key(pathway, node)
            image_path = self._pathway_graph_image_cache.get(cache_key)
            renderer = self._pathway_graph_renderer_cache.get(cache_key)
            error = self._pathway_graph_error_cache.get(cache_key)
            if renderer is not None:
                render_messages.append(f"{node.source_step_label}/{node.source_side}: {renderer}")
            elif error is not None:
                render_messages.append(f"{node.source_step_label}/{node.source_side}: {error}")
            elif queue_missing and cache_key not in queued_keys:
                task = self._pathway_graph_snapshot_task(pathway, node, steps_by_label, cache_key)
                if task is not None:
                    tasks.append(task)
                    queued_keys.add(cache_key)

            x, y = positions[node.key]
            self._draw_pathway_graph_node(axis, node.label, x, y, node_width, node_height, image_path)

        x_values = [position[0] for position in positions.values()]
        y_values = [position[1] for position in positions.values()]
        axis.set_xlim(min(x_values) - 3.8, max(x_values) + 3.8)
        axis.set_ylim(min(y_values) - 2.8, max(y_values) + 2.8)
        self._pathway_graph_canvas.draw_idle()
        self.root.after_idle(self._update_pathway_graph_scroll_region)
        if tasks:
            self._start_pathway_graph_worker(tasks)
        else:
            self._hide_pathway_graph_progress()
            self.pathway_graph_status.configure(text=self._pathway_graph_status_text(render_messages))

    def _pathway_graph_snapshot_task(
        self,
        pathway: ConnectedReactionPathway,
        node: PathwayGraphNode,
        steps_by_label: dict[str, ConnectedReactionStep],
        cache_key: SnapshotCacheKey,
    ) -> PathwaySnapshotTask | None:
        """Build one worker task for a graph node with a concrete source state."""

        if node.source_step_label is None or node.source_side is None:
            return None
        step = steps_by_label.get(node.source_step_label)
        if step is None:
            return None
        if self._pathway_graph_snapshot_dir is None:
            self._pathway_graph_snapshot_dir = Path(tempfile.mkdtemp(prefix="lammpalyze_pathway_graph_"))
        output_dir = self._pathway_graph_snapshot_dir / self._pathway_graph_safe_name(
            str(pathway.index),
            *cache_key,
        )
        return PathwaySnapshotTask(
            cache_key=cache_key,
            step=step,
            side=node.source_side,
            output_dir=output_dir,
        )

    def _start_pathway_graph_worker(self, tasks: list[PathwaySnapshotTask]) -> None:
        """Render missing node snapshots while leaving Tk responsive."""

        self._cancel_pathway_graph_worker()
        token = self._pathway_graph_worker_token
        cancel_event = threading.Event()
        self._pathway_graph_cancel_event = cancel_event
        self._pathway_graph_completed_snapshots = 0
        self._pathway_graph_total_snapshots = len(tasks)
        self.pathway_graph_progress.configure(maximum=max(1, len(tasks)))
        self.pathway_graph_progress_value.set(0)
        self.pathway_graph_progress.pack(
            fill="x",
            padx=8,
            pady=(0, 4),
            before=self.pathway_graph_frame,
        )
        self.pathway_graph_status.configure(text=f"Rendering pathway snapshots: 0/{len(tasks)}")
        worker = threading.Thread(
            target=render_pathway_snapshots_worker,
            args=(
                self._pathway_graph_result_queue,
                token,
                self.project,
                self.pathway_graph_notation.get(),
                tasks,
                cancel_event,
            ),
            daemon=True,
            name="lammpalyze-pathway-snapshots",
        )
        self._pathway_graph_worker_thread = worker
        worker.start()
        self._pathway_graph_poll_job = self.root.after(100, self._poll_pathway_graph_worker)

    def _poll_pathway_graph_worker(self) -> None:
        """Transfer completed worker results to Tk-owned graph state."""

        self._pathway_graph_poll_job = None
        active_token = self._pathway_graph_worker_token
        completed = False
        while True:
            try:
                event, token, payload = self._pathway_graph_result_queue.get_nowait()
            except queue.Empty:
                break
            if token != active_token:
                continue
            if event == "snapshot":
                self._store_pathway_graph_snapshot(payload)
                self._pathway_graph_completed_snapshots += 1
                self.pathway_graph_progress_value.set(self._pathway_graph_completed_snapshots)
                self.pathway_graph_status.configure(
                    text=(
                        "Rendering pathway snapshots: "
                        f"{self._pathway_graph_completed_snapshots}/{self._pathway_graph_total_snapshots}"
                    )
                )
            elif event == "complete":
                completed = True

        if completed:
            self._pathway_graph_worker_thread = None
            self._pathway_graph_cancel_event = None
            self._hide_pathway_graph_progress()
            if self._pathway_graph_current is not None:
                graph, pathway = self._pathway_graph_current
                self._draw_pathway_graph(graph, pathway, queue_missing=False)
            return
        if self._pathway_graph_worker_thread is not None:
            self._pathway_graph_poll_job = self.root.after(100, self._poll_pathway_graph_worker)

    def _store_pathway_graph_snapshot(self, result: PathwaySnapshotResult) -> None:
        """Store one background result for the next graph redraw."""

        if result.image_file is not None and result.renderer is not None:
            self._pathway_graph_image_cache[result.cache_key] = result.image_file
            self._pathway_graph_renderer_cache[result.cache_key] = result.renderer
            self._pathway_graph_error_cache.pop(result.cache_key, None)
        elif result.error is not None:
            self._pathway_graph_error_cache[result.cache_key] = result.error

    def _cancel_pathway_graph_worker(self) -> None:
        """Invalidate pending graph results and cancel future worker tasks."""

        self._pathway_graph_worker_token += 1
        if self._pathway_graph_cancel_event is not None:
            self._pathway_graph_cancel_event.set()
        self._pathway_graph_cancel_event = None
        self._pathway_graph_worker_thread = None
        if self._pathway_graph_poll_job is not None:
            try:
                self.root.after_cancel(self._pathway_graph_poll_job)
            except tk.TclError:
                pass
        self._pathway_graph_poll_job = None
        self._hide_pathway_graph_progress()

    def _close_pathway_graph_tab(self) -> None:
        """Release graph workers, temporary files, and canvas resources."""

        self._cancel_pathway_graph_worker()
        self._pathway_graph_current = None
        self._pathway_graph_image_cache.clear()
        self._pathway_graph_renderer_cache.clear()
        self._pathway_graph_error_cache.clear()
        remove_pathway_snapshot_directory(self._pathway_graph_snapshot_dir)
        self._pathway_graph_snapshot_dir = None
        if self._pathway_graph_canvas is not None:
            self._destroy_canvas(self._pathway_graph_canvas)
        self._pathway_graph_canvas = None

    def _pathway_graph_cache_key(
        self,
        pathway: ConnectedReactionPathway,
        node: PathwayGraphNode,
    ) -> SnapshotCacheKey:
        """Return a cache key unique across notation, pathway, step, and side."""

        return (
            self.pathway_graph_notation.get(),
            pathway.index,
            node.source_step_label or "",
            node.source_side or "",
        )

    def _pathway_graph_positions(self, graph: PathwayGraph) -> dict[str, tuple[float, float]]:
        """Return deterministic top-down node positions."""

        nodes_by_depth: dict[int, list] = {}
        for node in graph.nodes:
            nodes_by_depth.setdefault(node.depth, []).append(node)
        positions = {}
        for depth, nodes in sorted(nodes_by_depth.items()):
            count = len(nodes)
            for index, node in enumerate(nodes):
                x = (index - (count - 1) / 2) * PATHWAY_GRAPH_NODE_SPACING[0]
                y = -depth * PATHWAY_GRAPH_NODE_SPACING[1]
                positions[node.key] = (x, y)
        return positions

    def _resize_pathway_graph_figure(self, positions: dict[str, tuple[float, float]]) -> None:
        """Resize the graph canvas so large pathways can be scrolled."""

        if not positions:
            width_pixels, height_pixels = PATHWAY_GRAPH_MIN_SIZE
        else:
            depths = {round(-position[1] / PATHWAY_GRAPH_NODE_SPACING[1]) for position in positions.values()}
            rows = max(1, len(depths))
            max_columns = max(
                sum(1 for position in positions.values() if position[1] == y_value)
                for y_value in {position[1] for position in positions.values()}
            )
            width_pixels = max(PATHWAY_GRAPH_MIN_SIZE[0], 680 * max_columns + 320)
            height_pixels = max(PATHWAY_GRAPH_MIN_SIZE[1], 560 * rows + 300)
        self.pathway_graph_figure.set_size_inches(
            width_pixels / PATHWAY_GRAPH_DPI,
            height_pixels / PATHWAY_GRAPH_DPI,
            forward=True,
        )
        self._pathway_graph_canvas.get_tk_widget().configure(
            width=width_pixels,
            height=height_pixels,
        )

    def _draw_pathway_graph_node(
        self,
        axis,
        label: str,
        x: float,
        y: float,
        width: float,
        height: float,
        image_path: Path | None,
    ) -> None:
        """Draw one graph node with an optional rendered state image."""

        patch = FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.08,rounding_size=0.08",
            facecolor="#ffffff",
            edgecolor="#9a9a9a",
            linewidth=1.0,
            zorder=1,
        )
        axis.add_patch(patch)
        if image_path is not None and image_path.exists():
            try:
                image = mpimg.imread(image_path)
                axis.imshow(
                    image,
                    extent=pathway_graph_image_extent(x, y, width, height, image.shape),
                    zorder=2,
                    aspect="equal",
                )
            except Exception:
                image_path = None
        text_y = y - height * 0.39 if image_path is not None else y
        axis.text(
            x,
            text_y,
            label,
            ha="center",
            va="center",
            fontsize=11,
            wrap=True,
            zorder=3,
        )

    def _pathway_graph_status_text(self, messages: list[str]) -> str:
        """Return a compact renderer/status message for the graph tab."""

        if not messages:
            return "Graph rendered without frame snapshots."
        renderers = sorted({message.rsplit(": ", maxsplit=1)[-1] for message in messages})
        if len(renderers) == 1 and renderers[0] in {"matplotlib", "ovito-python"}:
            return f"Frame snapshots rendered with {renderers[0]}."
        return "Snapshot details: " + "; ".join(messages[:4])

    @staticmethod
    def _pathway_graph_safe_name(*parts) -> str:
        """Return a filesystem-safe cache directory name."""

        raw = "_".join(str(part) for part in parts)
        safe = "".join(character if character.isalnum() else "_" for character in raw)
        return safe[:120] or "snapshot"

    def _pathway_graph_threshold(self) -> int:
        """Return the selected graph minimum pathway count."""

        try:
            value = int(self.pathway_graph_min_count.get())
        except ValueError:
            value = 1
        value = max(1, value)
        self.pathway_graph_min_count.set(str(value))
        return value

    def _hide_pathway_graph_progress(self) -> None:
        """Hide the graph progress bar when no render is active."""

        if hasattr(self, "pathway_graph_progress"):
            self.pathway_graph_progress.pack_forget()

    def _update_pathway_graph_scroll_region(self, _event=None) -> None:
        """Refresh the scrollable area around the pathway graph canvas."""

        if hasattr(self, "pathway_graph_scroll_canvas"):
            self.pathway_graph_scroll_canvas.configure(
                scrollregion=self.pathway_graph_scroll_canvas.bbox("all")
            )

    def _sync_pathway_graph_canvas_width(self, event) -> None:
        """Keep small graphs filling the viewport while allowing large graphs to scroll."""

        if not hasattr(self, "pathway_graph_canvas_window"):
            return
        requested_width = self.pathway_graph_canvas_frame.winfo_reqwidth()
        self.pathway_graph_scroll_canvas.itemconfigure(
            self.pathway_graph_canvas_window,
            width=max(event.width, requested_width),
        )

    def _scroll_pathway_graph_vertically(self, event) -> str:
        """Scroll the pathway graph vertically with the mouse wheel."""

        direction = self._pathway_graph_scroll_direction(event)
        self.pathway_graph_scroll_canvas.yview_scroll(direction * 3, "units")
        return "break"

    def _scroll_pathway_graph_horizontally(self, event) -> str:
        """Scroll the pathway graph horizontally with shift-wheel."""

        direction = self._pathway_graph_scroll_direction(event)
        self.pathway_graph_scroll_canvas.xview_scroll(direction * 3, "units")
        return "break"

    @staticmethod
    def _pathway_graph_scroll_direction(event) -> int:
        """Return a normalized Tk mouse-wheel scroll direction."""

        if getattr(event, "num", None) == 4:
            return -1
        if getattr(event, "num", None) == 5:
            return 1
        return -1 if event.delta > 0 else 1
