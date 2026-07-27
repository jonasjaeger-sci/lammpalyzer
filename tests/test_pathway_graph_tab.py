"""Tests for pathway-graph background rendering helpers."""

from __future__ import annotations

import queue
import threading
from pathlib import Path
from types import SimpleNamespace

from lammpalyze.gui import pathway_graph_tab
from lammpalyze.gui.pathway_graph_tab import (
    PathwaySnapshotTask,
    remove_pathway_snapshot_directory,
    render_pathway_snapshots_worker,
)
from lammpalyze.reactions import ConnectedReactionStep


def _pathway_step() -> ConnectedReactionStep:
    return ConnectedReactionStep(
        label="A",
        parents=(),
        depth=1,
        source="A",
        arrow="<->",
        target="B",
        count=1,
        simulations=(1,),
        counts_by_simulation=((1, 1),),
    )


def test_snapshot_worker_reports_results_and_reverses_matched_side(tmp_path: Path, monkeypatch):
    """Render snapshots without Tk and honor reverse pathway occurrences."""

    rendered = []
    occurrence = SimpleNamespace()
    connected_occurrence = SimpleNamespace(
        occurrence=occurrence,
        matched_direction="reverse",
    )
    project = SimpleNamespace(
        first_connected_reaction_occurrence=lambda step, notation: (
            SimpleNamespace(index=1),
            connected_occurrence,
        )
    )

    def fake_snapshot(simulation, selected_occurrence, side, *, output_dir, image_size):
        rendered.append((simulation.index, selected_occurrence, side, output_dir, image_size))
        return SimpleNamespace(
            image_file=output_dir / "snapshot.png",
            renderer="matplotlib",
        )

    monkeypatch.setattr(pathway_graph_tab, "create_reaction_state_snapshot", fake_snapshot)
    task = PathwaySnapshotTask(
        cache_key=("formula", 1, "A", "reactants"),
        step=_pathway_step(),
        side="reactants",
        output_dir=tmp_path / "snapshot",
    )
    results = queue.Queue()

    render_pathway_snapshots_worker(
        results,
        7,
        project,
        "formula",
        [task],
        threading.Event(),
    )

    event, token, result = results.get_nowait()
    assert (event, token) == ("snapshot", 7)
    assert result.cache_key == task.cache_key
    assert result.renderer == "matplotlib"
    assert result.error is None
    assert rendered[0][2] == "products"
    assert results.get_nowait() == ("complete", 7, None)


def test_snapshot_worker_honors_cancellation_before_next_task(tmp_path: Path):
    """Stop queued renders promptly after the GUI invalidates a request."""

    cancel_event = threading.Event()
    cancel_event.set()
    results = queue.Queue()
    project = SimpleNamespace()
    task = PathwaySnapshotTask(
        cache_key=("formula", 1, "A", "reactants"),
        step=_pathway_step(),
        side="reactants",
        output_dir=tmp_path / "snapshot",
    )

    render_pathway_snapshots_worker(
        results,
        8,
        project,
        "formula",
        [task],
        cancel_event,
    )

    assert results.get_nowait() == ("complete", 8, None)
    assert results.empty()


def test_remove_pathway_snapshot_directory_removes_session_files(tmp_path: Path):
    """Delete generated graph snapshots when the application closes."""

    snapshot_dir = tmp_path / "lammpalyze_pathway_graph"
    snapshot_dir.mkdir()
    (snapshot_dir / "snapshot.png").write_bytes(b"image")

    remove_pathway_snapshot_directory(snapshot_dir)

    assert not snapshot_dir.exists()
