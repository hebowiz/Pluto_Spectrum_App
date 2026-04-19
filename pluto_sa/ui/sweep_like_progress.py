"""Sweep-like progress helper functions.

This module keeps Sweep SA / Time Analyzer progress control logic in one place,
while allowing `main_window.py` to remain the UI orchestrator.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui

from pluto_sa.modes.analyzer_mode import AnalyzerMode


def current_sweep_state(progress_state: Any) -> str:
    """Return current sweep-like run state."""
    return progress_state.sweep_state


def prepare_sweep_like_continuous_entry_state(
    progress_state: Any,
    *,
    running_state: str,
) -> None:
    """Prepare shared progress state for Continuous entry."""
    progress_state.sweep_state = running_state


def prepare_sweep_like_single_entry_state(
    progress_state: Any,
    *,
    single_state: str,
) -> None:
    """Prepare shared progress state for Single entry."""
    progress_state.sweep_state = single_state


def restore_mode_state_after_reset(
    *,
    analyzer_mode: AnalyzerMode,
    previous_state: str,
    progress_state: Any,
    running_state: str,
    single_state: str,
    stopped_state: str,
    restore_sweep_state: Callable[[str], None],
    start_time_analyzer_continuous: Callable[[], None],
    stop_timer: Callable[[], None],
    update_continuous_button: Callable[[], None],
    is_time_analyzer_mode: bool,
) -> None:
    """Restore mode-specific progress state after reset."""
    if analyzer_mode in (AnalyzerMode.SWEEP_SA, AnalyzerMode.WIDEBAND_REALTIME_SA):
        if analyzer_mode == AnalyzerMode.SWEEP_SA:
            progress_state.suppress_progress_until_first_complete = True
        restore_sweep_state(previous_state)
        return

    if is_time_analyzer_mode:
        progress_state.suppress_progress_until_first_complete = True
        if previous_state == running_state:
            start_time_analyzer_continuous()
        elif previous_state == single_state:
            progress_state.sweep_state = stopped_state
            stop_timer()
            update_continuous_button()


def update_sweep_like_write_index(
    current_index: int,
    point_count: int,
    *,
    is_single: bool,
) -> tuple[int, bool]:
    """Advance sweep-like cursor and report right-edge reach."""
    if point_count <= 0:
        return 0, True
    reached_right_edge = current_index >= (point_count - 1)
    if is_single:
        next_index = current_index if reached_right_edge else current_index + 1
    else:
        next_index = 0 if reached_right_edge else current_index + 1
    return int(next_index), bool(reached_right_edge)


def finish_single_sweep_like(
    *,
    progress_state: Any,
    stopped_state: str,
    stop_timer: Callable[[], None],
    update_continuous_button: Callable[[], None],
) -> None:
    """Finish Single run and normalize state to STOPPED."""
    stop_timer()
    progress_state.sweep_state = stopped_state
    update_continuous_button()


def set_sweep_like_progress_symbol(
    *,
    item: pg.ScatterPlotItem | None,
    point_index: int | None,
    freq_axis: np.ndarray | None,
    active_trace_state_getter: Callable[[], Any],
) -> None:
    """Place/hide progress symbol on active live trace."""
    if item is None or freq_axis is None or point_index is None:
        if item is not None:
            item.setData([], [])
            item.setVisible(False)
        return

    trace_state = active_trace_state_getter()
    if trace_state is None or trace_state.display_db is None:
        item.setData([], [])
        item.setVisible(False)
        return

    point_index = int(point_index)
    if point_index < 0 or point_index >= len(freq_axis):
        item.setData([], [])
        item.setVisible(False)
        return
    if point_index >= len(trace_state.display_db):
        item.setData([], [])
        item.setVisible(False)
        return

    marker_y = float(trace_state.display_db[point_index])
    if not np.isfinite(marker_y):
        item.setData([], [])
        item.setVisible(False)
        return

    color = QtGui.QColor(trace_state.color_hex)
    item.setPen(pg.mkPen(color))
    item.setBrush(pg.mkBrush(color))
    item.setData([freq_axis[point_index]], [marker_y])
    item.setVisible(True)

