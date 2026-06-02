"""Lightweight visualization helpers for TEBS experiment outputs."""

from __future__ import annotations

import hashlib
import math
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .environment import EnvironmentSlot
from .models import ScheduleDecision, ScheduleTrace, SolverTrace, SystemState, Task
from .solver_monitor import solver_summary_dict


class VisualizationError(ValueError):
    """Raised when a figure cannot be generated from the provided data."""


@dataclass(frozen=True, slots=True)
class FigureOutputs:
    """Paths emitted by generate_experiment_figures()."""

    gantt_chart: Path
    energy_temperature_curve: Path
    power_curve: Path
    solver_statistics: Path | None = None
    metrics_bar_chart: Path | None = None

    def as_dict(self) -> dict[str, Path | None]:
        return {
            "gantt_chart": self.gantt_chart,
            "energy_temperature_curve": self.energy_temperature_curve,
            "power_curve": self.power_curve,
            "solver_statistics": self.solver_statistics,
            "metrics_bar_chart": self.metrics_bar_chart,
        }


def generate_experiment_figures(
    *,
    schedule_trace: Sequence[ScheduleTrace],
    state_trace: Sequence[SystemState],
    metrics_rows: Sequence[Mapping[str, Any]] | None,
    solver_trace: Sequence[SolverTrace],
    output_dir: str | Path,
    method_name: str,
    tasks: Sequence[Task] | None = None,
    environment_slots: Sequence[EnvironmentSlot] | None = None,
) -> FigureOutputs:
    """Generate the standard figure set for one method run."""

    figure_dir = Path(output_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    suffix = _safe_filename(method_name)
    gantt_path = figure_dir / f"gantt_{suffix}.png"
    energy_path = figure_dir / f"energy_temperature_{suffix}.png"
    power_path = figure_dir / f"power_{suffix}.png"
    solver_path = figure_dir / f"solver_statistics_{suffix}.png" if solver_trace else None
    metrics_path = figure_dir / f"metrics_{suffix}.png" if metrics_rows else None

    render_gantt_chart(schedule_trace=schedule_trace, output_path=gantt_path)
    render_energy_temperature_curve(state_trace=state_trace, output_path=energy_path)
    render_power_curve(
        schedule_trace=schedule_trace,
        output_path=power_path,
        tasks=tasks,
        environment_slots=environment_slots,
    )
    if solver_path is not None:
        render_solver_statistics(solver_trace=solver_trace, output_path=solver_path)
    if metrics_path is not None:
        render_metrics_bar_chart(metrics_rows=metrics_rows or (), output_path=metrics_path)

    return FigureOutputs(
        gantt_chart=gantt_path,
        energy_temperature_curve=energy_path,
        power_curve=power_path,
        solver_statistics=solver_path,
        metrics_bar_chart=metrics_path,
    )


def render_gantt_chart(
    *,
    schedule_trace: Sequence[ScheduleTrace],
    output_path: str | Path,
    width: int = 960,
    row_height: int = 28,
) -> Path:
    """Render a slot-level Gantt chart as a PNG file."""

    traces = _require_schedule_trace(schedule_trace)
    core_ids = sorted({decision.core_id for trace in traces for decision in trace.decisions})
    if not core_ids:
        raise VisualizationError("schedule_trace must include at least one core decision.")

    left, top, right, bottom = 48, 24, 18, 28
    plot_width = max(1, width - left - right)
    height = top + bottom + row_height * len(core_ids)
    canvas = _Canvas(width, height)
    _draw_plot_background(canvas, left, top, plot_width, row_height * len(core_ids))

    slot_count = len(traces)
    slot_width = max(2, plot_width / max(1, slot_count))
    core_to_row = {core_id: index for index, core_id in enumerate(core_ids)}

    for index in range(slot_count + 1):
        x = int(left + index * slot_width)
        canvas.line(x, top, x, top + row_height * len(core_ids), _GRID)
    for row_index in range(len(core_ids) + 1):
        y = top + row_index * row_height
        canvas.line(left, y, left + plot_width, y, _GRID)

    for slot_index, trace in enumerate(traces):
        x0 = int(left + slot_index * slot_width)
        x1 = int(left + (slot_index + 1) * slot_width) - 1
        for decision in trace.decisions:
            row_index = core_to_row[decision.core_id]
            y0 = top + row_index * row_height + 4
            y1 = top + (row_index + 1) * row_height - 5
            if decision.is_idle:
                canvas.rect(x0 + 1, (y0 + y1) // 2, x1, (y0 + y1) // 2 + 1, _IDLE, fill=True)
                continue
            color = _task_color(decision)
            canvas.rect(x0 + 1, y0, max(x0 + 2, x1), y1, color, fill=True)
            canvas.rect(x0 + 1, y0, max(x0 + 2, x1), y1, _DARK, fill=False)

    return _write_png(output_path, canvas)


def render_energy_temperature_curve(
    *,
    state_trace: Sequence[SystemState],
    output_path: str | Path,
    width: int = 960,
    height: int = 420,
) -> Path:
    """Render battery energy in Wh and temperature in Celsius."""

    rows = state_trace_to_energy_temperature_rows(state_trace)
    canvas = _Canvas(width, height)
    panel_gap = 36
    left, right, top, bottom = 56, 24, 24, 28
    panel_height = (height - top - bottom - panel_gap) // 2
    plot_width = width - left - right

    energy_values = [row["energy_wh"] for row in rows]
    temp_values = [row["temperature_celsius"] for row in rows]
    _draw_line_panel(
        canvas=canvas,
        values=energy_values,
        left=left,
        top=top,
        width=plot_width,
        height=panel_height,
        color=_ENERGY,
    )
    _draw_line_panel(
        canvas=canvas,
        values=temp_values,
        left=left,
        top=top + panel_height + panel_gap,
        width=plot_width,
        height=panel_height,
        color=_THERMAL,
    )

    for row_index, row in enumerate(rows):
        x = _series_x(row_index, len(rows), left, plot_width)
        if row["energy_violation"]:
            canvas.line(x, top, x, top + panel_height, _VIOLATION)
        if row["thermal_violation"]:
            temp_top = top + panel_height + panel_gap
            canvas.line(x, temp_top, x, temp_top + panel_height, _VIOLATION)

    return _write_png(output_path, canvas)


def render_power_curve(
    *,
    schedule_trace: Sequence[ScheduleTrace],
    output_path: str | Path,
    tasks: Sequence[Task] | None = None,
    environment_slots: Sequence[EnvironmentSlot] | None = None,
    width: int = 960,
    height: int = 340,
) -> Path:
    """Render compute, base, total and harvested power curves."""

    rows = power_rows_from_schedule(
        schedule_trace=schedule_trace,
        tasks=tasks,
        environment_slots=environment_slots,
    )
    canvas = _Canvas(width, height)
    left, right, top, bottom = 56, 24, 24, 32
    plot_width = width - left - right
    plot_height = height - top - bottom
    all_values = []
    for key in ("compute_power_w", "base_power_w", "total_power_w", "harvested_power_w"):
        all_values.extend(row[key] for row in rows)
    min_value, max_value = _value_bounds(all_values)
    _draw_axes(canvas, left, top, plot_width, plot_height)
    for key, color in (
        ("total_power_w", _POWER_TOTAL),
        ("compute_power_w", _POWER_COMPUTE),
        ("base_power_w", _POWER_BASE),
        ("harvested_power_w", _POWER_HARVEST),
    ):
        points = [
            (
                _series_x(index, len(rows), left, plot_width),
                _series_y(row[key], min_value, max_value, top, plot_height),
            )
            for index, row in enumerate(rows)
        ]
        canvas.polyline(points, color)
    return _write_png(output_path, canvas)


def render_solver_statistics(
    *,
    solver_trace: Sequence[SolverTrace],
    output_path: str | Path,
    width: int = 720,
    height: int = 340,
) -> Path:
    """Render solver status ratios and solve-time statistics."""

    if not solver_trace:
        raise VisualizationError("solver_trace must not be empty.")
    summary = solver_summary_dict(solver_trace)
    ratio_keys = (
        "optimal_slot_ratio",
        "feasible_slot_ratio",
        "timeout_ratio",
        "infeasible_ratio",
    )
    values = [float(summary[key] or 0.0) for key in ratio_keys]
    canvas = _Canvas(width, height)
    left, right, top, bottom = 56, 28, 28, 38
    plot_width = width - left - right
    plot_height = height - top - bottom
    _draw_axes(canvas, left, top, plot_width, plot_height)
    _draw_bars(canvas, values, left, top, plot_width, plot_height, max_value=1.0)
    return _write_png(output_path, canvas)


def render_metrics_bar_chart(
    *,
    metrics_rows: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    metric_keys: Sequence[str] = (
        "average_response_time",
        "total_weighted_tardiness",
        "completed_task_count",
    ),
    width: int = 860,
    height: int = 340,
) -> Path:
    """Render selected summary metrics as grouped bars."""

    if not metrics_rows:
        raise VisualizationError("metrics_rows must not be empty.")
    numeric_values: list[float] = []
    for row in metrics_rows:
        for key in metric_keys:
            value = row.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric_values.append(float(value))
    if not numeric_values:
        raise VisualizationError("metrics_rows must contain at least one numeric metric.")

    max_value = max(1.0, max(numeric_values))
    canvas = _Canvas(width, height)
    left, right, top, bottom = 56, 28, 28, 38
    plot_width = width - left - right
    plot_height = height - top - bottom
    _draw_axes(canvas, left, top, plot_width, plot_height)

    bar_values = []
    for row in metrics_rows:
        for key in metric_keys:
            value = row.get(key)
            bar_values.append(float(value) if isinstance(value, (int, float)) else 0.0)
    _draw_bars(canvas, bar_values, left, top, plot_width, plot_height, max_value=max_value)
    return _write_png(output_path, canvas)


def state_trace_to_energy_temperature_rows(
    state_trace: Sequence[SystemState],
) -> list[dict[str, float | int | bool]]:
    """Return table rows preserving energy in J and adding Wh for plotting."""

    if not state_trace:
        raise VisualizationError("state_trace must not be empty.")
    rows: list[dict[str, float | int | bool]] = []
    for state in state_trace:
        rows.append(
            {
                "time_slot": state.time_slot,
                "energy_j": float(state.energy_joule),
                "energy_wh": float(state.energy_joule) / 3600.0,
                "temperature_celsius": float(state.temperature_celsius),
                "energy_violation": bool(state.energy_violation),
                "thermal_violation": bool(state.thermal_violation),
            }
        )
    return rows


def power_rows_from_schedule(
    *,
    schedule_trace: Sequence[ScheduleTrace],
    tasks: Sequence[Task] | None = None,
    environment_slots: Sequence[EnvironmentSlot] | None = None,
) -> list[dict[str, float | int]]:
    """Build per-slot power rows from decisions, tasks and optional environment data."""

    traces = _require_schedule_trace(schedule_trace)
    task_by_id = {task.task_id: task for task in tasks or ()}
    env_by_time = {slot.time_slot: slot for slot in environment_slots or ()}
    rows: list[dict[str, float | int]] = []
    for trace in traces:
        compute_power = 0.0
        active_decision_count = 0
        for decision in trace.decisions:
            if decision.is_idle:
                continue
            active_decision_count += 1
            compute_power += _decision_power_w(decision, task_by_id)
        if not task_by_id and active_decision_count:
            compute_power = float(active_decision_count)

        env_slot = env_by_time.get(trace.time_slot)
        base_power = float(env_slot.base_power_w) if env_slot is not None else 0.0
        harvested_power = float(env_slot.harvested_power_w) if env_slot is not None else 0.0
        rows.append(
            {
                "time_slot": trace.time_slot,
                "compute_power_w": compute_power,
                "base_power_w": base_power,
                "total_power_w": base_power + compute_power,
                "harvested_power_w": harvested_power,
            }
        )
    return rows


def _decision_power_w(decision: ScheduleDecision, task_by_id: Mapping[str, Task]) -> float:
    if not task_by_id:
        return 0.0
    task = task_by_id.get(decision.task_id or "")
    if task is None:
        raise VisualizationError(f"Unknown task_id in schedule trace: {decision.task_id}")
    block = _resolve_decision_block(task, decision.block_id)
    if decision.core_id not in block.power_by_core:
        raise VisualizationError(
            f"Core {decision.core_id} cannot execute task {task.task_id} block {block.block_id}."
        )
    power = float(block.power_by_core[decision.core_id])
    if decision.frequency_level:
        power *= _DVFS_POWER_SCALE_BY_LEVEL.get(decision.frequency_level, 1.0)
    return power


def _resolve_decision_block(task: Task, block_id: int | None):
    if block_id is not None:
        for block in task.blocks:
            if block.block_id == block_id:
                return block
        raise VisualizationError(f"Unknown block_id {block_id} for task_id {task.task_id}.")
    return sorted(task.blocks, key=lambda item: item.block_id)[0]


def _require_schedule_trace(schedule_trace: Sequence[ScheduleTrace]) -> tuple[ScheduleTrace, ...]:
    traces = tuple(schedule_trace)
    if not traces:
        raise VisualizationError("schedule_trace must not be empty.")
    return traces


def _safe_filename(value: str) -> str:
    safe = "".join(char.lower() if char.isalnum() else "_" for char in value.strip())
    safe = "_".join(part for part in safe.split("_") if part)
    return safe or "figure"


def _task_color(decision: ScheduleDecision) -> tuple[int, int, int]:
    key = f"{decision.task_id}:{decision.block_id}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return (60 + digest[0] % 150, 60 + digest[1] % 150, 60 + digest[2] % 150)


def _draw_plot_background(
    canvas: "_Canvas",
    left: int,
    top: int,
    width: int,
    height: int,
) -> None:
    canvas.rect(left, top, left + width, top + height, _PANEL, fill=True)
    canvas.rect(left, top, left + width, top + height, _AXIS, fill=False)


def _draw_line_panel(
    *,
    canvas: "_Canvas",
    values: Sequence[float],
    left: int,
    top: int,
    width: int,
    height: int,
    color: tuple[int, int, int],
) -> None:
    min_value, max_value = _value_bounds(values)
    _draw_axes(canvas, left, top, width, height)
    points = [
        (
            _series_x(index, len(values), left, width),
            _series_y(value, min_value, max_value, top, height),
        )
        for index, value in enumerate(values)
    ]
    canvas.polyline(points, color)


def _draw_axes(canvas: "_Canvas", left: int, top: int, width: int, height: int) -> None:
    canvas.rect(left, top, left + width, top + height, _PANEL, fill=True)
    canvas.line(left, top + height, left + width, top + height, _AXIS)
    canvas.line(left, top, left, top + height, _AXIS)
    for index in range(1, 4):
        y = top + int(height * index / 4)
        canvas.line(left, y, left + width, y, _GRID)


def _draw_bars(
    canvas: "_Canvas",
    values: Sequence[float],
    left: int,
    top: int,
    width: int,
    height: int,
    *,
    max_value: float,
) -> None:
    if not values:
        return
    max_value = max(max_value, 1e-9)
    step = width / len(values)
    palette = (_ENERGY, _THERMAL, _POWER_COMPUTE, _POWER_HARVEST, _POWER_TOTAL)
    for index, value in enumerate(values):
        bar_width = max(2, int(step * 0.62))
        x0 = int(left + index * step + (step - bar_width) / 2)
        x1 = x0 + bar_width
        y1 = top + height
        y0 = int(y1 - max(0.0, value) / max_value * height)
        color = palette[index % len(palette)]
        canvas.rect(x0, y0, x1, y1, color, fill=True)
        canvas.rect(x0, y0, x1, y1, _DARK, fill=False)


def _series_x(index: int, count: int, left: int, width: int) -> int:
    if count <= 1:
        return left + width // 2
    return int(left + index * width / (count - 1))


def _series_y(value: float, min_value: float, max_value: float, top: int, height: int) -> int:
    ratio = (value - min_value) / (max_value - min_value)
    ratio = max(0.0, min(1.0, ratio))
    return int(top + height - ratio * height)


def _value_bounds(values: Sequence[float]) -> tuple[float, float]:
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    if not finite_values:
        raise VisualizationError("figure data must contain at least one finite value.")
    min_value = min(finite_values)
    max_value = max(finite_values)
    if min_value == max_value:
        padding = max(1.0, abs(min_value) * 0.05)
        return min_value - padding, max_value + padding
    padding = (max_value - min_value) * 0.08
    return min_value - padding, max_value + padding


class _Canvas:
    def __init__(
        self,
        width: int,
        height: int,
        background: tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        if width <= 0 or height <= 0:
            raise VisualizationError("figure width and height must be positive.")
        self.width = int(width)
        self.height = int(height)
        self.pixels = bytearray(background * (self.width * self.height))

    def set_pixel(self, x: int, y: int, color: tuple[int, int, int]) -> None:
        if not (0 <= x < self.width and 0 <= y < self.height):
            return
        offset = (y * self.width + x) * 3
        self.pixels[offset : offset + 3] = bytes(color)

    def rect(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        color: tuple[int, int, int],
        *,
        fill: bool,
    ) -> None:
        x0, x1 = sorted((max(0, x0), min(self.width - 1, x1)))
        y0, y1 = sorted((max(0, y0), min(self.height - 1, y1)))
        if fill:
            for y in range(y0, y1 + 1):
                start = (y * self.width + x0) * 3
                end = (y * self.width + x1 + 1) * 3
                self.pixels[start:end] = bytes(color) * (x1 - x0 + 1)
            return
        self.line(x0, y0, x1, y0, color)
        self.line(x1, y0, x1, y1, color)
        self.line(x1, y1, x0, y1, color)
        self.line(x0, y1, x0, y0, color)

    def line(self, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        error = dx + dy
        while True:
            self.set_pixel(x0, y0, color)
            if x0 == x1 and y0 == y1:
                break
            twice_error = 2 * error
            if twice_error >= dy:
                error += dy
                x0 += sx
            if twice_error <= dx:
                error += dx
                y0 += sy

    def polyline(self, points: Sequence[tuple[int, int]], color: tuple[int, int, int]) -> None:
        if len(points) == 1:
            x, y = points[0]
            self.rect(x - 1, y - 1, x + 1, y + 1, color, fill=True)
            return
        for start, end in zip(points, points[1:]):
            self.line(start[0], start[1], end[0], end[1], color)


def _write_png(output_path: str | Path, canvas: _Canvas) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stride = canvas.width * 3
    raw = bytearray()
    for y in range(canvas.height):
        raw.append(0)
        row_start = y * stride
        raw.extend(canvas.pixels[row_start : row_start + stride])
    compressed = zlib.compress(bytes(raw), level=9)
    png = b"".join(
        (
            _PNG_SIGNATURE,
            _png_chunk(
                b"IHDR",
                struct.pack(">IIBBBBB", canvas.width, canvas.height, 8, 2, 0, 0, 0),
            ),
            _png_chunk(b"IDAT", compressed),
            _png_chunk(b"IEND", b""),
        )
    )
    path.write_bytes(png)
    if path.stat().st_size == 0:
        raise VisualizationError(f"Failed to write non-empty PNG: {path}")
    return path


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + chunk_type
        + payload
        + struct.pack(">I", zlib.crc32(chunk_type + payload) & 0xFFFFFFFF)
    )


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_DVFS_POWER_SCALE_BY_LEVEL = {
    "high": 1.0,
    "medium": 0.85 * 0.85 * 0.7,
    "low": 0.75 * 0.75 * 0.5,
}
_PANEL = (248, 250, 252)
_GRID = (226, 232, 240)
_AXIS = (71, 85, 105)
_DARK = (30, 41, 59)
_IDLE = (203, 213, 225)
_ENERGY = (37, 99, 235)
_THERMAL = (220, 38, 38)
_POWER_TOTAL = (15, 118, 110)
_POWER_COMPUTE = (202, 138, 4)
_POWER_BASE = (100, 116, 139)
_POWER_HARVEST = (22, 163, 74)
_VIOLATION = (190, 18, 60)


__all__ = [
    "FigureOutputs",
    "VisualizationError",
    "generate_experiment_figures",
    "power_rows_from_schedule",
    "render_energy_temperature_curve",
    "render_gantt_chart",
    "render_metrics_bar_chart",
    "render_power_curve",
    "render_solver_statistics",
    "state_trace_to_energy_temperature_rows",
]
