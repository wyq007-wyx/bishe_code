"""Metrics for TEBS schedule traces and simulation outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .models import ScheduleTrace, SolverTrace, Task
from .simulator import TaskCompletionRecord


class MetricsError(ValueError):
    """Raised when metrics inputs are invalid."""


@dataclass(frozen=True, slots=True)
class TaskMetric:
    """Per-task scheduling metric row."""

    task_id: str
    release_time: int
    deadline: int
    weight: float
    completed: bool
    completion_time: int | None
    response_time: int | None
    tardiness: int | None
    weighted_tardiness: float | None


def compute_metrics(
    *,
    tasks: Sequence[Task],
    schedule_trace: Sequence[ScheduleTrace],
    state_trace: Sequence[Any],
    solver_trace: Sequence[SolverTrace] = (),
    completion_records: Sequence[TaskCompletionRecord] | None = None,
) -> dict[str, float | int | None]:
    """Return summary metrics for one simulation run."""

    records = tuple(completion_records or infer_task_completion_records(tasks, schedule_trace))
    task_metrics = compute_task_metrics(tasks=tasks, completion_records=records)
    completed_metrics = [metric for metric in task_metrics if metric.completed]
    response_values = [metric.response_time for metric in completed_metrics if metric.response_time is not None]
    tardiness_values = [metric.tardiness for metric in completed_metrics if metric.tardiness is not None]
    weighted_tardiness_values = [
        metric.weighted_tardiness
        for metric in completed_metrics
        if metric.weighted_tardiness is not None
    ]
    energy_values = [float(state.energy_joule) for state in state_trace]
    temp_values = [float(state.temperature_celsius) for state in state_trace]
    solver_summary = compute_solver_metrics(solver_trace)

    summary: dict[str, float | int | None] = {
        "task_count": len(tasks),
        "completed_task_count": len(completed_metrics),
        "missed_task_count": len(tasks) - len(completed_metrics),
        "throughput_tasks_per_slot": (
            len(completed_metrics) / len(schedule_trace) if schedule_trace else 0.0
        ),
        "average_response_time": _mean(response_values),
        "average_tardiness": _mean(tardiness_values),
        "total_tardiness": sum(tardiness_values),
        "total_weighted_tardiness": sum(weighted_tardiness_values),
        "energy_violation_count": sum(1 for state in state_trace if state.energy_violation),
        "thermal_violation_count": sum(1 for state in state_trace if state.thermal_violation),
        "min_energy_j": min(energy_values) if energy_values else None,
        "avg_energy_j": _mean(energy_values),
        "max_temperature_celsius": max(temp_values) if temp_values else None,
        "avg_temperature_celsius": _mean(temp_values),
    }
    summary.update(solver_summary)
    return summary


def compute_task_metrics(
    *,
    tasks: Sequence[Task],
    completion_records: Sequence[TaskCompletionRecord],
) -> tuple[TaskMetric, ...]:
    """Return per-task response time and tardiness rows."""

    record_by_task = {record.task_id: record for record in completion_records}
    rows: list[TaskMetric] = []
    for task in tasks:
        record = record_by_task.get(task.task_id)
        if record is None:
            rows.append(
                TaskMetric(
                    task_id=task.task_id,
                    release_time=task.release_time,
                    deadline=task.deadline,
                    weight=task.weight,
                    completed=False,
                    completion_time=None,
                    response_time=None,
                    tardiness=None,
                    weighted_tardiness=None,
                )
            )
            continue
        response_time = record.completion_time - task.release_time
        tardiness = max(0, record.completion_time - task.deadline)
        rows.append(
            TaskMetric(
                task_id=task.task_id,
                release_time=task.release_time,
                deadline=task.deadline,
                weight=task.weight,
                completed=True,
                completion_time=record.completion_time,
                response_time=response_time,
                tardiness=tardiness,
                weighted_tardiness=task.weight * tardiness,
            )
        )
    return tuple(rows)


def task_metrics_rows(task_metrics: Sequence[TaskMetric]) -> list[dict[str, Any]]:
    """Convert task metric dataclasses into table-ready dictionaries."""

    return [
        {
            "task_id": metric.task_id,
            "release_time": metric.release_time,
            "deadline": metric.deadline,
            "weight": metric.weight,
            "completed": metric.completed,
            "completion_time": metric.completion_time,
            "response_time": metric.response_time,
            "tardiness": metric.tardiness,
            "weighted_tardiness": metric.weighted_tardiness,
        }
        for metric in task_metrics
    ]


def metrics_dataframe(task_metrics: Sequence[TaskMetric]) -> Any:
    """Return a pandas DataFrame when pandas is installed."""

    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise MetricsError("pandas is required for metrics_dataframe().") from exc
    return pd.DataFrame(task_metrics_rows(task_metrics))


def compute_solver_metrics(
    solver_trace: Sequence[SolverTrace],
) -> dict[str, float | int | None]:
    """Summarize solver status, feasibility and solve-time statistics."""

    total = len(solver_trace)
    if total == 0:
        return {
            "solver_trace_count": 0,
            "optimal_slot_ratio": 0.0,
            "feasible_slot_ratio": 0.0,
            "timeout_ratio": 0.0,
            "infeasible_ratio": 0.0,
            "avg_solve_time_sec": None,
            "p50_solve_time_sec": None,
            "p95_solve_time_sec": None,
            "avg_mip_gap": None,
            "p95_mip_gap": None,
            "avg_node_count": None,
            "avg_objective_value": None,
        }

    statuses = [trace.status.upper() for trace in solver_trace]
    solve_times = [trace.solve_time_sec for trace in solver_trace]
    mip_gaps = [trace.mip_gap for trace in solver_trace if trace.mip_gap is not None]
    objectives = [
        trace.objective_value for trace in solver_trace if trace.objective_value is not None
    ]
    node_counts = [trace.node_count for trace in solver_trace if trace.node_count is not None]
    feasible_count = sum(
        1
        for trace, status in zip(solver_trace, statuses)
        if trace.has_feasible_solution or status == "OPTIMAL"
    )
    return {
        "solver_trace_count": total,
        "optimal_slot_ratio": statuses.count("OPTIMAL") / total,
        "feasible_slot_ratio": feasible_count / total,
        "timeout_ratio": statuses.count("TIME_LIMIT") / total,
        "infeasible_ratio": statuses.count("INFEASIBLE") / total,
        "avg_solve_time_sec": _mean(solve_times),
        "p50_solve_time_sec": _percentile(solve_times, 50),
        "p95_solve_time_sec": _percentile(solve_times, 95),
        "avg_mip_gap": _mean(mip_gaps),
        "p95_mip_gap": _percentile(mip_gaps, 95),
        "avg_node_count": _mean(node_counts),
        "avg_objective_value": _mean(objectives),
    }


def infer_task_completion_records(
    tasks: Sequence[Task],
    schedule_trace: Sequence[ScheduleTrace],
) -> tuple[TaskCompletionRecord, ...]:
    """Infer task completion times by replaying schedule decisions."""

    task_by_id = {task.task_id: task for task in tasks}
    progress: dict[tuple[str, int], int] = {}
    completed_blocks: set[tuple[str, int]] = set()
    completed_tasks: set[str] = set()
    records: list[TaskCompletionRecord] = []

    for trace in schedule_trace:
        slot_completed: list[tuple[str, int]] = []
        for decision in trace.decisions:
            if decision.is_idle:
                continue
            task = task_by_id.get(decision.task_id or "")
            if task is None:
                raise MetricsError(f"Unknown task_id in schedule trace: {decision.task_id}")
            block = _resolve_metric_block(task, decision.block_id, completed_blocks)
            block_key = (task.task_id, block.block_id)
            if block_key in completed_blocks:
                continue
            if decision.core_id not in block.duration_by_core:
                raise MetricsError(
                    f"Core {decision.core_id} cannot execute task {task.task_id} block {block.block_id}."
                )
            if block_key not in progress:
                progress[block_key] = block.duration_by_core[decision.core_id]
            progress[block_key] -= 1
            if progress[block_key] <= 0:
                slot_completed.append(block_key)
        for block_key in slot_completed:
            completed_blocks.add(block_key)
            task = task_by_id[block_key[0]]
            if task.task_id not in completed_tasks and all(
                (task.task_id, block.block_id) in completed_blocks for block in task.blocks
            ):
                completed_tasks.add(task.task_id)
                records.append(
                    TaskCompletionRecord(
                        task_id=task.task_id,
                        release_time=task.release_time,
                        deadline=task.deadline,
                        weight=task.weight,
                        completion_time=trace.time_slot + 1,
                        completed_block_ids=tuple(block.block_id for block in task.blocks),
                    )
                )
    return tuple(records)


def _resolve_metric_block(task: Task, block_id: int | None, completed_blocks: set[tuple[str, int]]):
    if block_id is not None:
        for block in task.blocks:
            if block.block_id == block_id:
                return block
        raise MetricsError(f"Unknown block_id {block_id} for task_id {task.task_id}.")
    for block in sorted(task.blocks, key=lambda item: item.block_id):
        if (task.task_id, block.block_id) not in completed_blocks:
            return block
    return task.blocks[-1]


def _mean(values: Sequence[float | int]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _percentile(values: Sequence[float | int], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile / 100
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


__all__ = [
    "MetricsError",
    "TaskMetric",
    "compute_metrics",
    "compute_solver_metrics",
    "compute_task_metrics",
    "infer_task_completion_records",
    "metrics_dataframe",
    "task_metrics_rows",
]
