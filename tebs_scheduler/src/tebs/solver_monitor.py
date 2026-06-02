"""Solver-trace aggregation utilities for Gurobi-based schedulers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

from .models import SolverTrace


class SolverMonitorError(ValueError):
    """Raised when solver-monitor inputs are invalid."""


@dataclass(frozen=True, slots=True)
class SolverMonitorSummary:
    """Aggregated solver performance and quality statistics."""

    solver_trace_count: int
    optimal_slot_ratio: float
    feasible_slot_ratio: float
    timeout_ratio: float
    infeasible_ratio: float
    avg_solve_time_sec: float | None
    p50_solve_time_sec: float | None
    p95_solve_time_sec: float | None
    avg_mip_gap: float | None
    p95_mip_gap: float | None
    avg_node_count: float | None
    avg_objective_value: float | None

    def to_dict(self) -> dict[str, float | int | None]:
        """Return a table-ready dictionary."""

        return asdict(self)


def summarize_solver_trace(solver_trace: Sequence[SolverTrace]) -> SolverMonitorSummary:
    """Summarize solver status, feasibility, solve time and optimality gaps."""

    total = len(solver_trace)
    if total == 0:
        return SolverMonitorSummary(
            solver_trace_count=0,
            optimal_slot_ratio=0.0,
            feasible_slot_ratio=0.0,
            timeout_ratio=0.0,
            infeasible_ratio=0.0,
            avg_solve_time_sec=None,
            p50_solve_time_sec=None,
            p95_solve_time_sec=None,
            avg_mip_gap=None,
            p95_mip_gap=None,
            avg_node_count=None,
            avg_objective_value=None,
        )

    statuses = [trace.status.upper() for trace in solver_trace]
    solve_times = [trace.solve_time_sec for trace in solver_trace]
    mip_gaps = [trace.mip_gap for trace in solver_trace if trace.mip_gap is not None]
    node_counts = [trace.node_count for trace in solver_trace if trace.node_count is not None]
    objectives = [
        trace.objective_value for trace in solver_trace if trace.objective_value is not None
    ]
    feasible_count = sum(
        1
        for trace, status in zip(solver_trace, statuses)
        if trace.has_feasible_solution or status == "OPTIMAL"
    )
    return SolverMonitorSummary(
        solver_trace_count=total,
        optimal_slot_ratio=statuses.count("OPTIMAL") / total,
        feasible_slot_ratio=feasible_count / total,
        timeout_ratio=statuses.count("TIME_LIMIT") / total,
        infeasible_ratio=statuses.count("INFEASIBLE") / total,
        avg_solve_time_sec=_mean(solve_times),
        p50_solve_time_sec=_percentile(solve_times, 50),
        p95_solve_time_sec=_percentile(solve_times, 95),
        avg_mip_gap=_mean(mip_gaps),
        p95_mip_gap=_percentile(mip_gaps, 95),
        avg_node_count=_mean(node_counts),
        avg_objective_value=_mean(objectives),
    )


def solver_summary_dict(solver_trace: Sequence[SolverTrace]) -> dict[str, float | int | None]:
    """Return the solver monitor summary as a plain dictionary."""

    return summarize_solver_trace(solver_trace).to_dict()


def _mean(values: Sequence[float | int]) -> float | None:
    if not values:
        return None
    return sum(float(value) for value in values) / len(values)


def _percentile(values: Sequence[float | int], percentile: float) -> float | None:
    if not values:
        return None
    if not 0 <= percentile <= 100:
        raise SolverMonitorError("percentile must be within [0, 100].")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile / 100
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


__all__ = [
    "SolverMonitorError",
    "SolverMonitorSummary",
    "solver_summary_dict",
    "summarize_solver_trace",
]
