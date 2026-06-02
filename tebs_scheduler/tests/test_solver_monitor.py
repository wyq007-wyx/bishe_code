from __future__ import annotations

import pytest

from tebs.models import SolverTrace
from tebs.solver_monitor import solver_summary_dict, summarize_solver_trace


def _trace(
    time_slot: int,
    status: str,
    *,
    feasible: bool = False,
    mip_gap: float | None = None,
    node_count: float | None = None,
) -> SolverTrace:
    return SolverTrace(
        time_slot=time_slot,
        solver_name="gurobi",
        status=status,
        solve_time_sec=float(time_slot),
        mip_gap=mip_gap,
        objective_value=10.0 + time_slot if feasible or status == "OPTIMAL" else None,
        has_feasible_solution=feasible,
        node_count=node_count,
    )


def test_summarize_solver_trace_counts_optimal_ratio() -> None:
    traces = [
        *[_trace(i, "OPTIMAL", feasible=True, mip_gap=0.0, node_count=i) for i in range(6)],
        _trace(6, "TIME_LIMIT", feasible=True, mip_gap=0.05, node_count=12),
        _trace(7, "TIME_LIMIT", feasible=False),
        _trace(8, "INFEASIBLE"),
        _trace(9, "INFEASIBLE"),
    ]

    summary = summarize_solver_trace(traces)

    assert summary.solver_trace_count == 10
    assert summary.optimal_slot_ratio == pytest.approx(0.6)
    assert summary.timeout_ratio == pytest.approx(0.2)
    assert summary.infeasible_ratio == pytest.approx(0.2)


def test_time_limit_with_solution_counts_as_feasible_only_when_solution_exists() -> None:
    traces = [
        _trace(0, "TIME_LIMIT", feasible=True),
        _trace(1, "TIME_LIMIT", feasible=False),
    ]

    summary = summarize_solver_trace(traces)

    assert summary.feasible_slot_ratio == pytest.approx(0.5)


def test_percentiles_and_node_count_are_aggregated() -> None:
    traces = [_trace(i, "OPTIMAL", feasible=True, node_count=i * 2) for i in range(10)]

    summary = solver_summary_dict(traces)

    assert summary["p50_solve_time_sec"] == pytest.approx(4.5)
    assert summary["p95_solve_time_sec"] == pytest.approx(8.55)
    assert summary["avg_node_count"] == pytest.approx(9.0)


def test_empty_solver_trace_returns_clear_defaults() -> None:
    summary = summarize_solver_trace([])

    assert summary.solver_trace_count == 0
    assert summary.optimal_slot_ratio == 0.0
    assert summary.feasible_slot_ratio == 0.0
    assert summary.avg_solve_time_sec is None
