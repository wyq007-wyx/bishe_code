from __future__ import annotations

from tebs.metrics import compute_metrics, compute_solver_metrics, compute_task_metrics
from tebs.models import SolverTrace, SystemState, Task
from tebs.simulator import TaskCompletionRecord


def _task(task_id: str, release: int, deadline: int, weight: float) -> Task:
    return Task.single_block(
        task_id=task_id,
        release_time=release,
        deadline=deadline,
        weight=weight,
        duration_by_core={"hp_0": 1},
        power_by_core={"hp_0": 5.0},
        core_ids=("hp_0",),
    )


def test_task_metrics_compute_response_and_weighted_tardiness() -> None:
    tasks = [_task("T1", 2, 5, 3.0)]
    records = [
        TaskCompletionRecord(
            task_id="T1",
            release_time=2,
            deadline=5,
            weight=3.0,
            completion_time=8,
            completed_block_ids=(0,),
        )
    ]

    row = compute_task_metrics(tasks=tasks, completion_records=records)[0]

    assert row.response_time == 6
    assert row.tardiness == 3
    assert row.weighted_tardiness == 9.0


def test_summary_metrics_count_safety_violations_and_throughput() -> None:
    tasks = [_task("T1", 0, 2, 1.0), _task("T2", 0, 4, 2.0)]
    records = [
        TaskCompletionRecord(
            task_id="T1",
            release_time=0,
            deadline=2,
            weight=1.0,
            completion_time=2,
            completed_block_ids=(0,),
        )
    ]
    states = [
        SystemState(time_slot=0, energy_joule=50.0, temperature_celsius=30.0),
        SystemState(
            time_slot=1,
            energy_joule=20.0,
            temperature_celsius=75.0,
            energy_violation=True,
            thermal_violation=True,
        ),
    ]

    summary = compute_metrics(
        tasks=tasks,
        schedule_trace=[],
        state_trace=states,
        completion_records=records,
    )

    assert summary["completed_task_count"] == 1
    assert summary["missed_task_count"] == 1
    assert summary["energy_violation_count"] == 1
    assert summary["thermal_violation_count"] == 1
    assert summary["max_temperature_celsius"] == 75.0
    assert summary["min_energy_j"] == 20.0


def test_solver_metrics_count_feasible_time_limit_separately() -> None:
    traces = [
        SolverTrace(
            time_slot=0,
            solver_name="gurobi",
            status="OPTIMAL",
            solve_time_sec=1.0,
            mip_gap=0.0,
            objective_value=10.0,
            has_feasible_solution=True,
        ),
        SolverTrace(
            time_slot=1,
            solver_name="gurobi",
            status="TIME_LIMIT",
            solve_time_sec=3.0,
            mip_gap=0.05,
            objective_value=12.0,
            has_feasible_solution=True,
        ),
        SolverTrace(
            time_slot=2,
            solver_name="gurobi",
            status="INFEASIBLE",
            solve_time_sec=2.0,
            has_feasible_solution=False,
        ),
    ]

    summary = compute_solver_metrics(traces)

    assert summary["optimal_slot_ratio"] == 1 / 3
    assert summary["feasible_slot_ratio"] == 2 / 3
    assert summary["timeout_ratio"] == 1 / 3
    assert summary["infeasible_ratio"] == 1 / 3
    assert summary["avg_solve_time_sec"] == 2.0
