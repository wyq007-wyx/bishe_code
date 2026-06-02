from __future__ import annotations

from typing import Any, Sequence

from tebs.config import load_config_from_mapping
from tebs.environment import generate_orbit_environment
from tebs.learning.hybrid_scheduler import HybridGurobiTcnScheduler
from tebs.models import Core, ScheduleDecision, SolverTrace, SystemState, Task
from tebs.scheduler_base import BaseScheduler, SchedulerResult


def _config():
    return load_config_from_mapping(
        {
            "simulation": {
                "delta_t_seconds": 1,
                "horizon_slots": 2,
                "simulation_slots": 4,
                "high_performance_core_count": 1,
                "low_power_core_count": 0,
            },
            "energy": {
                "e_max_j": 100.0,
                "e_min_j": 0.0,
                "initial_energy_j": 50.0,
                "default_base_power_w": 1.0,
                "default_max_solar_power_w": 10.0,
            },
            "thermal": {
                "initial_temperature_celsius": 25.0,
                "t_max_celsius": 80.0,
                "ambient_temperature_celsius": 25.0,
                "thermal_capacity_j_per_c": 1000.0,
                "cooling_coeff": 0.0,
            },
        }
    )


def _core() -> Core:
    return Core(core_id="hp_0", core_type="high_performance")


def _task() -> Task:
    return Task.single_block(
        task_id="T0",
        release_time=0,
        deadline=5,
        weight=2.0,
        duration_by_core={"hp_0": 1},
        power_by_core={"hp_0": 2.0},
        core_ids=("hp_0",),
    )


def _environment(config):
    return generate_orbit_environment(
        simulation_slots=4,
        delta_t_seconds=config.simulation.delta_t_seconds,
        sunlight_duration_slots=4,
        eclipse_duration_slots=0,
        max_solar_power_w=10.0,
        base_power_w=1.0,
        solar_power_profile="constant",
        ambient_temperature_celsius=config.thermal.ambient_temperature_celsius,
    )


class FeasiblePrimary(BaseScheduler):
    def decide(
        self,
        *,
        current_time: int,
        tasks: Sequence[Task],
        cores: Sequence[Core],
        system_state: SystemState,
        environment_window: Sequence[Any],
    ) -> SchedulerResult:
        return SchedulerResult(
            [
                ScheduleDecision.run_block(
                    time_slot=current_time,
                    core_id=cores[0].core_id,
                    task_id=tasks[0].task_id,
                    block_id=0,
                )
            ],
            solver_trace=SolverTrace(
                time_slot=current_time,
                solver_name="gurobi",
                status="OPTIMAL",
                solve_time_sec=0.01,
                has_feasible_solution=True,
            ),
        )


class InfeasiblePrimary(BaseScheduler):
    def decide(
        self,
        *,
        current_time: int,
        tasks: Sequence[Task],
        cores: Sequence[Core],
        system_state: SystemState,
        environment_window: Sequence[Any],
    ) -> SchedulerResult:
        return SchedulerResult(
            [ScheduleDecision.idle(time_slot=current_time, core_id=cores[0].core_id)],
            solver_trace=SolverTrace(
                time_slot=current_time,
                solver_name="gurobi",
                status="INFEASIBLE",
                solve_time_sec=0.01,
                has_feasible_solution=False,
            ),
        )


def test_hybrid_scheduler_uses_primary_feasible_gurobi_result() -> None:
    config = _config()
    scheduler = HybridGurobiTcnScheduler(config=config, primary_scheduler=FeasiblePrimary())

    result = scheduler.decide(
        current_time=0,
        tasks=[_task()],
        cores=[_core()],
        system_state=SystemState(time_slot=0, energy_joule=50.0, temperature_celsius=25.0),
        environment_window=_environment(config).window(0, 2),
    )

    assert scheduler.last_fallback_used is False
    assert result.decisions[0].task_id == "T0"
    assert result.solver_trace is not None
    assert result.solver_trace.status == "OPTIMAL"


def test_hybrid_scheduler_falls_back_to_tcn_when_primary_has_no_feasible_solution() -> None:
    config = _config()
    scheduler = HybridGurobiTcnScheduler(config=config, primary_scheduler=InfeasiblePrimary())

    result = scheduler.decide(
        current_time=0,
        tasks=[_task()],
        cores=[_core()],
        system_state=SystemState(time_slot=0, energy_joule=50.0, temperature_celsius=25.0),
        environment_window=_environment(config).window(0, 2),
    )

    assert scheduler.last_fallback_used is True
    assert result.decisions[0].task_id == "T0"
    assert scheduler.last_filter_result is not None
    assert result.solver_trace is not None
    assert result.solver_trace.status == "INFEASIBLE"
