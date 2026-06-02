from __future__ import annotations

import pytest

from tebs.config import ConfigBundle, load_config_from_mapping
from tebs.environment import generate_orbit_environment
from tebs.gurobi_adapter import GurobiSolveParams
from tebs.models import Core, SystemState, Task, TaskBlock
from tebs.rhc_milp_gurobi import RhcMilpGurobiError, RhcMilpGurobiScheduler


def _config(e_min_j: float = 0.0) -> ConfigBundle:
    return load_config_from_mapping(
        {
            "simulation": {
                "delta_t_seconds": 1,
                "horizon_slots": 3,
                "simulation_slots": 5,
                "high_performance_core_count": 1,
                "low_power_core_count": 1,
            },
            "gurobi": {"gurobi_time_limit": 5.0, "gurobi_mip_gap": 0.0},
            "energy": {
                "e_max_j": 1000.0,
                "e_min_j": e_min_j,
                "initial_energy_j": max(100.0, e_min_j),
                "default_base_power_w": 1.0,
                "default_max_solar_power_w": 20.0,
            },
            "thermal": {
                "initial_temperature_celsius": 25.0,
                "t_max_celsius": 100.0,
                "ambient_temperature_celsius": 25.0,
                "thermal_capacity_j_per_c": 10000.0,
                "cooling_coeff": 0.0,
            },
        }
    )


def _cores() -> tuple[Core, ...]:
    return (
        Core(core_id="hp_0", core_type="high_performance"),
        Core(core_id="lp_0", core_type="low_power"),
    )


def _task(task_id: str, release_time: int = 0, weight: float = 1.0) -> Task:
    return Task.single_block(
        task_id=task_id,
        release_time=release_time,
        deadline=10,
        weight=weight,
        duration_by_core={"hp_0": 2, "lp_0": 1},
        power_by_core={"hp_0": 3.0, "lp_0": 1.0},
        core_ids=("hp_0", "lp_0"),
    )


def _two_block_task() -> Task:
    block0 = TaskBlock(
        task_id="T0",
        block_id=0,
        release_time=0,
        deadline=10,
        weight=1.0,
        duration_by_core={"hp_0": 1, "lp_0": 1},
        power_by_core={"hp_0": 2.0, "lp_0": 1.0},
    )
    block1 = TaskBlock(
        task_id="T0",
        block_id=1,
        release_time=0,
        deadline=10,
        weight=1.0,
        duration_by_core={"hp_0": 1, "lp_0": 1},
        power_by_core={"hp_0": 2.0, "lp_0": 1.0},
        predecessor_block_id=0,
    )
    return Task(
        task_id="T0",
        release_time=0,
        deadline=10,
        weight=1.0,
        blocks=[block0, block1],
        core_ids=("hp_0", "lp_0"),
    )


def _environment(config: ConfigBundle, max_solar_power_w: float = 20.0):
    return generate_orbit_environment(
        simulation_slots=5,
        delta_t_seconds=config.simulation.delta_t_seconds,
        sunlight_duration_slots=5,
        eclipse_duration_slots=0,
        max_solar_power_w=max_solar_power_w,
        base_power_w=1.0,
        solar_power_profile="constant",
        ambient_temperature_celsius=config.thermal.ambient_temperature_celsius,
    )


def _require_gurobi() -> None:
    pytest.importorskip("gurobipy")


def _scheduler(config: ConfigBundle) -> RhcMilpGurobiScheduler:
    return RhcMilpGurobiScheduler(
        config=config,
        gurobi_params=GurobiSolveParams(time_limit_sec=5.0, mip_gap=0.0),
        fallback_to_idle_on_error=False,
    )


def test_rhc_milp_solves_small_case_with_core_capacity() -> None:
    _require_gurobi()
    config = _config()
    scheduler = _scheduler(config)

    try:
        result = scheduler.decide(
            current_time=0,
            tasks=[_task("T0"), _task("T1", weight=2.0)],
            cores=_cores(),
            system_state=SystemState(time_slot=0, energy_joule=100.0, temperature_celsius=25.0),
            environment_window=_environment(config).window(0, 3),
        )
    except RhcMilpGurobiError as exc:
        pytest.skip(f"Gurobi is installed but not usable in this environment: {exc}")

    running = [decision for decision in result.decisions if not decision.is_idle]
    assert len({decision.core_id for decision in running}) == len(running)
    assert result.solver_trace is not None
    assert result.solver_trace.has_feasible_solution is True
    assert scheduler.last_window_plan


def test_rhc_milp_does_not_schedule_successor_before_predecessor() -> None:
    _require_gurobi()
    config = _config()
    scheduler = _scheduler(config)

    try:
        result = scheduler.decide(
            current_time=0,
            tasks=[_two_block_task()],
            cores=_cores(),
            system_state=SystemState(time_slot=0, energy_joule=100.0, temperature_celsius=25.0),
            environment_window=_environment(config).window(0, 3),
        )
    except RhcMilpGurobiError as exc:
        pytest.skip(f"Gurobi is installed but not usable in this environment: {exc}")

    assert all(assignment.block_id != 1 for assignment in scheduler.last_window_plan)
    assert all(decision.block_id != 1 for decision in result.decisions if not decision.is_idle)


def test_rhc_milp_respects_release_time_before_current_slot() -> None:
    _require_gurobi()
    config = _config()
    scheduler = _scheduler(config)

    try:
        result = scheduler.decide(
            current_time=0,
            tasks=[_task("T0", release_time=2)],
            cores=_cores(),
            system_state=SystemState(time_slot=0, energy_joule=100.0, temperature_celsius=25.0),
            environment_window=_environment(config).window(0, 3),
        )
    except RhcMilpGurobiError as exc:
        pytest.skip(f"Gurobi is installed but not usable in this environment: {exc}")

    assert all(decision.is_idle for decision in result.decisions)
    assert all(assignment.time_slot >= 2 for assignment in scheduler.last_window_plan)


def test_rhc_milp_idles_when_energy_constraint_is_infeasible() -> None:
    _require_gurobi()
    config = _config(e_min_j=99.0)
    scheduler = _scheduler(config)

    try:
        result = scheduler.decide(
            current_time=0,
            tasks=[_task("T0")],
            cores=_cores(),
            system_state=SystemState(time_slot=0, energy_joule=100.0, temperature_celsius=25.0),
            environment_window=_environment(config, max_solar_power_w=0.0).window(0, 3),
        )
    except RhcMilpGurobiError as exc:
        pytest.skip(f"Gurobi is installed but not usable in this environment: {exc}")

    assert all(decision.is_idle for decision in result.decisions)
    assert result.solver_trace is not None
    assert result.solver_trace.has_feasible_solution is False
