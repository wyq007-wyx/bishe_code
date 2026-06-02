from __future__ import annotations

import pytest

from tebs.baseline_fcfs_gurobi import FcfsGurobiError, FcfsGurobiScheduler
from tebs.config import ConfigBundle, load_config_from_mapping
from tebs.environment import generate_orbit_environment
from tebs.gurobi_adapter import GurobiSolveParams
from tebs.models import Core, SystemState, Task


def _config() -> ConfigBundle:
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
                "e_min_j": 0.0,
                "initial_energy_j": 100.0,
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


def _task(task_id: str, release_time: int = 0) -> Task:
    return Task.single_block(
        task_id=task_id,
        release_time=release_time,
        deadline=10,
        weight=1.0,
        duration_by_core={"hp_0": 3, "lp_0": 1},
        power_by_core={"hp_0": 3.0, "lp_0": 1.0},
        core_ids=("hp_0", "lp_0"),
    )


def _environment(config: ConfigBundle):
    return generate_orbit_environment(
        simulation_slots=5,
        delta_t_seconds=config.simulation.delta_t_seconds,
        sunlight_duration_slots=5,
        eclipse_duration_slots=0,
        max_solar_power_w=20.0,
        base_power_w=1.0,
        solar_power_profile="constant",
        ambient_temperature_celsius=config.thermal.ambient_temperature_celsius,
    )


def _require_gurobi() -> None:
    pytest.importorskip("gurobipy")


def test_fcfs_gurobi_does_not_skip_queue_head_when_unreleased() -> None:
    scheduler = FcfsGurobiScheduler(config=_config())

    result = scheduler.decide(
        current_time=0,
        tasks=[_task("T0", release_time=2), _task("T1", release_time=0)],
        cores=_cores(),
        system_state=SystemState(time_slot=0, energy_joule=100.0, temperature_celsius=25.0),
        environment_window=[],
    )

    assert all(decision.is_idle for decision in result.decisions)


def test_fcfs_gurobi_solves_toy_case_and_keeps_fcfs_order() -> None:
    _require_gurobi()
    config = _config()
    scheduler = FcfsGurobiScheduler(
        config=config,
        gurobi_params=GurobiSolveParams(time_limit_sec=5.0, mip_gap=0.0),
        fallback_to_idle_on_error=False,
    )

    try:
        result = scheduler.decide(
            current_time=0,
            tasks=[_task("T0"), _task("T1")],
            cores=_cores(),
            system_state=SystemState(time_slot=0, energy_joule=100.0, temperature_celsius=25.0),
            environment_window=_environment(config).window(0, 3),
        )
    except FcfsGurobiError as exc:
        pytest.skip(f"Gurobi is installed but not usable in this environment: {exc}")

    running = [decision for decision in result.decisions if not decision.is_idle]
    assert len(running) == 1
    assert running[0].task_id == "T0"
    assert result.solver_trace is not None
    assert result.solver_trace.has_feasible_solution is True
    assert scheduler.last_window_plan[0].task_id == "T0"
