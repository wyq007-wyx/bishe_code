from __future__ import annotations

from tebs.config import load_config_from_mapping
from tebs.environment import generate_orbit_environment
from tebs.learning.constraint_filter import ConstraintFilter
from tebs.learning.tcn_policy import TcnActionProposal
from tebs.models import Core, SystemState, Task, TaskBlock


def _config(e_min_j: float = 0.0):
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
                "e_min_j": e_min_j,
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


def _two_block_task() -> Task:
    block0 = TaskBlock(
        task_id="T0",
        block_id=0,
        release_time=0,
        deadline=5,
        weight=1.0,
        duration_by_core={"hp_0": 1},
        power_by_core={"hp_0": 2.0},
    )
    block1 = TaskBlock(
        task_id="T0",
        block_id=1,
        release_time=0,
        deadline=5,
        weight=1.0,
        duration_by_core={"hp_0": 1},
        power_by_core={"hp_0": 2.0},
        predecessor_block_id=0,
    )
    return Task(
        task_id="T0",
        release_time=0,
        deadline=5,
        weight=1.0,
        blocks=[block0, block1],
        core_ids=("hp_0",),
    )


def _environment(config, max_solar_power_w: float = 10.0):
    return generate_orbit_environment(
        simulation_slots=4,
        delta_t_seconds=config.simulation.delta_t_seconds,
        sunlight_duration_slots=4,
        eclipse_duration_slots=0,
        max_solar_power_w=max_solar_power_w,
        base_power_w=1.0,
        solar_power_profile="constant",
        ambient_temperature_celsius=config.thermal.ambient_temperature_celsius,
    )


def test_constraint_filter_rejects_predecessor_violation_and_selects_ready_block() -> None:
    config = _config()
    result = ConstraintFilter(config=config).select_feasible_decisions(
        current_time=0,
        proposals=[
            TcnActionProposal(core_id="hp_0", task_id="T0", block_id=1, is_idle=False, score=10.0),
            TcnActionProposal(core_id="hp_0", task_id="T0", block_id=0, is_idle=False, score=5.0),
            TcnActionProposal(core_id="hp_0", is_idle=True, score=0.0),
        ],
        tasks=[_two_block_task()],
        cores=[_core()],
        system_state=SystemState(time_slot=0, energy_joule=50.0, temperature_celsius=25.0),
        environment_window=_environment(config).window(0, 2),
    )

    assert result.decisions[0].task_id == "T0"
    assert result.decisions[0].block_id == 0
    assert [item.reason for item in result.rejected_actions] == ["predecessor_incomplete"]
    assert result.retention_rate == 1.0


def test_constraint_filter_falls_back_to_idle_when_energy_is_infeasible() -> None:
    config = _config(e_min_j=49.0)
    result = ConstraintFilter(config=config).select_feasible_decisions(
        current_time=0,
        proposals=[
            TcnActionProposal(core_id="hp_0", task_id="T0", block_id=0, is_idle=False, score=1.0),
        ],
        tasks=[_two_block_task()],
        cores=[_core()],
        system_state=SystemState(time_slot=0, energy_joule=50.0, temperature_celsius=25.0),
        environment_window=_environment(config, max_solar_power_w=0.0).window(0, 2),
    )

    assert result.decisions[0].is_idle is True
    assert result.rejected_actions[0].reason == "energy_infeasible"
