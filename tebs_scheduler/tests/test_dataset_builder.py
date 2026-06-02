from __future__ import annotations

from tebs.environment import generate_orbit_environment
from tebs.learning.dataset_builder import build_dataset_from_traces, build_feature_names
from tebs.models import Core, ScheduleDecision, ScheduleTrace, SystemState, Task, TaskBlock


def _cores() -> tuple[Core, ...]:
    return (
        Core(core_id="hp_0", core_type="high_performance"),
        Core(core_id="lp_0", core_type="low_power"),
    )


def _tasks() -> list[Task]:
    block0 = TaskBlock(
        task_id="T0",
        block_id=0,
        release_time=0,
        deadline=5,
        weight=2.0,
        duration_by_core={"hp_0": 1, "lp_0": 2},
        power_by_core={"hp_0": 4.0, "lp_0": 2.0},
    )
    block1 = TaskBlock(
        task_id="T0",
        block_id=1,
        release_time=0,
        deadline=5,
        weight=2.0,
        duration_by_core={"hp_0": 1, "lp_0": 2},
        power_by_core={"hp_0": 4.0, "lp_0": 2.0},
        predecessor_block_id=0,
    )
    return [
        Task(
            task_id="T0",
            release_time=0,
            deadline=5,
            weight=2.0,
            blocks=[block0, block1],
            core_ids=("hp_0", "lp_0"),
        ),
        Task.single_block(
            task_id="T1",
            release_time=2,
            deadline=7,
            weight=1.0,
            duration_by_core={"hp_0": 2, "lp_0": 3},
            power_by_core={"hp_0": 3.0, "lp_0": 1.5},
            core_ids=("hp_0", "lp_0"),
        ),
    ]


def test_dataset_builder_emits_fixed_windows_and_labels() -> None:
    cores = _cores()
    schedule_trace = (
        ScheduleTrace(
            time_slot=0,
            decisions=[
                ScheduleDecision.run_block(0, "hp_0", "T0", 0),
                ScheduleDecision.idle(0, "lp_0"),
            ],
        ),
        ScheduleTrace(
            time_slot=1,
            decisions=[
                ScheduleDecision.run_block(1, "hp_0", "T0", 1),
                ScheduleDecision.run_block(1, "lp_0", "T1", 0),
            ],
        ),
    )
    states = (
        SystemState(time_slot=0, energy_joule=100.0, temperature_celsius=25.0),
        SystemState(time_slot=1, energy_joule=98.0, temperature_celsius=25.1),
    )
    environment = generate_orbit_environment(
        simulation_slots=4,
        delta_t_seconds=1,
        sunlight_duration_slots=4,
        eclipse_duration_slots=0,
        max_solar_power_w=10.0,
        base_power_w=1.0,
        solar_power_profile="constant",
    )

    dataset = build_dataset_from_traces(
        tasks=_tasks(),
        cores=cores,
        schedule_trace=schedule_trace,
        state_trace=states,
        environment_slots=environment.slots,
        horizon_slots=2,
        max_tasks=2,
    )

    assert len(dataset.samples) == 2
    assert dataset.feature_spec.horizon_slots == 2
    assert len(dataset.samples[0].feature_window) == 2
    assert len(dataset.samples[0].feature_window[0]) == dataset.feature_dim
    assert dataset.samples[0].labels[0].task_id == "T0"
    assert dataset.samples[0].labels[0].action_index == 1
    assert dataset.samples[0].labels[1].is_idle is True
    assert dataset.samples[1].labels[0].block_id == 1
    assert dataset.as_xy()[1][0] == (1, 0)


def test_feature_name_contract_matches_task_and_core_counts() -> None:
    names = build_feature_names(max_tasks=3, max_blocks_per_task=2, cores=_cores())

    assert "energy_joule" in names
    assert "core_hp_0_is_hp" in names
    assert "task_2_deadline_slack" in names
    assert len(names) == 9 + 4 * 2 + 7 * 3
