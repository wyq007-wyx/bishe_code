from __future__ import annotations

from tebs.baseline_intermittent import (
    FcfsIntermittentScheduler,
    IntermittentThresholds,
    should_pause,
    should_resume,
)
from tebs.models import Core, SystemState, Task


def _core() -> Core:
    return Core(core_id="hp_0", core_type="high_performance")


def _task(task_id: str, duration: int = 3, release_time: int = 0) -> Task:
    return Task.single_block(
        task_id=task_id,
        release_time=release_time,
        deadline=20,
        weight=1.0,
        duration_by_core={"hp_0": duration},
        power_by_core={"hp_0": 10.0},
        core_ids=("hp_0",),
    )


def _thresholds() -> IntermittentThresholds:
    return IntermittentThresholds(
        pause_temperature_celsius=60.0,
        resume_temperature_celsius=50.0,
        pause_energy_j=40.0,
        resume_energy_j=80.0,
    )


def test_pause_threshold_makes_all_cores_idle() -> None:
    scheduler = FcfsIntermittentScheduler(thresholds=_thresholds())
    result = scheduler.decide(
        current_time=0,
        tasks=[_task("T0")],
        cores=[_core()],
        system_state=SystemState(time_slot=0, energy_joule=100.0, temperature_celsius=65.0),
        environment_window=[],
    )

    assert should_pause(system_state=SystemState(0, 100.0, 65.0), thresholds=_thresholds())
    assert all(decision.is_idle for decision in result.decisions)
    assert scheduler.pause_state.is_paused is True


def test_pause_period_has_zero_compute_power_and_no_task_progress() -> None:
    scheduler = FcfsIntermittentScheduler(thresholds=_thresholds())
    scheduler.decide(
        current_time=0,
        tasks=[_task("T0", duration=3)],
        cores=[_core()],
        system_state=SystemState(time_slot=0, energy_joule=100.0, temperature_celsius=25.0),
        environment_window=[],
    )
    remaining_before_pause = scheduler.remaining_work_for("T0")
    result = scheduler.decide(
        current_time=1,
        tasks=[_task("T0", duration=3)],
        cores=[_core()],
        system_state=SystemState(time_slot=1, energy_joule=100.0, temperature_celsius=65.0),
        environment_window=[],
    )

    assert all(decision.is_idle for decision in result.decisions)
    assert scheduler.last_compute_power_w == 0.0
    assert scheduler.remaining_work_for("T0") == remaining_before_pause


def test_resume_threshold_continues_previous_fcfs_task() -> None:
    scheduler = FcfsIntermittentScheduler(thresholds=_thresholds())
    tasks = [_task("T0", duration=3), _task("T1", duration=1)]
    scheduler.decide(
        current_time=0,
        tasks=tasks,
        cores=[_core()],
        system_state=SystemState(time_slot=0, energy_joule=100.0, temperature_celsius=25.0),
        environment_window=[],
    )
    scheduler.decide(
        current_time=1,
        tasks=tasks,
        cores=[_core()],
        system_state=SystemState(time_slot=1, energy_joule=100.0, temperature_celsius=65.0),
        environment_window=[],
    )
    result = scheduler.decide(
        current_time=2,
        tasks=tasks,
        cores=[_core()],
        system_state=SystemState(time_slot=2, energy_joule=100.0, temperature_celsius=45.0),
        environment_window=[],
    )

    assert should_resume(system_state=SystemState(2, 100.0, 45.0), thresholds=_thresholds())
    assert result.decisions[0].task_id == "T0"
    assert scheduler.pause_state.is_paused is False


def test_fcfs_intermittent_does_not_skip_queue_head_task() -> None:
    scheduler = FcfsIntermittentScheduler(thresholds=_thresholds())
    result = scheduler.decide(
        current_time=0,
        tasks=[_task("T0"), _task("T1")],
        cores=[_core()],
        system_state=SystemState(time_slot=0, energy_joule=100.0, temperature_celsius=25.0),
        environment_window=[],
    )

    assert result.decisions[0].task_id == "T0"
