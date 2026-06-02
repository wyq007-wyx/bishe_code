from __future__ import annotations

import pytest

from tebs.baseline_dvfs import (
    DEFAULT_FREQUENCY_LEVELS,
    DvfsThresholds,
    FcfsDvfsScheduler,
    progress_increment_slots,
    scaled_power_w,
    select_frequency_level,
)
from tebs.models import Core, SystemState, Task


def _core() -> Core:
    return Core(core_id="hp_0", core_type="high_performance")


def _task(task_id: str, duration: int = 4, release_time: int = 0) -> Task:
    return Task.single_block(
        task_id=task_id,
        release_time=release_time,
        deadline=20,
        weight=1.0,
        duration_by_core={"hp_0": duration},
        power_by_core={"hp_0": 10.0},
        core_ids=("hp_0",),
    )


def _thresholds() -> DvfsThresholds:
    return DvfsThresholds(
        temperature_warn_celsius=50.0,
        temperature_crit_celsius=60.0,
        energy_warn_j=80.0,
        energy_crit_j=40.0,
    )


def test_fcfs_dvfs_does_not_skip_queue_head_task() -> None:
    scheduler = FcfsDvfsScheduler(thresholds=_thresholds())
    result = scheduler.decide(
        current_time=0,
        tasks=[_task("T0"), _task("T1")],
        cores=[_core()],
        system_state=SystemState(time_slot=0, energy_joule=100.0, temperature_celsius=25.0),
        environment_window=[],
    )

    assert result.decisions[0].task_id == "T0"
    assert result.decisions[0].frequency_level == "high"


def test_high_temperature_selects_low_frequency() -> None:
    level = select_frequency_level(
        system_state=SystemState(time_slot=0, energy_joule=100.0, temperature_celsius=65.0),
        thresholds=_thresholds(),
    )

    assert level.name == "low"


def test_low_energy_selects_low_frequency() -> None:
    level = select_frequency_level(
        system_state=SystemState(time_slot=0, energy_joule=35.0, temperature_celsius=25.0),
        thresholds=_thresholds(),
    )

    assert level.name == "low"


def test_safe_thermal_electric_state_selects_high_frequency() -> None:
    level = select_frequency_level(
        system_state=SystemState(time_slot=0, energy_joule=100.0, temperature_celsius=25.0),
        thresholds=_thresholds(),
    )

    assert level.name == "high"


def test_dvfs_progress_and_power_follow_selected_frequency_level() -> None:
    scheduler = FcfsDvfsScheduler(thresholds=_thresholds())
    result = scheduler.decide(
        current_time=0,
        tasks=[_task("T0", duration=4)],
        cores=[_core()],
        system_state=SystemState(time_slot=0, energy_joule=100.0, temperature_celsius=65.0),
        environment_window=[],
    )
    low = {level.name: level for level in DEFAULT_FREQUENCY_LEVELS}["low"]
    expected_power = scaled_power_w(10.0, low)

    assert progress_increment_slots(low) == pytest.approx(0.5)
    assert scheduler.remaining_work_for("T0") == pytest.approx(3.5)
    assert scheduler.last_compute_power_w == pytest.approx(expected_power)
    assert result.runtime_effect_by_core["hp_0"].progress_slots == pytest.approx(0.5)
    assert result.runtime_effect_by_core["hp_0"].compute_power_w == pytest.approx(expected_power)
