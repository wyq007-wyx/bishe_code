from __future__ import annotations

import pytest

from tebs.models import Core, ScheduleDecision, SystemState, Task
from tebs.scheduler_base import (
    SafeIdleScheduler,
    SchedulerError,
    SchedulerResult,
    validate_scheduler_result,
)


def _cores() -> tuple[Core, ...]:
    return (
        Core(core_id="hp_0", core_type="high_performance"),
        Core(core_id="lp_0", core_type="low_power"),
    )


def _task() -> Task:
    return Task.single_block(
        task_id="T1",
        release_time=0,
        deadline=5,
        weight=1.0,
        duration_by_core={"hp_0": 1, "lp_0": 2},
        power_by_core={"hp_0": 4.0, "lp_0": 2.0},
        core_ids=("hp_0", "lp_0"),
    )


def test_safe_idle_scheduler_returns_one_idle_decision_per_core() -> None:
    scheduler = SafeIdleScheduler()
    result = scheduler.decide(
        current_time=3,
        tasks=[_task()],
        cores=_cores(),
        system_state=SystemState(time_slot=3, energy_joule=100.0, temperature_celsius=25.0),
        environment_window=[],
    )

    assert len(result.decisions) == 2
    assert all(decision.is_idle for decision in result.decisions)
    assert {decision.core_id for decision in result.decisions} == {"hp_0", "lp_0"}
    validate_scheduler_result(
        result=result,
        current_time=3,
        tasks=[_task()],
        cores=_cores(),
        require_all_cores=True,
    )


def test_scheduler_result_rejects_unknown_core_id() -> None:
    result = SchedulerResult(
        [ScheduleDecision.run_block(time_slot=0, core_id="bad", task_id="T1", block_id=0)]
    )

    with pytest.raises(SchedulerError, match="Unknown core_id"):
        validate_scheduler_result(result=result, current_time=0, tasks=[_task()], cores=_cores())


def test_scheduler_result_rejects_unknown_task_block() -> None:
    result = SchedulerResult(
        [ScheduleDecision.run_block(time_slot=0, core_id="hp_0", task_id="T1", block_id=99)]
    )

    with pytest.raises(SchedulerError, match="Unknown block_id"):
        validate_scheduler_result(result=result, current_time=0, tasks=[_task()], cores=_cores())
