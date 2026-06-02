"""Common scheduler interface and validation helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

from .models import Core, ScheduleDecision, ScheduleTrace, SolverTrace, SystemState, Task


class SchedulerError(ValueError):
    """Raised when a scheduler returns invalid decisions."""


@dataclass(slots=True)
class SchedulerResult:
    """Scheduler output for one time slot."""

    decisions: tuple[ScheduleDecision, ...]
    solver_trace: SolverTrace | None = None

    def __init__(
        self,
        decisions: Sequence[ScheduleDecision],
        solver_trace: SolverTrace | None = None,
    ) -> None:
        self.decisions = tuple(decisions)
        self.solver_trace = solver_trace

    def to_schedule_trace(self, time_slot: int) -> ScheduleTrace:
        return ScheduleTrace(time_slot=time_slot, decisions=list(self.decisions))


class BaseScheduler(ABC):
    """Abstract base class for all TEBS schedulers."""

    name = "base"

    @abstractmethod
    def decide(
        self,
        *,
        current_time: int,
        tasks: Sequence[Task],
        cores: Sequence[Core],
        system_state: SystemState,
        environment_window: Sequence[Any],
    ) -> SchedulerResult:
        """Return decisions for the current slot."""


class SafeIdleScheduler(BaseScheduler):
    """A conservative scheduler that idles every core."""

    name = "safe_idle"

    def decide(
        self,
        *,
        current_time: int,
        tasks: Sequence[Task],
        cores: Sequence[Core],
        system_state: SystemState,
        environment_window: Sequence[Any],
    ) -> SchedulerResult:
        del tasks, system_state, environment_window
        return SchedulerResult(make_idle_decisions(current_time=current_time, cores=cores))


def make_idle_decisions(
    *,
    current_time: int,
    cores: Sequence[Core],
) -> tuple[ScheduleDecision, ...]:
    """Create one idle decision per core."""

    return tuple(ScheduleDecision.idle(time_slot=current_time, core_id=core.core_id) for core in cores)


def validate_scheduler_result(
    *,
    result: SchedulerResult,
    current_time: int,
    tasks: Sequence[Task],
    cores: Sequence[Core],
    require_all_cores: bool = False,
) -> None:
    """Validate core IDs, time slots and task/block references."""

    core_ids = {core.core_id for core in cores}
    if not core_ids:
        raise SchedulerError("cores must not be empty.")
    task_by_id = {task.task_id: task for task in tasks}
    seen_cores: set[str] = set()

    for decision in result.decisions:
        if decision.time_slot != current_time:
            raise SchedulerError("All decisions must match current_time.")
        if decision.core_id not in core_ids:
            raise SchedulerError(f"Unknown core_id in scheduler decision: {decision.core_id}")
        if decision.core_id in seen_cores:
            raise SchedulerError(f"Duplicate decision for core_id: {decision.core_id}")
        seen_cores.add(decision.core_id)
        if decision.is_idle:
            continue
        if decision.task_id not in task_by_id:
            raise SchedulerError(f"Unknown task_id in scheduler decision: {decision.task_id}")
        if decision.block_id is not None:
            block_ids = {block.block_id for block in task_by_id[decision.task_id].blocks}
            if decision.block_id not in block_ids:
                raise SchedulerError(
                    f"Unknown block_id {decision.block_id} for task_id {decision.task_id}"
                )

    if require_all_cores and seen_cores != core_ids:
        missing = ", ".join(sorted(core_ids - seen_cores))
        raise SchedulerError(f"Scheduler did not return decisions for all cores: {missing}")


__all__ = [
    "BaseScheduler",
    "SafeIdleScheduler",
    "SchedulerError",
    "SchedulerResult",
    "make_idle_decisions",
    "validate_scheduler_result",
]
