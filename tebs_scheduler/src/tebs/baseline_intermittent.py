"""FCFS intermittent-computing rule baseline scheduler."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .models import Core, ScheduleDecision, SystemState, Task, TaskBlock
from .scheduler_base import BaseScheduler, SchedulerResult, make_idle_decisions


class IntermittentBaselineError(ValueError):
    """Raised when intermittent baseline inputs are invalid."""


@dataclass(frozen=True, slots=True)
class IntermittentThresholds:
    """Pause/resume thresholds with hysteresis."""

    pause_temperature_celsius: float = 65.0
    resume_temperature_celsius: float = 55.0
    pause_energy_j: float = 116000.0
    resume_energy_j: float = 122400.0

    def __post_init__(self) -> None:
        if self.resume_temperature_celsius > self.pause_temperature_celsius:
            raise IntermittentBaselineError(
                "resume_temperature_celsius must be <= pause_temperature_celsius."
            )
        if self.resume_energy_j < self.pause_energy_j:
            raise IntermittentBaselineError("resume_energy_j must be >= pause_energy_j.")


@dataclass(slots=True)
class IntermittentPauseState:
    """Current pause/resume state."""

    is_paused: bool = False
    pause_count: int = 0


def should_pause(
    *,
    system_state: SystemState,
    thresholds: IntermittentThresholds = IntermittentThresholds(),
) -> bool:
    """Return whether compute should pause now."""

    return (
        system_state.temperature_celsius >= thresholds.pause_temperature_celsius
        or system_state.energy_joule <= thresholds.pause_energy_j
    )


def should_resume(
    *,
    system_state: SystemState,
    thresholds: IntermittentThresholds = IntermittentThresholds(),
) -> bool:
    """Return whether a paused scheduler can resume compute."""

    return (
        system_state.temperature_celsius <= thresholds.resume_temperature_celsius
        and system_state.energy_joule >= thresholds.resume_energy_j
    )


@dataclass(slots=True)
class FcfsIntermittentScheduler(BaseScheduler):
    """Fixed FCFS queue with pause/resume intermittent execution."""

    thresholds: IntermittentThresholds = field(default_factory=IntermittentThresholds)
    pause_state: IntermittentPauseState = field(default_factory=IntermittentPauseState)
    name: str = "fcfs_intermittent"
    _remaining_work_slots: dict[tuple[str, int], float] = field(default_factory=dict)
    _completed_blocks: set[tuple[str, int]] = field(default_factory=set)
    _completed_tasks: set[str] = field(default_factory=set)
    last_compute_power_w: float = 0.0

    def decide(
        self,
        *,
        current_time: int,
        tasks: Sequence[Task],
        cores: Sequence[Core],
        system_state: SystemState,
        environment_window: Sequence[Any],
    ) -> SchedulerResult:
        del environment_window
        self.last_compute_power_w = 0.0

        if self.pause_state.is_paused:
            if should_resume(system_state=system_state, thresholds=self.thresholds):
                self.pause_state.is_paused = False
            else:
                return SchedulerResult(make_idle_decisions(current_time=current_time, cores=cores))

        if should_pause(system_state=system_state, thresholds=self.thresholds):
            self.pause_state.is_paused = True
            self.pause_state.pause_count += 1
            return SchedulerResult(make_idle_decisions(current_time=current_time, cores=cores))

        candidate = self._select_fcfs_candidate(
            current_time=current_time,
            tasks=tasks,
            cores=cores,
        )
        if candidate is None:
            return SchedulerResult(make_idle_decisions(current_time=current_time, cores=cores))

        task, block, core = candidate
        decision = ScheduleDecision.run_block(
            time_slot=current_time,
            core_id=core.core_id,
            task_id=task.task_id,
            block_id=block.block_id,
        )
        self.last_compute_power_w = block.power_by_core[core.core_id]
        self._advance_progress(task=task, block=block, core=core)

        decisions = [decision]
        decisions.extend(
            ScheduleDecision.idle(time_slot=current_time, core_id=other_core.core_id)
            for other_core in cores
            if other_core.core_id != core.core_id
        )
        return SchedulerResult(decisions=decisions)

    def remaining_work_for(self, task_id: str, block_id: int = 0) -> float | None:
        """Return scheduler-side remaining work for tests/inspection."""

        return self._remaining_work_slots.get((task_id, block_id))

    def _select_fcfs_candidate(
        self,
        *,
        current_time: int,
        tasks: Sequence[Task],
        cores: Sequence[Core],
    ) -> tuple[Task, TaskBlock, Core] | None:
        if not cores:
            return None
        for task in sorted(tasks, key=lambda item: (item.release_time, item.task_id)):
            if task.task_id in self._completed_tasks:
                continue
            if current_time < task.release_time:
                return None
            block = self._first_unfinished_ready_block(task, current_time)
            if block is None:
                return None
            core = _first_compatible_core(block, cores)
            if core is None:
                return None
            return task, block, core
        return None

    def _first_unfinished_ready_block(
        self,
        task: Task,
        current_time: int,
    ) -> TaskBlock | None:
        for block in sorted(task.blocks, key=lambda item: item.block_id):
            key = (task.task_id, block.block_id)
            if key in self._completed_blocks:
                continue
            if current_time < block.release_time:
                return None
            predecessor = block.predecessor_block_id
            if predecessor is not None and (task.task_id, predecessor) not in self._completed_blocks:
                return None
            return block
        return None

    def _advance_progress(
        self,
        *,
        task: Task,
        block: TaskBlock,
        core: Core,
    ) -> None:
        key = (task.task_id, block.block_id)
        if key not in self._remaining_work_slots:
            self._remaining_work_slots[key] = float(block.duration_by_core[core.core_id])
        self._remaining_work_slots[key] -= 1.0
        if self._remaining_work_slots[key] <= 0:
            self._completed_blocks.add(key)
            if all((task.task_id, item.block_id) in self._completed_blocks for item in task.blocks):
                self._completed_tasks.add(task.task_id)


def _first_compatible_core(block: TaskBlock, cores: Sequence[Core]) -> Core | None:
    for core in cores:
        if core.core_id in block.duration_by_core:
            return core
    return None


__all__ = [
    "FcfsIntermittentScheduler",
    "FcfSIntermittentScheduler",
    "IntermittentBaselineError",
    "IntermittentPauseState",
    "IntermittentThresholds",
    "should_pause",
    "should_resume",
]

FcfSIntermittentScheduler = FcfsIntermittentScheduler
