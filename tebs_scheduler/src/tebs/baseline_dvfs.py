"""FCFS-DVFS rule baseline scheduler."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .models import Core, ScheduleDecision, SystemState, Task, TaskBlock
from .scheduler_base import BaseScheduler, DecisionRuntimeEffect, SchedulerResult, make_idle_decisions


class DvfsBaselineError(ValueError):
    """Raised when FCFS-DVFS inputs are invalid."""


def _require_non_empty_str(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise DvfsBaselineError(f"{field_name} must be a non-empty string.")


def _require_positive_number(value: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise DvfsBaselineError(f"{field_name} must be > 0.")


def _require_non_negative_number(value: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise DvfsBaselineError(f"{field_name} must be >= 0.")


@dataclass(frozen=True, slots=True)
class FrequencyLevel:
    """One DVFS frequency/voltage level relative to the high level."""

    name: str
    frequency_scale: float
    voltage_scale: float

    def __post_init__(self) -> None:
        _require_non_empty_str(self.name, "name")
        _require_positive_number(self.frequency_scale, "frequency_scale")
        _require_positive_number(self.voltage_scale, "voltage_scale")

    @property
    def power_scale(self) -> float:
        return self.voltage_scale * self.voltage_scale * self.frequency_scale


@dataclass(frozen=True, slots=True)
class DvfsThresholds:
    """Temperature and energy thresholds for selecting DVFS levels."""

    temperature_warn_celsius: float = 55.0
    temperature_crit_celsius: float = 65.0
    energy_warn_j: float = 122400.0
    energy_crit_j: float = 116000.0

    def __post_init__(self) -> None:
        if self.temperature_warn_celsius > self.temperature_crit_celsius:
            raise DvfsBaselineError("temperature_warn_celsius must be <= temperature_crit_celsius.")
        if self.energy_crit_j > self.energy_warn_j:
            raise DvfsBaselineError("energy_crit_j must be <= energy_warn_j.")


DEFAULT_FREQUENCY_LEVELS: tuple[FrequencyLevel, ...] = (
    FrequencyLevel(name="high", frequency_scale=1.0, voltage_scale=1.0),
    FrequencyLevel(name="medium", frequency_scale=0.7, voltage_scale=0.85),
    FrequencyLevel(name="low", frequency_scale=0.5, voltage_scale=0.75),
)


def select_frequency_level(
    *,
    system_state: SystemState,
    thresholds: DvfsThresholds = DvfsThresholds(),
    frequency_levels: Sequence[FrequencyLevel] = DEFAULT_FREQUENCY_LEVELS,
) -> FrequencyLevel:
    """Select high/medium/low DVFS level from thermal-electric thresholds."""

    levels = _frequency_levels_by_name(frequency_levels)
    if (
        system_state.temperature_celsius >= thresholds.temperature_crit_celsius
        or system_state.energy_joule <= thresholds.energy_crit_j
    ):
        return levels["low"]
    if (
        system_state.temperature_celsius >= thresholds.temperature_warn_celsius
        or system_state.energy_joule <= thresholds.energy_warn_j
    ):
        return levels["medium"]
    return levels["high"]


def scaled_power_w(base_power_w: float, frequency_level: FrequencyLevel) -> float:
    """Scale dynamic compute power by voltage squared and frequency."""

    _require_non_negative_number(base_power_w, "base_power_w")
    return base_power_w * frequency_level.power_scale


def progress_increment_slots(frequency_level: FrequencyLevel) -> float:
    """Return progress made in one slot relative to high frequency."""

    return frequency_level.frequency_scale


@dataclass(slots=True)
class FcfsDvfsScheduler(BaseScheduler):
    """Fixed FCFS queue with high/medium/low DVFS control."""

    thresholds: DvfsThresholds = field(default_factory=DvfsThresholds)
    frequency_levels: tuple[FrequencyLevel, ...] = DEFAULT_FREQUENCY_LEVELS
    name: str = "fcfs_dvfs"
    _remaining_work_slots: dict[tuple[str, int], float] = field(default_factory=dict)
    _completed_blocks: set[tuple[str, int]] = field(default_factory=set)
    _completed_tasks: set[str] = field(default_factory=set)
    last_selected_frequency_level: FrequencyLevel | None = None
    last_compute_power_w: float = 0.0

    def __post_init__(self) -> None:
        _frequency_levels_by_name(self.frequency_levels)

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
        level = select_frequency_level(
            system_state=system_state,
            thresholds=self.thresholds,
            frequency_levels=self.frequency_levels,
        )
        self.last_selected_frequency_level = level
        self.last_compute_power_w = 0.0

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
            frequency_level=level.name,
        )
        compute_power_w = scaled_power_w(block.power_by_core[core.core_id], level)
        progress_slots = progress_increment_slots(level)
        self.last_compute_power_w = compute_power_w
        self._advance_progress(task=task, block=block, core=core, progress_slots=progress_slots)

        decisions = [decision]
        decisions.extend(
            ScheduleDecision.idle(time_slot=current_time, core_id=other_core.core_id)
            for other_core in cores
            if other_core.core_id != core.core_id
        )
        return SchedulerResult(
            decisions=decisions,
            runtime_effect_by_core={
                core.core_id: DecisionRuntimeEffect(
                    progress_slots=progress_slots,
                    compute_power_w=compute_power_w,
                )
            },
        )

    def remaining_work_for(self, task_id: str, block_id: int = 0) -> float | None:
        """Return scheduler-side remaining high-frequency work for tests/inspection."""

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
        progress_slots: float,
    ) -> None:
        key = (task.task_id, block.block_id)
        if key not in self._remaining_work_slots:
            self._remaining_work_slots[key] = float(block.duration_by_core[core.core_id])
        self._remaining_work_slots[key] -= progress_slots
        if self._remaining_work_slots[key] <= 0:
            self._completed_blocks.add(key)
            if all((task.task_id, item.block_id) in self._completed_blocks for item in task.blocks):
                self._completed_tasks.add(task.task_id)


def _first_compatible_core(block: TaskBlock, cores: Sequence[Core]) -> Core | None:
    for core in cores:
        if core.core_id in block.duration_by_core:
            return core
    return None


def _frequency_levels_by_name(
    frequency_levels: Sequence[FrequencyLevel],
) -> dict[str, FrequencyLevel]:
    levels = {level.name: level for level in frequency_levels}
    required = {"high", "medium", "low"}
    if set(levels) != required:
        raise DvfsBaselineError("frequency_levels must contain high, medium and low exactly.")
    return levels


__all__ = [
    "DEFAULT_FREQUENCY_LEVELS",
    "DvfsBaselineError",
    "DvfsThresholds",
    "FcfsDvfsScheduler",
    "FcfSDvfsScheduler",
    "FrequencyLevel",
    "progress_increment_slots",
    "scaled_power_w",
    "select_frequency_level",
]

FcfSDvfsScheduler = FcfsDvfsScheduler
