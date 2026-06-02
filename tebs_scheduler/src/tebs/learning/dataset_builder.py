"""Build supervised TCN samples from scheduler traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from tebs.environment import EnvironmentSlot
from tebs.models import Core, ScheduleTrace, SystemState, Task, TaskBlock


class DatasetBuilderError(ValueError):
    """Raised when TCN dataset inputs are invalid."""


@dataclass(frozen=True, slots=True)
class TcnFeatureSpec:
    """Shape and feature-name contract for one TCN dataset."""

    horizon_slots: int
    max_tasks: int
    max_blocks_per_task: int
    core_ids: tuple[str, ...]
    feature_names: tuple[str, ...]

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)


@dataclass(frozen=True, slots=True)
class ActionLabel:
    """Supervised label for one core at one slot."""

    core_id: str
    action_index: int
    is_idle: bool
    task_id: str | None = None
    block_id: int | None = None


@dataclass(frozen=True, slots=True)
class TcnTrainingSample:
    """One fixed-window TCN input and per-core action labels."""

    time_slot: int
    feature_window: tuple[tuple[float, ...], ...]
    labels: tuple[ActionLabel, ...]


@dataclass(frozen=True, slots=True)
class TcnDataset:
    """A compact in-memory supervised dataset."""

    feature_spec: TcnFeatureSpec
    samples: tuple[TcnTrainingSample, ...]

    @property
    def feature_dim(self) -> int:
        return self.feature_spec.feature_dim

    def as_xy(self) -> tuple[tuple[tuple[tuple[float, ...], ...], ...], tuple[tuple[int, ...], ...]]:
        """Return tuple-based X/y arrays for lightweight tests or exporters."""

        x = tuple(sample.feature_window for sample in self.samples)
        y = tuple(tuple(label.action_index for label in sample.labels) for sample in self.samples)
        return x, y


def build_dataset_from_traces(
    *,
    tasks: Sequence[Task],
    cores: Sequence[Core],
    schedule_trace: Sequence[ScheduleTrace],
    state_trace: Sequence[SystemState],
    environment_slots: Sequence[EnvironmentSlot],
    horizon_slots: int,
    max_tasks: int | None = None,
) -> TcnDataset:
    """Convert simulation traces into fixed-window supervised TCN samples."""

    if horizon_slots <= 0:
        raise DatasetBuilderError("horizon_slots must be > 0.")
    if not cores:
        raise DatasetBuilderError("cores must not be empty.")
    if len(schedule_trace) != len(state_trace):
        raise DatasetBuilderError("schedule_trace and state_trace must have the same length.")
    if len(environment_slots) < len(schedule_trace):
        raise DatasetBuilderError("environment_slots must cover schedule_trace.")

    task_limit = max_tasks or len(tasks)
    if task_limit <= 0:
        raise DatasetBuilderError("max_tasks must be > 0.")
    max_blocks = _max_blocks_per_task(tasks)
    feature_names = build_feature_names(
        max_tasks=task_limit,
        max_blocks_per_task=max_blocks,
        cores=cores,
    )
    feature_spec = TcnFeatureSpec(
        horizon_slots=horizon_slots,
        max_tasks=task_limit,
        max_blocks_per_task=max_blocks,
        core_ids=tuple(core.core_id for core in cores),
        feature_names=feature_names,
    )

    samples: list[TcnTrainingSample] = []
    previously_scheduled_blocks: set[tuple[str, int]] = set()
    tasks_in_feature_order = _ordered_tasks(tasks)[:task_limit]

    for trace, state in zip(schedule_trace, state_trace):
        if trace.time_slot != state.time_slot:
            raise DatasetBuilderError("schedule_trace and state_trace time slots must match.")
        feature_window = build_feature_window(
            current_time=trace.time_slot,
            tasks=tasks_in_feature_order,
            cores=cores,
            system_state=state,
            environment_window=environment_slots[trace.time_slot : trace.time_slot + horizon_slots],
            horizon_slots=horizon_slots,
            max_tasks=task_limit,
            completed_blocks=previously_scheduled_blocks,
        )
        labels = tuple(
            _label_for_core(
                core=core,
                trace=trace,
                tasks_in_feature_order=tasks_in_feature_order,
                max_blocks_per_task=max_blocks,
            )
            for core in cores
        )
        samples.append(
            TcnTrainingSample(
                time_slot=trace.time_slot,
                feature_window=feature_window,
                labels=labels,
            )
        )
        previously_scheduled_blocks.update(
            (decision.task_id, _decision_block_id(decision.block_id))
            for decision in trace.decisions
            if not decision.is_idle and decision.task_id is not None
        )

    return TcnDataset(feature_spec=feature_spec, samples=tuple(samples))


def build_feature_window(
    *,
    current_time: int,
    tasks: Sequence[Task],
    cores: Sequence[Core],
    system_state: SystemState,
    environment_window: Sequence[EnvironmentSlot],
    horizon_slots: int,
    max_tasks: int | None = None,
    completed_blocks: set[tuple[str, int]] | frozenset[tuple[str, int]] | None = None,
) -> tuple[tuple[float, ...], ...]:
    """Build an H x F feature window for TCN inference or dataset generation."""

    if horizon_slots <= 0:
        raise DatasetBuilderError("horizon_slots must be > 0.")
    if current_time < 0:
        raise DatasetBuilderError("current_time must be >= 0.")
    if not cores:
        raise DatasetBuilderError("cores must not be empty.")

    task_limit = max_tasks or len(tasks)
    if task_limit <= 0:
        raise DatasetBuilderError("max_tasks must be > 0.")
    completed = completed_blocks or set()
    ordered_tasks = _ordered_tasks(tasks)[:task_limit]
    rows: list[tuple[float, ...]] = []

    for offset in range(horizon_slots):
        env_slot = _environment_at(environment_window, offset)
        slot_time = current_time + offset
        row: list[float] = [
            float(offset),
            float(system_state.energy_joule),
            float(system_state.temperature_celsius),
            1.0 if system_state.energy_violation else 0.0,
            1.0 if system_state.thermal_violation else 0.0,
            float(env_slot.harvested_power_w),
            float(env_slot.base_power_w),
            float(env_slot.ambient_temperature_celsius),
            1.0 if env_slot.is_sunlight else 0.0,
        ]
        row.extend(_core_features(cores))
        for task_index in range(task_limit):
            if task_index < len(ordered_tasks):
                row.extend(_task_features(ordered_tasks[task_index], slot_time, completed))
            else:
                row.extend(_empty_task_features())
        rows.append(tuple(row))

    return tuple(rows)


def build_feature_names(
    *,
    max_tasks: int,
    max_blocks_per_task: int,
    cores: Sequence[Core],
) -> tuple[str, ...]:
    """Return the feature names emitted by build_feature_window."""

    if max_tasks <= 0:
        raise DatasetBuilderError("max_tasks must be > 0.")
    if max_blocks_per_task <= 0:
        raise DatasetBuilderError("max_blocks_per_task must be > 0.")
    names = [
        "horizon_offset",
        "energy_joule",
        "temperature_celsius",
        "energy_violation",
        "thermal_violation",
        "harvested_power_w",
        "base_power_w",
        "ambient_temperature_celsius",
        "is_sunlight",
    ]
    for core in cores:
        names.extend(
            [
                f"core_{core.core_id}_is_hp",
                f"core_{core.core_id}_is_lp",
                f"core_{core.core_id}_speed_scale",
                f"core_{core.core_id}_power_scale",
            ]
        )
    for task_index in range(max_tasks):
        names.extend(
            [
                f"task_{task_index}_present",
                f"task_{task_index}_released",
                f"task_{task_index}_remaining_blocks",
                f"task_{task_index}_weight",
                f"task_{task_index}_deadline_slack",
                f"task_{task_index}_min_duration",
                f"task_{task_index}_min_power_w",
            ]
        )
    return tuple(names)


def _label_for_core(
    *,
    core: Core,
    trace: ScheduleTrace,
    tasks_in_feature_order: Sequence[Task],
    max_blocks_per_task: int,
) -> ActionLabel:
    decision = next((item for item in trace.decisions if item.core_id == core.core_id), None)
    if decision is None or decision.is_idle:
        return ActionLabel(core_id=core.core_id, action_index=0, is_idle=True)
    task_index = next(
        (
            index
            for index, task in enumerate(tasks_in_feature_order)
            if task.task_id == decision.task_id
        ),
        None,
    )
    if task_index is None:
        return ActionLabel(core_id=core.core_id, action_index=0, is_idle=True)
    block_id = _decision_block_id(decision.block_id)
    return ActionLabel(
        core_id=core.core_id,
        action_index=1 + task_index * max_blocks_per_task + block_id,
        is_idle=False,
        task_id=decision.task_id,
        block_id=block_id,
    )


def _decision_block_id(block_id: int | None) -> int:
    return 0 if block_id is None else block_id


def _ordered_tasks(tasks: Sequence[Task]) -> list[Task]:
    return sorted(tasks, key=lambda task: (task.release_time, task.deadline, task.task_id))


def _max_blocks_per_task(tasks: Sequence[Task]) -> int:
    if not tasks:
        return 1
    return max(1, max(len(task.blocks) for task in tasks))


def _environment_at(environment_window: Sequence[EnvironmentSlot], offset: int) -> EnvironmentSlot:
    if not environment_window:
        raise DatasetBuilderError("environment_window must not be empty.")
    if offset < len(environment_window):
        return environment_window[offset]
    return environment_window[-1]


def _core_features(cores: Sequence[Core]) -> list[float]:
    values: list[float] = []
    for core in cores:
        values.extend(
            [
                1.0 if core.core_type == "high_performance" else 0.0,
                1.0 if core.core_type == "low_power" else 0.0,
                float(core.speed_scale),
                float(core.power_scale),
            ]
        )
    return values


def _task_features(
    task: Task,
    slot_time: int,
    completed_blocks: set[tuple[str, int]] | frozenset[tuple[str, int]],
) -> list[float]:
    unfinished_blocks = [
        block for block in task.blocks if (task.task_id, block.block_id) not in completed_blocks
    ]
    min_duration = min(
        (min(block.duration_by_core.values()) for block in unfinished_blocks),
        default=0,
    )
    min_power = min(
        (min(block.power_by_core.values()) for block in unfinished_blocks),
        default=0.0,
    )
    return [
        1.0,
        1.0 if slot_time >= task.release_time else 0.0,
        float(len(unfinished_blocks)),
        float(task.weight),
        float(task.deadline - slot_time),
        float(min_duration),
        float(min_power),
    ]


def _empty_task_features() -> list[float]:
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


__all__ = [
    "ActionLabel",
    "DatasetBuilderError",
    "TcnDataset",
    "TcnFeatureSpec",
    "TcnTrainingSample",
    "build_dataset_from_traces",
    "build_feature_names",
    "build_feature_window",
]
