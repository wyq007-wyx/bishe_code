"""Discrete-time simulation loop for TEBS schedulers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .config import ConfigBundle, load_config
from .energy_model import update_energy
from .environment import EnvironmentSlot, OrbitEnvironment, ensure_environment_slots, environment_from_config
from .models import Core, ScheduleDecision, ScheduleTrace, SolverTrace, SystemState, Task, TaskBlock
from .scheduler_base import BaseScheduler, DecisionRuntimeEffect, SchedulerResult, validate_scheduler_result
from .thermal_model import update_temperature


class SimulationError(ValueError):
    """Raised when a simulation cannot be executed safely."""


@dataclass(frozen=True, slots=True)
class TaskCompletionRecord:
    """Task-level completion record emitted by the simulator."""

    task_id: str
    release_time: int
    deadline: int
    weight: float
    completion_time: int
    completed_block_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Complete simulation output."""

    schedule_trace: tuple[ScheduleTrace, ...]
    state_trace: tuple[SystemState, ...]
    task_completion_records: tuple[TaskCompletionRecord, ...]
    solver_trace: tuple[SolverTrace, ...]


@dataclass(slots=True)
class _BlockProgress:
    remaining_slots: float | None = None
    completed: bool = False


def initial_state_from_config(config: ConfigBundle | None = None) -> SystemState:
    """Create the default initial system state."""

    cfg = config or load_config()
    return SystemState(
        time_slot=0,
        energy_joule=cfg.energy.initial_energy_j,
        temperature_celsius=cfg.thermal.initial_temperature_celsius,
    )


def run_simulation(
    *,
    scheduler: BaseScheduler,
    tasks: Sequence[Task],
    cores: Sequence[Core],
    environment: OrbitEnvironment | Sequence[EnvironmentSlot] | None = None,
    initial_system_state: SystemState | None = None,
    simulation_slots: int | None = None,
    config: ConfigBundle | None = None,
) -> SimulationResult:
    """Run a scheduler against a task set and orbit environment."""

    if not cores:
        raise SimulationError("cores must not be empty.")
    cfg = config or load_config()
    env = environment or environment_from_config(cfg, simulation_slots=simulation_slots)
    environment_slots = ensure_environment_slots(env)
    total_slots = simulation_slots or len(environment_slots)
    if total_slots <= 0:
        raise SimulationError("simulation_slots must be > 0.")
    if total_slots > len(environment_slots):
        raise SimulationError("environment length is shorter than simulation_slots.")

    state = initial_system_state or initial_state_from_config(cfg)
    task_by_id = {task.task_id: task for task in tasks}
    if len(task_by_id) != len(tasks):
        raise SimulationError("tasks must have unique task_id values.")

    progress: dict[tuple[str, int], _BlockProgress] = {
        (task.task_id, block.block_id): _BlockProgress()
        for task in tasks
        for block in task.blocks
    }
    completed_blocks: set[tuple[str, int]] = set()
    completed_tasks: set[str] = set()

    schedule_trace: list[ScheduleTrace] = []
    state_trace: list[SystemState] = []
    completion_records: list[TaskCompletionRecord] = []
    solver_trace: list[SolverTrace] = []

    for current_time in range(total_slots):
        env_slot = environment_slots[current_time]
        result = scheduler.decide(
            current_time=current_time,
            tasks=tasks,
            cores=cores,
            system_state=state,
            environment_window=environment_slots[current_time : current_time + cfg.simulation.horizon_slots],
        )
        if not isinstance(result, SchedulerResult):
            raise SimulationError("scheduler.decide() must return SchedulerResult.")
        validate_scheduler_result(
            result=result,
            current_time=current_time,
            tasks=tasks,
            cores=cores,
        )
        trace = result.to_schedule_trace(current_time)
        compute_power_w = _apply_decisions(
            decisions=trace.decisions,
            runtime_effect_by_core=result.runtime_effect_by_core or {},
            current_time=current_time,
            task_by_id=task_by_id,
            progress=progress,
            completed_blocks=completed_blocks,
            completed_tasks=completed_tasks,
            completion_records=completion_records,
        )
        if result.solver_trace is not None:
            solver_trace.append(result.solver_trace)

        energy_step = update_energy(
            energy_prev_j=state.energy_joule,
            harvested_power_w=env_slot.harvested_power_w,
            base_power_w=env_slot.base_power_w,
            compute_power_w=compute_power_w,
            delta_t_seconds=cfg.simulation.delta_t_seconds,
            energy_min_j=cfg.energy.e_min_j,
            energy_max_j=cfg.energy.e_max_j,
        )
        thermal_step = update_temperature(
            temperature_prev_celsius=state.temperature_celsius,
            compute_power_w=compute_power_w,
            base_power_w=env_slot.base_power_w,
            delta_t_seconds=cfg.simulation.delta_t_seconds,
            temperature_max_celsius=cfg.thermal.t_max_celsius,
            thermal_capacity_j_per_c=cfg.thermal.thermal_capacity_j_per_c,
            cooling_coeff=cfg.thermal.cooling_coeff,
            ambient_temperature_celsius=env_slot.ambient_temperature_celsius,
        )
        state = SystemState(
            time_slot=current_time,
            energy_joule=energy_step.energy_next_j,
            temperature_celsius=thermal_step.temperature_next_celsius,
            energy_violation=energy_step.energy_violation,
            thermal_violation=thermal_step.thermal_violation,
        )
        schedule_trace.append(trace)
        state_trace.append(state)

    return SimulationResult(
        schedule_trace=tuple(schedule_trace),
        state_trace=tuple(state_trace),
        task_completion_records=tuple(completion_records),
        solver_trace=tuple(solver_trace),
    )


def _apply_decisions(
    *,
    decisions: Sequence[ScheduleDecision],
    runtime_effect_by_core: dict[str, DecisionRuntimeEffect],
    current_time: int,
    task_by_id: dict[str, Task],
    progress: dict[tuple[str, int], _BlockProgress],
    completed_blocks: set[tuple[str, int]],
    completed_tasks: set[str],
    completion_records: list[TaskCompletionRecord],
) -> float:
    compute_power_w = 0.0
    slot_completed_blocks: list[tuple[str, int]] = []
    running_blocks: set[tuple[str, int]] = set()

    for decision in decisions:
        if decision.is_idle:
            continue
        task = task_by_id[decision.task_id or ""]
        block = _resolve_decision_block(
            task=task,
            decision=decision,
            current_time=current_time,
            completed_blocks=completed_blocks,
        )
        block_key = (task.task_id, block.block_id)
        if block_key in running_blocks:
            raise SimulationError(f"Block {block_key} is assigned more than once in one slot.")
        running_blocks.add(block_key)
        if decision.core_id not in block.duration_by_core:
            raise SimulationError(
                f"Core {decision.core_id} cannot execute task {task.task_id} block {block.block_id}."
            )
        block_progress = progress[block_key]
        if block_progress.completed:
            raise SimulationError(f"Task {task.task_id} block {block.block_id} is already complete.")
        if block_progress.remaining_slots is None:
            block_progress.remaining_slots = float(block.duration_by_core[decision.core_id])
        runtime_effect = runtime_effect_by_core.get(decision.core_id)
        progress_slots = runtime_effect.progress_slots if runtime_effect is not None else 1.0
        decision_power_w = (
            runtime_effect.compute_power_w
            if runtime_effect is not None and runtime_effect.compute_power_w is not None
            else block.power_by_core[decision.core_id]
        )
        block_progress.remaining_slots -= progress_slots
        compute_power_w += decision_power_w
        if block_progress.remaining_slots <= 0:
            block_progress.completed = True
            slot_completed_blocks.append(block_key)

    for block_key in slot_completed_blocks:
        completed_blocks.add(block_key)
        task = task_by_id[block_key[0]]
        if task.task_id not in completed_tasks and all(
            (task.task_id, block.block_id) in completed_blocks for block in task.blocks
        ):
            completed_tasks.add(task.task_id)
            completion_records.append(
                TaskCompletionRecord(
                    task_id=task.task_id,
                    release_time=task.release_time,
                    deadline=task.deadline,
                    weight=task.weight,
                    completion_time=current_time + 1,
                    completed_block_ids=tuple(block.block_id for block in task.blocks),
                )
            )
    return compute_power_w


def _resolve_decision_block(
    *,
    task: Task,
    decision: ScheduleDecision,
    current_time: int,
    completed_blocks: set[tuple[str, int]],
) -> TaskBlock:
    if current_time < task.release_time:
        raise SimulationError(f"Task {task.task_id} is not released at time {current_time}.")
    if decision.block_id is not None:
        block = next(block for block in task.blocks if block.block_id == decision.block_id)
        _validate_block_ready(task, block, current_time, completed_blocks)
        return block
    for block in sorted(task.blocks, key=lambda item: item.block_id):
        if (task.task_id, block.block_id) in completed_blocks:
            continue
        if _is_block_ready(task, block, current_time, completed_blocks):
            return block
    raise SimulationError(f"No ready block found for task {task.task_id}.")


def _validate_block_ready(
    task: Task,
    block: TaskBlock,
    current_time: int,
    completed_blocks: set[tuple[str, int]],
) -> None:
    if not _is_block_ready(task, block, current_time, completed_blocks):
        raise SimulationError(f"Task {task.task_id} block {block.block_id} is not ready.")


def _is_block_ready(
    task: Task,
    block: TaskBlock,
    current_time: int,
    completed_blocks: set[tuple[str, int]],
) -> bool:
    if current_time < block.release_time:
        return False
    predecessor = block.predecessor_block_id
    if predecessor is None:
        return True
    return (task.task_id, predecessor) in completed_blocks


__all__ = [
    "SimulationError",
    "SimulationResult",
    "TaskCompletionRecord",
    "initial_state_from_config",
    "run_simulation",
]
