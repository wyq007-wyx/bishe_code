"""Rolling-horizon MILP scheduler using Gurobi."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .config import ConfigBundle, load_config
from .gurobi_adapter import (
    GurobiAdapterError,
    GurobiSolveParams,
    GurobiSolveResult,
    GurobiUnavailableError,
    STATUS_INFEASIBLE,
    STATUS_UNKNOWN,
    optimize_gurobi_model,
    require_gurobi,
)
from .models import Core, ScheduleDecision, SolverTrace, SystemState, Task, TaskBlock
from .scheduler_base import BaseScheduler, SchedulerResult, make_idle_decisions
from .thermal_model import compute_thermal_budget_j


class RhcMilpGurobiError(ValueError):
    """Raised when RHC-MILP-Gurobi inputs are invalid."""


@dataclass(frozen=True, slots=True)
class RhcObjectiveWeights:
    """Objective weights for response, tardiness and served-work reward."""

    alpha: float = 1.0
    beta: float = 1.0
    mu: float = 0.1

    def __post_init__(self) -> None:
        if self.alpha < 0 or self.beta < 0 or self.mu < 0:
            raise RhcMilpGurobiError("objective weights must be >= 0.")


@dataclass(frozen=True, slots=True)
class RhcWindowAssignment:
    """One planned task-block/core assignment inside a rolling horizon."""

    time_slot: int
    core_id: str
    task_id: str
    block_id: int
    duration_slots: int
    compute_power_w: float


@dataclass(slots=True)
class RhcMilpGurobiScheduler(BaseScheduler):
    """RHC-MILP-Gurobi scheduler with thermal-electric constraints."""

    config: ConfigBundle | None = None
    gurobi_params: GurobiSolveParams | None = None
    objective_weights: RhcObjectiveWeights = field(default_factory=RhcObjectiveWeights)
    fallback_to_idle_on_error: bool = True
    name: str = "rhc_milp_gurobi"
    _remaining_work_slots: dict[tuple[str, int], float] = field(default_factory=dict)
    _completed_blocks: set[tuple[str, int]] = field(default_factory=set)
    _completed_tasks: set[str] = field(default_factory=set)
    last_window_plan: tuple[RhcWindowAssignment, ...] = ()
    last_solve_result: GurobiSolveResult | None = None

    def decide(
        self,
        *,
        current_time: int,
        tasks: Sequence[Task],
        cores: Sequence[Core],
        system_state: SystemState,
        environment_window: Sequence[Any],
    ) -> SchedulerResult:
        self.last_window_plan = ()
        self.last_solve_result = None

        cfg = self._config()
        candidate_blocks = _candidate_blocks_for_window(
            current_time=current_time,
            horizon_slots=cfg.simulation.horizon_slots,
            tasks=tasks,
            completed_blocks=self._completed_blocks,
            completed_tasks=self._completed_tasks,
        )
        if not candidate_blocks:
            return SchedulerResult(make_idle_decisions(current_time=current_time, cores=cores))

        try:
            result, plan = _solve_rhc_window(
                current_time=current_time,
                tasks_and_blocks=candidate_blocks,
                cores=cores,
                system_state=system_state,
                environment_window=environment_window,
                config=cfg,
                params=self._params(),
                objective_weights=self.objective_weights,
            )
        except (GurobiUnavailableError, GurobiAdapterError) as exc:
            if not self.fallback_to_idle_on_error:
                raise RhcMilpGurobiError(str(exc)) from exc
            trace = _fallback_solver_trace(
                current_time=current_time,
                status="GUROBI_UNAVAILABLE"
                if isinstance(exc, GurobiUnavailableError)
                else STATUS_UNKNOWN,
            )
            return SchedulerResult(
                make_idle_decisions(current_time=current_time, cores=cores),
                solver_trace=trace,
            )

        self.last_solve_result = result
        solver_trace = result.to_solver_trace(time_slot=current_time)
        if not result.has_feasible_solution:
            return SchedulerResult(
                make_idle_decisions(current_time=current_time, cores=cores),
                solver_trace=solver_trace,
            )

        self.last_window_plan = plan
        current_assignments = [
            assignment for assignment in plan if assignment.time_slot == current_time
        ]
        decisions: list[ScheduleDecision] = []
        assigned_core_ids: set[str] = set()
        task_by_id = {task.task_id: task for task in tasks}
        for assignment in current_assignments:
            decisions.append(
                ScheduleDecision.run_block(
                    time_slot=current_time,
                    core_id=assignment.core_id,
                    task_id=assignment.task_id,
                    block_id=assignment.block_id,
                )
            )
            assigned_core_ids.add(assignment.core_id)
            task = task_by_id[assignment.task_id]
            block = next(block for block in task.blocks if block.block_id == assignment.block_id)
            core = next(core for core in cores if core.core_id == assignment.core_id)
            self._advance_progress(task=task, block=block, core=core)

        decisions.extend(
            ScheduleDecision.idle(time_slot=current_time, core_id=core.core_id)
            for core in cores
            if core.core_id not in assigned_core_ids
        )
        return SchedulerResult(decisions=decisions, solver_trace=solver_trace)

    def remaining_work_for(self, task_id: str, block_id: int = 0) -> float | None:
        """Return scheduler-side remaining work for tests and inspection."""

        return self._remaining_work_slots.get((task_id, block_id))

    def _config(self) -> ConfigBundle:
        return self.config or load_config()

    def _params(self) -> GurobiSolveParams:
        return self.gurobi_params or GurobiSolveParams.from_config(self._config())

    def _advance_progress(self, *, task: Task, block: TaskBlock, core: Core) -> None:
        key = (task.task_id, block.block_id)
        if key not in self._remaining_work_slots:
            self._remaining_work_slots[key] = float(block.duration_by_core[core.core_id])
        self._remaining_work_slots[key] -= 1.0
        if self._remaining_work_slots[key] <= 0:
            self._completed_blocks.add(key)
            if all((task.task_id, item.block_id) in self._completed_blocks for item in task.blocks):
                self._completed_tasks.add(task.task_id)


def _candidate_blocks_for_window(
    *,
    current_time: int,
    horizon_slots: int,
    tasks: Sequence[Task],
    completed_blocks: set[tuple[str, int]],
    completed_tasks: set[str],
) -> tuple[tuple[Task, TaskBlock], ...]:
    window_end = current_time + horizon_slots - 1
    candidates: list[tuple[Task, TaskBlock]] = []
    for task in tasks:
        if task.task_id in completed_tasks:
            continue
        if task.release_time > window_end:
            continue
        for block in sorted(task.blocks, key=lambda item: item.block_id):
            key = (task.task_id, block.block_id)
            if key in completed_blocks:
                continue
            if block.release_time > window_end:
                continue
            predecessor = block.predecessor_block_id
            if predecessor is not None and (task.task_id, predecessor) not in completed_blocks:
                continue
            candidates.append((task, block))
    return tuple(candidates)


def _solve_rhc_window(
    *,
    current_time: int,
    tasks_and_blocks: Sequence[tuple[Task, TaskBlock]],
    cores: Sequence[Core],
    system_state: SystemState,
    environment_window: Sequence[Any],
    config: ConfigBundle,
    params: GurobiSolveParams,
    objective_weights: RhcObjectiveWeights,
) -> tuple[GurobiSolveResult, tuple[RhcWindowAssignment, ...]]:
    gp = require_gurobi()
    try:
        horizon_slots = config.simulation.horizon_slots
        model = gp.Model("rhc_milp_gurobi_window")
        x: dict[tuple[int, str, int], Any] = {}
        for block_index, (task, block) in enumerate(tasks_and_blocks):
            earliest = max(current_time, task.release_time, block.release_time)
            for slot in range(earliest, current_time + horizon_slots):
                for core in cores:
                    if core.core_id not in block.duration_by_core:
                        continue
                    x[(block_index, core.core_id, slot)] = model.addVar(
                        vtype=gp.GRB.BINARY,
                        name=f"x_{block_index}_{core.core_id}_{slot}",
                    )

        if not x:
            return (
                GurobiSolveResult(
                    status=STATUS_INFEASIBLE,
                    raw_status=None,
                    solve_time_sec=0.0,
                    has_feasible_solution=False,
                    model_name="rhc_milp_gurobi_window",
                ),
                (),
            )

        for slot in range(current_time, current_time + horizon_slots):
            for core in cores:
                model.addConstr(
                    gp.quicksum(
                        var
                        for (_block_index, core_id, var_slot), var in x.items()
                        if core_id == core.core_id and var_slot == slot
                    )
                    <= 1,
                    name=f"core_capacity_{core.core_id}_{slot}",
                )
            for block_index, _ in enumerate(tasks_and_blocks):
                model.addConstr(
                    gp.quicksum(
                        var
                        for (var_block_index, _core_id, var_slot), var in x.items()
                        if var_block_index == block_index and var_slot == slot
                    )
                    <= 1,
                    name=f"block_capacity_{block_index}_{slot}",
                )

        model.addConstr(gp.quicksum(x.values()) >= 1, name="serve_at_least_one_block")

        delta_t = config.simulation.delta_t_seconds
        compute_power_by_slot: dict[int, Any] = {}
        for slot in range(current_time, current_time + horizon_slots):
            compute_power_by_slot[slot] = gp.quicksum(
                tasks_and_blocks[block_index][1].power_by_core[core_id] * var
                for (block_index, core_id, var_slot), var in x.items()
                if var_slot == slot
            )
            env_slot = _window_slot_or_default(
                environment_window=environment_window,
                offset=slot - current_time,
                config=config,
            )
            thermal_budget_j = compute_thermal_budget_j(
                temperature_prev_celsius=system_state.temperature_celsius,
                base_power_w=env_slot.base_power_w,
                delta_t_seconds=delta_t,
                temperature_max_celsius=config.thermal.t_max_celsius,
                thermal_capacity_j_per_c=config.thermal.thermal_capacity_j_per_c,
                cooling_coeff=config.thermal.cooling_coeff,
                ambient_temperature_celsius=env_slot.ambient_temperature_celsius,
            )
            model.addConstr(
                compute_power_by_slot[slot] * delta_t <= thermal_budget_j,
                name=f"thermal_budget_{slot}",
            )

        for slot in range(current_time, current_time + horizon_slots):
            cumulative_harvest_base_j = 0.0
            cumulative_compute_j = 0
            for inner_slot in range(current_time, slot + 1):
                env_slot = _window_slot_or_default(
                    environment_window=environment_window,
                    offset=inner_slot - current_time,
                    config=config,
                )
                cumulative_harvest_base_j += (
                    env_slot.harvested_power_w - env_slot.base_power_w
                ) * delta_t
                cumulative_compute_j += compute_power_by_slot[inner_slot] * delta_t
            model.addConstr(
                system_state.energy_joule + cumulative_harvest_base_j - cumulative_compute_j
                >= config.energy.e_min_j,
                name=f"energy_floor_{slot}",
            )

        model.setObjective(
            gp.quicksum(
                _assignment_cost(
                    current_time=current_time,
                    task=tasks_and_blocks[block_index][0],
                    block=tasks_and_blocks[block_index][1],
                    core_id=core_id,
                    slot=slot,
                    objective_weights=objective_weights,
                )
                * var
                for (block_index, core_id, slot), var in x.items()
            ),
            gp.GRB.MINIMIZE,
        )

        result = optimize_gurobi_model(model, params=params, model_name="rhc_milp_gurobi_window")
        if not result.has_feasible_solution:
            return result, ()

        plan = []
        for (block_index, core_id, slot), var in x.items():
            if float(var.X) < 0.5:
                continue
            task, block = tasks_and_blocks[block_index]
            plan.append(
                RhcWindowAssignment(
                    time_slot=slot,
                    core_id=core_id,
                    task_id=task.task_id,
                    block_id=block.block_id,
                    duration_slots=block.duration_by_core[core_id],
                    compute_power_w=block.power_by_core[core_id],
                )
            )
        plan.sort(key=lambda item: (item.time_slot, item.core_id, item.task_id, item.block_id))
        return result, tuple(plan)
    except GurobiAdapterError:
        raise
    except Exception as exc:
        gurobi_error = getattr(gp, "GurobiError", None)
        if gurobi_error is not None and isinstance(exc, gurobi_error):
            raise GurobiAdapterError(str(exc)) from exc
        raise


def _assignment_cost(
    *,
    current_time: int,
    task: Task,
    block: TaskBlock,
    core_id: str,
    slot: int,
    objective_weights: RhcObjectiveWeights,
) -> float:
    completion_proxy = slot + block.duration_by_core[core_id]
    response_proxy = max(0, completion_proxy - task.release_time)
    tardiness_proxy = max(0, completion_proxy - task.deadline)
    served_work_reward = task.weight
    receding_penalty = max(0, slot - current_time) * 0.01
    return (
        objective_weights.alpha * task.weight * response_proxy
        + objective_weights.beta * task.weight * tardiness_proxy
        - objective_weights.mu * served_work_reward
        + receding_penalty
    )


@dataclass(frozen=True, slots=True)
class _DefaultEnvironmentView:
    harvested_power_w: float
    base_power_w: float
    ambient_temperature_celsius: float


def _window_slot_or_default(
    *,
    environment_window: Sequence[Any],
    offset: int,
    config: ConfigBundle,
) -> Any:
    if 0 <= offset < len(environment_window):
        return environment_window[offset]
    if environment_window:
        return environment_window[-1]
    return _DefaultEnvironmentView(
        harvested_power_w=0.0,
        base_power_w=config.energy.default_base_power_w,
        ambient_temperature_celsius=config.thermal.ambient_temperature_celsius,
    )


def _fallback_solver_trace(*, current_time: int, status: str) -> SolverTrace:
    return SolverTrace(
        time_slot=current_time,
        solver_name="gurobi",
        status=status,
        solve_time_sec=0.0,
        has_feasible_solution=False,
    )


__all__ = [
    "RhcMilpGurobiError",
    "RhcMilpGurobiScheduler",
    "RhcObjectiveWeights",
    "RhcWindowAssignment",
]
