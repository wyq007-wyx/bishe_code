"""FCFS-Gurobi strong baseline scheduler."""

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


class FcfsGurobiError(ValueError):
    """Raised when FCFS-Gurobi inputs are invalid."""


@dataclass(frozen=True, slots=True)
class GurobiWindowAssignment:
    """One task-block/core assignment in a Gurobi window plan."""

    time_slot: int
    core_id: str
    task_id: str
    block_id: int
    duration_slots: int
    compute_power_w: float


@dataclass(slots=True)
class FcfsGurobiScheduler(BaseScheduler):
    """Fixed FCFS queue; Gurobi selects the current core assignment."""

    config: ConfigBundle | None = None
    gurobi_params: GurobiSolveParams | None = None
    fallback_to_idle_on_error: bool = True
    name: str = "fcfs_gurobi"
    _remaining_work_slots: dict[tuple[str, int], float] = field(default_factory=dict)
    _completed_blocks: set[tuple[str, int]] = field(default_factory=set)
    _completed_tasks: set[str] = field(default_factory=set)
    last_window_plan: tuple[GurobiWindowAssignment, ...] = ()
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

        candidate = self._select_fcfs_candidate(
            current_time=current_time,
            tasks=tasks,
            cores=cores,
        )
        if candidate is None:
            return SchedulerResult(make_idle_decisions(current_time=current_time, cores=cores))

        task, block = candidate
        try:
            result, chosen_core = _solve_fcfs_current_slot(
                current_time=current_time,
                task=task,
                block=block,
                cores=cores,
                system_state=system_state,
                environment_window=environment_window,
                config=self._config(),
                params=self._params(),
            )
        except (GurobiUnavailableError, GurobiAdapterError) as exc:
            if not self.fallback_to_idle_on_error:
                raise FcfsGurobiError(str(exc)) from exc
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
        if chosen_core is None:
            return SchedulerResult(
                make_idle_decisions(current_time=current_time, cores=cores),
                solver_trace=solver_trace,
            )

        decision = ScheduleDecision.run_block(
            time_slot=current_time,
            core_id=chosen_core.core_id,
            task_id=task.task_id,
            block_id=block.block_id,
        )
        self._advance_progress(task=task, block=block, core=chosen_core)
        self.last_window_plan = (
            GurobiWindowAssignment(
                time_slot=current_time,
                core_id=chosen_core.core_id,
                task_id=task.task_id,
                block_id=block.block_id,
                duration_slots=block.duration_by_core[chosen_core.core_id],
                compute_power_w=block.power_by_core[chosen_core.core_id],
            ),
        )
        decisions = [decision]
        decisions.extend(
            ScheduleDecision.idle(time_slot=current_time, core_id=core.core_id)
            for core in cores
            if core.core_id != chosen_core.core_id
        )
        return SchedulerResult(decisions=decisions, solver_trace=solver_trace)

    def remaining_work_for(self, task_id: str, block_id: int = 0) -> float | None:
        """Return scheduler-side remaining work for tests and inspection."""

        return self._remaining_work_slots.get((task_id, block_id))

    def _config(self) -> ConfigBundle:
        return self.config or load_config()

    def _params(self) -> GurobiSolveParams:
        return self.gurobi_params or GurobiSolveParams.from_config(self._config())

    def _select_fcfs_candidate(
        self,
        *,
        current_time: int,
        tasks: Sequence[Task],
        cores: Sequence[Core],
    ) -> tuple[Task, TaskBlock] | None:
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
            if not any(core.core_id in block.duration_by_core for core in cores):
                return None
            return task, block
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

    def _advance_progress(self, *, task: Task, block: TaskBlock, core: Core) -> None:
        key = (task.task_id, block.block_id)
        if key not in self._remaining_work_slots:
            self._remaining_work_slots[key] = float(block.duration_by_core[core.core_id])
        self._remaining_work_slots[key] -= 1.0
        if self._remaining_work_slots[key] <= 0:
            self._completed_blocks.add(key)
            if all((task.task_id, item.block_id) in self._completed_blocks for item in task.blocks):
                self._completed_tasks.add(task.task_id)


def _solve_fcfs_current_slot(
    *,
    current_time: int,
    task: Task,
    block: TaskBlock,
    cores: Sequence[Core],
    system_state: SystemState,
    environment_window: Sequence[Any],
    config: ConfigBundle,
    params: GurobiSolveParams,
) -> tuple[GurobiSolveResult, Core | None]:
    gp = require_gurobi()
    try:
        compatible_cores = [core for core in cores if core.core_id in block.duration_by_core]
        if not compatible_cores:
            return (
                GurobiSolveResult(
                    status=STATUS_INFEASIBLE,
                    raw_status=None,
                    solve_time_sec=0.0,
                    has_feasible_solution=False,
                    model_name="fcfs_gurobi_current_slot",
                ),
                None,
            )

        model = gp.Model("fcfs_gurobi_current_slot")
        x = {
            core.core_id: model.addVar(vtype=gp.GRB.BINARY, name=f"x_{core.core_id}")
            for core in compatible_cores
        }
        model.addConstr(gp.quicksum(x.values()) == 1, name="assign_fcfs_head")
        compute_power = gp.quicksum(
            block.power_by_core[core.core_id] * x[core.core_id] for core in compatible_cores
        )
        env_slot = _slot_or_config_default(environment_window, config)
        delta_t = config.simulation.delta_t_seconds
        available_energy_j = (
            system_state.energy_joule
            + env_slot.harvested_power_w * delta_t
            - env_slot.base_power_w * delta_t
            - config.energy.e_min_j
        )
        model.addConstr(compute_power * delta_t <= available_energy_j, name="energy_floor")
        thermal_budget_j = compute_thermal_budget_j(
            temperature_prev_celsius=system_state.temperature_celsius,
            base_power_w=env_slot.base_power_w,
            delta_t_seconds=delta_t,
            temperature_max_celsius=config.thermal.t_max_celsius,
            thermal_capacity_j_per_c=config.thermal.thermal_capacity_j_per_c,
            cooling_coeff=config.thermal.cooling_coeff,
            ambient_temperature_celsius=env_slot.ambient_temperature_celsius,
        )
        model.addConstr(compute_power * delta_t <= thermal_budget_j, name="thermal_budget")
        model.setObjective(
            gp.quicksum(
                (
                    block.duration_by_core[core.core_id]
                    + 0.001 * block.power_by_core[core.core_id]
                )
                * x[core.core_id]
                for core in compatible_cores
            ),
            gp.GRB.MINIMIZE,
        )

        result = optimize_gurobi_model(model, params=params, model_name="fcfs_gurobi_current_slot")
        if not result.has_feasible_solution:
            return result, None
        chosen = max(compatible_cores, key=lambda core: float(x[core.core_id].X))
        if float(x[chosen.core_id].X) < 0.5:
            return result, None
        return result, chosen
    except GurobiAdapterError:
        raise
    except Exception as exc:
        gurobi_error = getattr(gp, "GurobiError", None)
        if gurobi_error is not None and isinstance(exc, gurobi_error):
            raise GurobiAdapterError(str(exc)) from exc
        raise


@dataclass(frozen=True, slots=True)
class _DefaultEnvironmentView:
    harvested_power_w: float
    base_power_w: float
    ambient_temperature_celsius: float


def _slot_or_config_default(
    environment_window: Sequence[Any],
    config: ConfigBundle,
) -> Any:
    if environment_window:
        return environment_window[0]
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
    "FcfsGurobiError",
    "FcfsGurobiScheduler",
    "FcfSGurobiScheduler",
    "GurobiWindowAssignment",
]

FcfSGurobiScheduler = FcfsGurobiScheduler
