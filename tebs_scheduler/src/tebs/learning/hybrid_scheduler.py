"""TCN and Hybrid-Gurobi-TCN schedulers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from tebs.config import ConfigBundle, load_config
from tebs.models import Core, ScheduleDecision, SolverTrace, SystemState, Task, TaskBlock
from tebs.rhc_milp_gurobi import RhcMilpGurobiError, RhcMilpGurobiScheduler
from tebs.scheduler_base import BaseScheduler, SchedulerResult

from .constraint_filter import ConstraintFilter, ConstraintFilterResult, safe_idle_filter_result
from .dataset_builder import build_feature_window
from .tcn_policy import TcnPolicyConfig, TcnPolicyNetwork


@dataclass(slots=True)
class _ProgressTracker:
    remaining_work_slots: dict[tuple[str, int], float] = field(default_factory=dict)
    completed_blocks: set[tuple[str, int]] = field(default_factory=set)
    completed_tasks: set[str] = field(default_factory=set)

    def advance(
        self,
        *,
        decisions: Sequence[ScheduleDecision],
        tasks: Sequence[Task],
    ) -> None:
        task_by_id = {task.task_id: task for task in tasks}
        for decision in decisions:
            if decision.is_idle or decision.task_id is None:
                continue
            task = task_by_id[decision.task_id]
            block = _decision_block(task=task, decision=decision)
            key = (task.task_id, block.block_id)
            if key in self.completed_blocks:
                continue
            if key not in self.remaining_work_slots:
                self.remaining_work_slots[key] = float(block.duration_by_core[decision.core_id])
            self.remaining_work_slots[key] -= 1.0
            if self.remaining_work_slots[key] <= 0:
                self.completed_blocks.add(key)
                if all((task.task_id, item.block_id) in self.completed_blocks for item in task.blocks):
                    self.completed_tasks.add(task.task_id)


@dataclass(slots=True)
class TcnScheduler(BaseScheduler):
    """Learning-only scheduler using TCN proposals plus constraint filtering."""

    config: ConfigBundle | None = None
    policy: TcnPolicyNetwork | None = None
    constraint_filter: ConstraintFilter | None = None
    max_tasks: int | None = None
    name: str = "tcn_scheduler"
    last_filter_result: ConstraintFilterResult | None = None
    _progress: _ProgressTracker = field(default_factory=_ProgressTracker)

    def decide(
        self,
        *,
        current_time: int,
        tasks: Sequence[Task],
        cores: Sequence[Core],
        system_state: SystemState,
        environment_window: Sequence[Any],
    ) -> SchedulerResult:
        cfg = self._config()
        if not tasks:
            self.last_filter_result = safe_idle_filter_result(current_time=current_time, cores=cores)
            return SchedulerResult(self.last_filter_result.decisions)

        feature_window = build_feature_window(
            current_time=current_time,
            tasks=tasks,
            cores=cores,
            system_state=system_state,
            environment_window=environment_window,
            horizon_slots=max(1, min(cfg.simulation.horizon_slots, len(environment_window) or 1)),
            max_tasks=self.max_tasks or len(tasks),
            completed_blocks=self._progress.completed_blocks,
        )
        proposals = self._policy().propose_actions(
            feature_window=feature_window,
            current_time=current_time,
            tasks=tasks,
            cores=cores,
        )
        result = self._filter().select_feasible_decisions(
            current_time=current_time,
            proposals=proposals,
            tasks=tasks,
            cores=cores,
            system_state=system_state,
            environment_window=environment_window,
            completed_blocks=self._progress.completed_blocks,
        )
        self.last_filter_result = result
        self._progress.advance(decisions=result.decisions, tasks=tasks)
        return SchedulerResult(result.decisions)

    def remaining_work_for(self, task_id: str, block_id: int = 0) -> float | None:
        """Return scheduler-side remaining work for tests and inspection."""

        return self._progress.remaining_work_slots.get((task_id, block_id))

    def _config(self) -> ConfigBundle:
        return self.config or load_config()

    def _policy(self) -> TcnPolicyNetwork:
        if self.policy is None:
            self.policy = TcnPolicyNetwork(
                TcnPolicyConfig(input_feature_dim=None, hidden_channels=8, num_actions=128)
            )
        return self.policy

    def _filter(self) -> ConstraintFilter:
        if self.constraint_filter is None:
            self.constraint_filter = ConstraintFilter(config=self._config())
        return self.constraint_filter


@dataclass(slots=True)
class HybridGurobiTcnScheduler(BaseScheduler):
    """Prefer Gurobi when feasible, otherwise fall back to TCN + filtering."""

    config: ConfigBundle | None = None
    primary_scheduler: BaseScheduler | None = None
    tcn_scheduler: TcnScheduler | None = None
    name: str = "hybrid_gurobi_tcn"
    last_primary_result: SchedulerResult | None = None
    last_filter_result: ConstraintFilterResult | None = None
    last_fallback_used: bool = False
    _progress: _ProgressTracker = field(default_factory=_ProgressTracker)

    def decide(
        self,
        *,
        current_time: int,
        tasks: Sequence[Task],
        cores: Sequence[Core],
        system_state: SystemState,
        environment_window: Sequence[Any],
    ) -> SchedulerResult:
        self.last_fallback_used = False
        self.last_filter_result = None
        primary_trace: SolverTrace | None = None

        try:
            primary_result = self._primary().decide(
                current_time=current_time,
                tasks=tasks,
                cores=cores,
                system_state=system_state,
                environment_window=environment_window,
            )
            self.last_primary_result = primary_result
            primary_trace = primary_result.solver_trace
            if _primary_result_is_usable(primary_result):
                self._progress.advance(decisions=primary_result.decisions, tasks=tasks)
                return primary_result
        except RhcMilpGurobiError as exc:
            primary_trace = SolverTrace(
                time_slot=current_time,
                solver_name="gurobi",
                status=f"PRIMARY_ERROR:{type(exc).__name__}",
                solve_time_sec=0.0,
                has_feasible_solution=False,
            )

        fallback = self._fallback()
        fallback._progress = self._progress
        fallback_result = fallback.decide(
            current_time=current_time,
            tasks=tasks,
            cores=cores,
            system_state=system_state,
            environment_window=environment_window,
        )
        self._progress = fallback._progress
        self.last_filter_result = fallback.last_filter_result
        self.last_fallback_used = True
        return SchedulerResult(decisions=fallback_result.decisions, solver_trace=primary_trace)

    def remaining_work_for(self, task_id: str, block_id: int = 0) -> float | None:
        """Return hybrid-side remaining work for tests and inspection."""

        return self._progress.remaining_work_slots.get((task_id, block_id))

    def _config(self) -> ConfigBundle:
        return self.config or load_config()

    def _primary(self) -> BaseScheduler:
        if self.primary_scheduler is None:
            self.primary_scheduler = RhcMilpGurobiScheduler(config=self._config())
        return self.primary_scheduler

    def _fallback(self) -> TcnScheduler:
        if self.tcn_scheduler is None:
            self.tcn_scheduler = TcnScheduler(config=self._config())
        return self.tcn_scheduler


def _primary_result_is_usable(result: SchedulerResult) -> bool:
    has_non_idle = any(not decision.is_idle for decision in result.decisions)
    trace = result.solver_trace
    solver_ok = trace is None or trace.has_feasible_solution
    return has_non_idle and solver_ok


def _decision_block(*, task: Task, decision: ScheduleDecision) -> TaskBlock:
    if decision.block_id is not None:
        return next(block for block in task.blocks if block.block_id == decision.block_id)
    return next(block for block in sorted(task.blocks, key=lambda item: item.block_id))


__all__ = [
    "HybridGurobiTcnScheduler",
    "TcnScheduler",
]
