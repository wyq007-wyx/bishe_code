"""Constraint filtering for learning-model scheduling actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from tebs.config import ConfigBundle, load_config
from tebs.energy_model import is_energy_feasible
from tebs.environment import EnvironmentSlot
from tebs.models import Core, ScheduleDecision, SystemState, Task, TaskBlock
from tebs.scheduler_base import make_idle_decisions
from tebs.thermal_model import is_thermal_feasible

from .tcn_policy import TcnActionProposal


class ConstraintFilterError(ValueError):
    """Raised when action filtering inputs are invalid."""


@dataclass(frozen=True, slots=True)
class RejectedAction:
    """One rejected proposal and the reason it was not executable."""

    proposal: TcnActionProposal
    reason: str


@dataclass(frozen=True, slots=True)
class ConstraintFilterResult:
    """Filtered decisions and diagnostic ratios."""

    decisions: tuple[ScheduleDecision, ...]
    accepted_proposals: tuple[TcnActionProposal, ...]
    rejected_actions: tuple[RejectedAction, ...]
    feasible_action_rate: float
    retention_rate: float


@dataclass(slots=True)
class ConstraintFilter:
    """Select safe executable actions from ranked TCN proposals."""

    config: ConfigBundle | None = None

    def select_feasible_decisions(
        self,
        *,
        current_time: int,
        proposals: Sequence[TcnActionProposal],
        tasks: Sequence[Task],
        cores: Sequence[Core],
        system_state: SystemState,
        environment_window: Sequence[EnvironmentSlot],
        completed_blocks: set[tuple[str, int]] | frozenset[tuple[str, int]] | None = None,
    ) -> ConstraintFilterResult:
        """Return at most one feasible decision for every core."""

        if current_time < 0:
            raise ConstraintFilterError("current_time must be >= 0.")
        if not cores:
            raise ConstraintFilterError("cores must not be empty.")

        cfg = self.config or load_config()
        completed = completed_blocks or set()
        task_by_id = {task.task_id: task for task in tasks}
        core_by_id = {core.core_id: core for core in cores}
        if len(core_by_id) != len(cores):
            raise ConstraintFilterError("cores must contain unique core_id values.")

        proposals_by_core = _group_ranked_proposals(proposals)
        decisions: list[ScheduleDecision] = []
        accepted: list[TcnActionProposal] = []
        rejected: list[RejectedAction] = []
        running_blocks: set[tuple[str, int]] = set()
        selected_compute_power_w = 0.0
        checked_non_idle = 0
        feasible_non_idle = 0

        for core in cores:
            chosen: TcnActionProposal | None = None
            for proposal in proposals_by_core.get(core.core_id, ()):
                if proposal.is_idle:
                    if chosen is None:
                        chosen = proposal
                    continue
                checked_non_idle += 1
                reason = self._rejection_reason(
                    proposal=proposal,
                    core_by_id=core_by_id,
                    task_by_id=task_by_id,
                    current_time=current_time,
                    system_state=system_state,
                    environment_window=environment_window,
                    completed_blocks=completed,
                    running_blocks=running_blocks,
                    selected_compute_power_w=selected_compute_power_w,
                    config=cfg,
                )
                if reason is not None:
                    rejected.append(RejectedAction(proposal=proposal, reason=reason))
                    continue
                chosen = proposal
                feasible_non_idle += 1
                break

            if chosen is None:
                chosen = TcnActionProposal(core_id=core.core_id, score=0.0, is_idle=True)

            if chosen.is_idle:
                decisions.append(ScheduleDecision.idle(time_slot=current_time, core_id=core.core_id))
                accepted.append(chosen)
                continue

            block = _proposal_block(chosen, task_by_id)
            assert block is not None
            decisions.append(
                ScheduleDecision.run_block(
                    time_slot=current_time,
                    core_id=core.core_id,
                    task_id=chosen.task_id or "",
                    block_id=block.block_id,
                )
            )
            accepted.append(chosen)
            running_blocks.add((chosen.task_id or "", block.block_id))
            selected_compute_power_w += block.power_by_core[core.core_id]

        feasible_action_rate = (
            feasible_non_idle / checked_non_idle if checked_non_idle > 0 else 1.0
        )
        retention_rate = len([item for item in accepted if not item.is_idle]) / max(1, len(cores))
        return ConstraintFilterResult(
            decisions=tuple(decisions),
            accepted_proposals=tuple(accepted),
            rejected_actions=tuple(rejected),
            feasible_action_rate=feasible_action_rate,
            retention_rate=retention_rate,
        )

    def _rejection_reason(
        self,
        *,
        proposal: TcnActionProposal,
        core_by_id: dict[str, Core],
        task_by_id: dict[str, Task],
        current_time: int,
        system_state: SystemState,
        environment_window: Sequence[EnvironmentSlot],
        completed_blocks: set[tuple[str, int]] | frozenset[tuple[str, int]],
        running_blocks: set[tuple[str, int]],
        selected_compute_power_w: float,
        config: ConfigBundle,
    ) -> str | None:
        if proposal.core_id not in core_by_id:
            return "unknown_core"
        if proposal.task_id is None:
            return "missing_task"
        task = task_by_id.get(proposal.task_id)
        if task is None:
            return "unknown_task"
        if current_time < task.release_time:
            return "task_not_released"
        block = _proposal_block(proposal, task_by_id)
        if block is None:
            return "unknown_block"
        if current_time < block.release_time:
            return "block_not_released"
        block_key = (task.task_id, block.block_id)
        if block_key in completed_blocks:
            return "block_completed"
        if block_key in running_blocks:
            return "block_already_assigned"
        if proposal.core_id not in block.duration_by_core:
            return "core_incompatible"
        predecessor = block.predecessor_block_id
        if predecessor is not None and (task.task_id, predecessor) not in completed_blocks:
            return "predecessor_incomplete"
        env_slot = _first_environment_slot(environment_window)
        compute_power_w = selected_compute_power_w + block.power_by_core[proposal.core_id]
        if not is_energy_feasible(
            energy_prev_j=system_state.energy_joule,
            harvested_power_w=env_slot.harvested_power_w,
            base_power_w=env_slot.base_power_w,
            compute_power_w=compute_power_w,
            delta_t_seconds=config.simulation.delta_t_seconds,
            energy_min_j=config.energy.e_min_j,
        ):
            return "energy_infeasible"
        if not is_thermal_feasible(
            temperature_prev_celsius=system_state.temperature_celsius,
            compute_power_w=compute_power_w,
            base_power_w=env_slot.base_power_w,
            delta_t_seconds=config.simulation.delta_t_seconds,
            temperature_max_celsius=config.thermal.t_max_celsius,
            thermal_capacity_j_per_c=config.thermal.thermal_capacity_j_per_c,
            cooling_coeff=config.thermal.cooling_coeff,
            ambient_temperature_celsius=env_slot.ambient_temperature_celsius,
        ):
            return "thermal_infeasible"
        return None


def safe_idle_filter_result(*, current_time: int, cores: Sequence[Core]) -> ConstraintFilterResult:
    """Return a filter result that idles all cores."""

    decisions = make_idle_decisions(current_time=current_time, cores=cores)
    proposals = tuple(
        TcnActionProposal(core_id=decision.core_id, score=0.0, is_idle=True)
        for decision in decisions
    )
    return ConstraintFilterResult(
        decisions=decisions,
        accepted_proposals=proposals,
        rejected_actions=(),
        feasible_action_rate=1.0,
        retention_rate=0.0,
    )


def _group_ranked_proposals(
    proposals: Sequence[TcnActionProposal],
) -> dict[str, tuple[TcnActionProposal, ...]]:
    grouped: dict[str, list[TcnActionProposal]] = {}
    for proposal in proposals:
        grouped.setdefault(proposal.core_id, []).append(proposal)
    return {
        core_id: tuple(sorted(items, key=lambda item: (-item.score, item.is_idle)))
        for core_id, items in grouped.items()
    }


def _proposal_block(
    proposal: TcnActionProposal,
    task_by_id: dict[str, Task],
) -> TaskBlock | None:
    if proposal.task_id is None:
        return None
    task = task_by_id.get(proposal.task_id)
    if task is None:
        return None
    block_id = 0 if proposal.block_id is None else proposal.block_id
    return next((block for block in task.blocks if block.block_id == block_id), None)


@dataclass(frozen=True, slots=True)
class _DefaultEnvironmentSlot:
    harvested_power_w: float
    base_power_w: float
    ambient_temperature_celsius: float


def _first_environment_slot(environment_window: Sequence[EnvironmentSlot]):
    if environment_window:
        return environment_window[0]
    cfg = load_config()
    return _DefaultEnvironmentSlot(
        harvested_power_w=0.0,
        base_power_w=cfg.energy.default_base_power_w,
        ambient_temperature_celsius=cfg.thermal.ambient_temperature_celsius,
    )


__all__ = [
    "ConstraintFilter",
    "ConstraintFilterError",
    "ConstraintFilterResult",
    "RejectedAction",
    "safe_idle_filter_result",
]
