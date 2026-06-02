"""A lightweight temporal-convolution policy interface."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from tebs.models import Core, Task


class TcnPolicyError(ValueError):
    """Raised when a TCN policy receives invalid input."""


@dataclass(frozen=True, slots=True)
class TcnPolicyConfig:
    """Configuration for the lightweight pure-Python TCN policy."""

    input_feature_dim: int | None = None
    hidden_channels: int = 8
    num_actions: int = 64
    kernel_size: int = 3
    num_layers: int = 2

    def __post_init__(self) -> None:
        if self.input_feature_dim is not None and self.input_feature_dim <= 0:
            raise TcnPolicyError("input_feature_dim must be > 0 when provided.")
        if self.hidden_channels <= 0:
            raise TcnPolicyError("hidden_channels must be > 0.")
        if self.num_actions <= 0:
            raise TcnPolicyError("num_actions must be > 0.")
        if self.kernel_size <= 0:
            raise TcnPolicyError("kernel_size must be > 0.")
        if self.num_layers <= 0:
            raise TcnPolicyError("num_layers must be > 0.")


@dataclass(frozen=True, slots=True)
class TcnActionProposal:
    """One model-proposed action before constraint filtering."""

    core_id: str
    score: float
    is_idle: bool
    task_id: str | None = None
    block_id: int | None = None


@dataclass(slots=True)
class TcnPolicyNetwork:
    """Deterministic TCN-shaped scorer with stable inference semantics.

    The implementation intentionally avoids heavyweight ML dependencies. It
    gives the project a tested policy-network contract that can later be backed
    by PyTorch without changing schedulers or tests.
    """

    config: TcnPolicyConfig

    def predict_logits(self, feature_window: Sequence[Sequence[float]]) -> tuple[float, ...]:
        """Return fixed-size action logits for an H x F feature window."""

        matrix = _validate_feature_window(feature_window, self.config.input_feature_dim)
        hidden = matrix
        for layer_index in range(self.config.num_layers):
            hidden = _temporal_conv(
                hidden,
                hidden_channels=self.config.hidden_channels,
                kernel_size=self.config.kernel_size,
                layer_index=layer_index,
            )
        pooled = _global_average_pool(hidden)
        logits = []
        for action_index in range(self.config.num_actions):
            value = 0.0
            for channel_index, pooled_value in enumerate(pooled):
                sign = -1.0 if (action_index + channel_index) % 2 else 1.0
                value += sign * pooled_value / (channel_index + 1)
            logits.append(math.tanh(value + 0.01 * action_index))
        return tuple(logits)

    def propose_actions(
        self,
        *,
        feature_window: Sequence[Sequence[float]],
        current_time: int,
        tasks: Sequence[Task],
        cores: Sequence[Core],
    ) -> tuple[TcnActionProposal, ...]:
        """Rank idle and task-block actions for each core."""

        if current_time < 0:
            raise TcnPolicyError("current_time must be >= 0.")
        if not cores:
            raise TcnPolicyError("cores must not be empty.")

        logits = self.predict_logits(feature_window)
        proposals: list[TcnActionProposal] = []
        ordinal = 0
        for core in sorted(cores, key=lambda item: item.core_id):
            proposals.append(
                TcnActionProposal(
                    core_id=core.core_id,
                    is_idle=True,
                    score=_logit_at(logits, ordinal) - 0.05,
                )
            )
            ordinal += 1
            for task in sorted(tasks, key=lambda item: (item.release_time, item.deadline, item.task_id)):
                for block in sorted(task.blocks, key=lambda item: item.block_id):
                    if core.core_id not in block.duration_by_core:
                        continue
                    heuristic = _task_block_heuristic(
                        current_time=current_time,
                        task=task,
                        block_id=block.block_id,
                        duration_slots=block.duration_by_core[core.core_id],
                        power_w=block.power_by_core[core.core_id],
                    )
                    proposals.append(
                        TcnActionProposal(
                            core_id=core.core_id,
                            task_id=task.task_id,
                            block_id=block.block_id,
                            is_idle=False,
                            score=_logit_at(logits, ordinal) + heuristic,
                        )
                    )
                    ordinal += 1

        return tuple(sorted(proposals, key=lambda item: (item.core_id, -item.score, item.task_id or "")))


def _validate_feature_window(
    feature_window: Sequence[Sequence[float]],
    expected_dim: int | None,
) -> tuple[tuple[float, ...], ...]:
    if not feature_window:
        raise TcnPolicyError("feature_window must not be empty.")
    matrix = tuple(tuple(float(value) for value in row) for row in feature_window)
    feature_dim = len(matrix[0])
    if feature_dim <= 0:
        raise TcnPolicyError("feature rows must not be empty.")
    if expected_dim is not None and feature_dim != expected_dim:
        raise TcnPolicyError(
            f"feature_dim mismatch: expected {expected_dim}, received {feature_dim}."
        )
    if any(len(row) != feature_dim for row in matrix):
        raise TcnPolicyError("all feature rows must have the same length.")
    return matrix


def _temporal_conv(
    matrix: tuple[tuple[float, ...], ...],
    *,
    hidden_channels: int,
    kernel_size: int,
    layer_index: int,
) -> tuple[tuple[float, ...], ...]:
    rows: list[tuple[float, ...]] = []
    for time_index in range(len(matrix)):
        channels: list[float] = []
        for channel_index in range(hidden_channels):
            acc = 0.0
            weight_norm = 0.0
            for kernel_index in range(kernel_size):
                source_index = max(0, time_index - kernel_index)
                source_row = matrix[source_index]
                for feature_index, value in enumerate(source_row):
                    weight = (
                        (channel_index + 1)
                        * (kernel_index + 1)
                        / ((feature_index + 1) * (layer_index + 1))
                    )
                    acc += value * weight
                    weight_norm += abs(weight)
            channels.append(math.tanh(acc / max(1.0, weight_norm)))
        rows.append(tuple(channels))
    return tuple(rows)


def _global_average_pool(matrix: tuple[tuple[float, ...], ...]) -> tuple[float, ...]:
    channel_count = len(matrix[0])
    return tuple(
        sum(row[channel_index] for row in matrix) / len(matrix)
        for channel_index in range(channel_count)
    )


def _logit_at(logits: Sequence[float], ordinal: int) -> float:
    return logits[ordinal % len(logits)]


def _task_block_heuristic(
    *,
    current_time: int,
    task: Task,
    block_id: int,
    duration_slots: int,
    power_w: float,
) -> float:
    slack = max(0, task.deadline - current_time)
    urgency = task.weight / (1.0 + slack)
    work_cost = 0.01 * duration_slots + 0.001 * power_w
    block_order_bias = -0.001 * block_id
    release_bias = 0.05 if current_time >= task.release_time else -0.2
    return urgency + release_bias - work_cost + block_order_bias


__all__ = [
    "TcnActionProposal",
    "TcnPolicyConfig",
    "TcnPolicyError",
    "TcnPolicyNetwork",
]
