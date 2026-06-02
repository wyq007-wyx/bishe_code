from __future__ import annotations

import pytest

from tebs.learning.tcn_policy import TcnPolicyConfig, TcnPolicyError, TcnPolicyNetwork
from tebs.models import Core, Task


def _feature_window() -> tuple[tuple[float, ...], ...]:
    return (
        (1.0, 2.0, 3.0),
        (2.0, 3.0, 4.0),
        (3.0, 4.0, 5.0),
    )


def test_tcn_policy_predict_logits_has_stable_shape_and_values() -> None:
    policy = TcnPolicyNetwork(
        TcnPolicyConfig(input_feature_dim=3, hidden_channels=4, num_actions=6)
    )

    first = policy.predict_logits(_feature_window())
    second = policy.predict_logits(_feature_window())

    assert len(first) == 6
    assert first == second
    assert all(-1.0 <= value <= 1.0 for value in first)


def test_tcn_policy_validates_feature_shape() -> None:
    policy = TcnPolicyNetwork(TcnPolicyConfig(input_feature_dim=4, num_actions=3))

    with pytest.raises(TcnPolicyError, match="feature_dim mismatch"):
        policy.predict_logits(_feature_window())


def test_tcn_policy_proposes_idle_and_task_block_actions_per_core() -> None:
    policy = TcnPolicyNetwork(TcnPolicyConfig(input_feature_dim=3, num_actions=8))
    cores = (Core(core_id="hp_0", core_type="high_performance"),)
    tasks = [
        Task.single_block(
            task_id="T0",
            release_time=0,
            deadline=5,
            weight=2.0,
            duration_by_core={"hp_0": 1},
            power_by_core={"hp_0": 2.0},
            core_ids=("hp_0",),
        )
    ]

    proposals = policy.propose_actions(
        feature_window=_feature_window(),
        current_time=0,
        tasks=tasks,
        cores=cores,
    )

    assert {proposal.core_id for proposal in proposals} == {"hp_0"}
    assert any(proposal.is_idle for proposal in proposals)
    assert any(proposal.task_id == "T0" for proposal in proposals)
