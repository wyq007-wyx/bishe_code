from __future__ import annotations

import pytest

from tebs.models import Core
from tebs.task_factory import (
    TaskFactoryError,
    count_high_power_tasks,
    generate_task_set,
    high_power_task_ratio,
    task_type_id_from_task,
)


def test_fixed_seed_generates_reproducible_task_set() -> None:
    tasks_a = generate_task_set("S2_mixed", num_tasks=12, random_seed=123)
    tasks_b = generate_task_set("S2_mixed", num_tasks=12, random_seed=123)

    assert tasks_a == tasks_b
    assert len(tasks_a) == 12


def test_stress_scenario_has_more_high_power_tasks_than_light_scenario() -> None:
    light_tasks = generate_task_set("S1_light", num_tasks=30, random_seed=7)
    stress_tasks = generate_task_set("S3_stress", num_tasks=30, random_seed=7)

    assert count_high_power_tasks(stress_tasks) > count_high_power_tasks(light_tasks)
    assert high_power_task_ratio(stress_tasks) > high_power_task_ratio(light_tasks)


def test_generated_tasks_have_valid_blocks_and_positive_core_profiles() -> None:
    tasks = generate_task_set("S2_mixed", num_tasks=20, random_seed=11)

    for task in tasks:
        assert task.blocks
        assert task.release_time <= task.deadline
        assert task.weight > 0
        for block in task.blocks:
            assert block.release_time == task.release_time
            assert block.deadline == task.deadline
            assert block.duration_by_core.keys() == block.power_by_core.keys()
            assert set(block.duration_by_core.keys()) == set(task.core_ids)
            assert all(duration > 0 for duration in block.duration_by_core.values())
            assert all(power > 0 for power in block.power_by_core.values())


def test_multistage_predecessors_form_a_legal_chain() -> None:
    tasks = generate_task_set("mixed", num_tasks=16, random_seed=19)

    assert any(len(task.blocks) > 1 for task in tasks)
    for task in tasks:
        for block in task.blocks:
            if block.block_id == 0:
                assert block.predecessor_block_id is None
            else:
                assert block.predecessor_block_id == block.block_id - 1


def test_custom_core_specs_are_used_in_all_task_blocks() -> None:
    cores = (
        Core(core_id="hp_custom", core_type="high_performance"),
        Core(core_id="lp_custom", core_type="low_power"),
    )

    tasks = generate_task_set("S1", num_tasks=5, random_seed=5, core_specs=cores)

    for task in tasks:
        assert task.core_ids == ("hp_custom", "lp_custom")
        for block in task.blocks:
            assert set(block.duration_by_core) == {"hp_custom", "lp_custom"}
            assert set(block.power_by_core) == {"hp_custom", "lp_custom"}


def test_task_type_id_can_be_read_from_generated_task_id() -> None:
    task = generate_task_set("S1_light", num_tasks=1, random_seed=3)[0]

    assert task_type_id_from_task(task).startswith("T")


def test_unknown_scenario_raises_clear_error() -> None:
    with pytest.raises(TaskFactoryError, match="Unknown scenario_name"):
        generate_task_set("does_not_exist", num_tasks=1, random_seed=1)
