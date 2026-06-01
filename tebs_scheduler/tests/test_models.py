from __future__ import annotations

import pytest

from tebs.models import ScheduleDecision, SystemState, Task, TaskBlock

"""
    测试TaskBlock,Task,SystemState,ScheduleDecision参数合法性校验
    TaskBlock：任务块
    Task：任务，由多个任务块组成
    SystemState：系统状态
    ScheduleDecision：调度决策
    作用：测试模型层是否能防止非法输入
"""

def _build_block(#构造默认任务块
    *,
    block_id: int,
    predecessor_block_id: int | None = None,
    duration_by_core: dict[str, int] | None = None,
    power_by_core: dict[str, float] | None = None,
    release_time: int = 0,
    deadline: int = 10,
) -> TaskBlock:
    return TaskBlock(
        task_id="T1", # 该任务块属于的任务
        block_id=block_id, # 该任务块的id
        release_time=release_time, #该任务块的到达时间
        deadline=deadline,# 该任务块的截止时间
        weight=3.0, # 该任务块的权重
        duration_by_core=duration_by_core or {"c0": 5, "c1": 7}, # 该任务块在高性能核与低性能核的运行时隙长度
        power_by_core=power_by_core or {"c0": 30.0, "c1": 20.0}, # 该任务块在高性能核上的运行功率和在低性能核上的运行功率
        predecessor_block_id=predecessor_block_id, # 前驱block_id
    )


def test_task_block_core_maps_must_match() -> None:
    # 检查 duration_by_core 和 power_by_core 中的核心 ID 必须一致。
    with pytest.raises(ValueError, match="same core IDs"):
        _build_block(
            block_id=0,
            duration_by_core={"c0": 5, "c1": 7},
            power_by_core={"c0": 30.0},
        ) # 执行时间里面有 c0 和 c1, 但是功耗里面只有 c0,缺少 c1. 这是非法的.


def test_release_time_must_not_exceed_deadline() -> None:
    # 任务块的释放时间不能晚于截止时间。
    with pytest.raises(ValueError, match="release_time must be <="):
        _build_block(block_id=0, release_time=11, deadline=10)
    # 任务的释放时间不晚于截止时间
    with pytest.raises(ValueError, match="release_time must be <="):
        Task(
            task_id="T1",
            release_time=20,
            deadline=10,
            weight=1.0,
            blocks=[_build_block(block_id=0)],
        )


def test_predecessor_order_must_be_legal() -> None:
    # 检查任务块之间的依赖关系是否合法
    # 前驱块id必须小于当前块id
    with pytest.raises(ValueError, match="smaller than block_id"):
        _build_block(block_id=1, predecessor_block_id=1)

    # 前驱块必须真实存在
    with pytest.raises(ValueError, match="existing earlier block"):
        Task(
            task_id="T1",
            release_time=0,
            deadline=20,
            weight=2.0,
            blocks=[_build_block(block_id=1, predecessor_block_id=0)],
        )

    # 合法任务
    task = Task(
        task_id="T1",
        release_time=0,
        deadline=20,
        weight=2.0,
        blocks=[
            _build_block(block_id=0),
            _build_block(block_id=1, predecessor_block_id=0),
        ],
    )
    assert len(task.blocks) == 2


def test_task_blocks_cover_all_cores_when_core_ids_given() -> None:
    # 如果 Task 指定了可用核心列表 core_ids，那么每个任务块都必须覆盖这些核心。
    # 合法情况
    Task(
        task_id="T1",
        release_time=0,
        deadline=20,
        weight=1.0,
        blocks=[_build_block(block_id=0)],
        core_ids=("c0", "c1"), # 指定了c0,c1
    )
    # 非法情况
    with pytest.raises(ValueError, match="cover all cores"):
        Task(
            task_id="T1",
            release_time=0,
            deadline=20,
            weight=1.0,
            blocks=[
                _build_block(
                    block_id=0,
                    duration_by_core={"c0": 5},
                    power_by_core={"c0": 20.0},
                ) # 构建任务块时只有c0的数据
            ],
            core_ids=("c0", "c1"), # task可用核心是c0和c1
        )


def test_system_state_has_required_fields() -> None:
    # 检查 SystemState 是否能正确保存系统状态字段。
    state = SystemState(time_slot=0, energy_joule=8000.0, temperature_celsius=35.0)
    assert state.time_slot == 0
    assert state.energy_joule == 8000.0
    assert state.temperature_celsius == 35.0


def test_schedule_decision_supports_run_and_idle() -> None:
    idle_decision = ScheduleDecision.idle(time_slot=3, core_id="c0")
    assert idle_decision.is_idle is True
    assert idle_decision.task_id is None
    assert idle_decision.block_id is None

    run_decision = ScheduleDecision.run(
        time_slot=3,
        core_id="c1",
        task_id="T1",
        block_id=2,
        frequency_level="high",
    )
    assert run_decision.is_idle is False
    assert run_decision.task_id == "T1"
    assert run_decision.block_id == 2

    with pytest.raises(ValueError, match="must include task_id and block_id"):
        ScheduleDecision(time_slot=0, core_id="c0", is_idle=False, task_id="T1")
