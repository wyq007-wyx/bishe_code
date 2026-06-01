from __future__ import annotations

import pytest

from tebs.models import (
    HIGH_PERFORMANCE_CORE_TYPE,
    LOW_POWER_CORE_TYPE,
    ScheduleDecision,
    SystemState,
    Task,
    TaskBlock,
    build_cores_from_type_counts,
)

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
        duration_by_core=duration_by_core or {"hp_0": 5, "lp_0": 7}, # 该任务块在高性能核与低性能核的运行时隙长度
        power_by_core=power_by_core or {"hp_0": 30.0, "lp_0": 20.0}, # 该任务块在高性能核上的运行功率和在低性能核上的运行功率
        predecessor_block_id=predecessor_block_id, # 前驱block_id
    )


def test_task_block_core_maps_must_match() -> None:
    """验证任务块中时长映射与功耗映射的核心集合必须一致。"""
    # 检查 duration_by_core 和 power_by_core 中的核心 ID 必须一致
    with pytest.raises(ValueError, match="same core IDs"):
        _build_block(
            block_id=0,
            duration_by_core={"hp_0": 5, "lp_0": 7},
            power_by_core={"hp_0": 30.0},
        ) # 执行时间里面有 hp_0 和 lp_0, 但是功耗里面只有 hp_0,缺少 lp_0. 这是非法的.


def test_release_time_must_not_exceed_deadline() -> None:
    """验证任务块和任务的 release_time 不能晚于 deadline。"""
    # 任务块的释放时间不能晚于截止时间
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
    """验证任务块前驱关系必须存在且满足拓扑顺序。"""
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


def test_task_block_core_set_is_subset_of_task_core_ids() -> None:
    """验证任务块可用核心集合必须是任务 core_ids 的子集。"""
    # Task 指定 core_ids 时，任务块可以只使用其中部分核心（子集语义）。
    task = Task(
        task_id="T1",
        release_time=0,
        deadline=20,
        weight=1.0,
        blocks=[
            _build_block(
                block_id=0,
                duration_by_core={"hp_0": 5},
                power_by_core={"hp_0": 20.0},
            ),
            _build_block(
                block_id=1,
                predecessor_block_id=0,
                duration_by_core={"lp_0": 7},
                power_by_core={"lp_0": 18.0},
            ),
        ],
        core_ids=("hp_0", "lp_0"),
    )
    assert len(task.blocks) == 2

    # 非法：任务块使用了 core_ids 之外的核心。
    with pytest.raises(ValueError, match="subset of core_ids"):
        Task(
            task_id="T1",
            release_time=0,
            deadline=20,
            weight=1.0,
            blocks=[
                _build_block(
                    block_id=0,
                    duration_by_core={"hp_2": 5},
                    power_by_core={"hp_2": 20.0},
                )
            ],
            core_ids=("hp_0", "lp_0"),
        )


def test_task_can_optionally_require_uniform_block_core_set() -> None:
    """验证可选的一致性约束：任务内所有块核心集合可要求一致。"""
    # 多阶段任务默认允许块使用不同核心集合。
    Task(
        task_id="T1",
        release_time=0,
        deadline=20,
        weight=1.0,
        blocks=[
            _build_block(block_id=0, duration_by_core={"hp_0": 5}, power_by_core={"hp_0": 20.0}),
            _build_block(
                block_id=1,
                predecessor_block_id=0,
                duration_by_core={"lp_0": 6},
                power_by_core={"lp_0": 16.0},
            ),
        ],
        core_ids=("hp_0", "lp_0"),
        require_uniform_block_core_ids=False,
    )

    # 若显式要求一致，则核心集合不一致应报错。
    with pytest.raises(ValueError, match="require_uniform_block_core_ids=True"):
        Task(
            task_id="T1",
            release_time=0,
            deadline=20,
            weight=1.0,
            blocks=[
                _build_block(block_id=0, duration_by_core={"hp_0": 5}, power_by_core={"hp_0": 20.0}),
                _build_block(
                    block_id=1,
                    predecessor_block_id=0,
                    duration_by_core={"lp_0": 6},
                    power_by_core={"lp_0": 16.0},
                ),
            ],
            core_ids=("hp_0", "lp_0"),
            require_uniform_block_core_ids=True,
        )


def test_task_single_block_constructor_for_task_level_baselines() -> None:
    """验证不划分任务块的任务可统一构造成单块任务。"""
    task = Task.single_block(
        task_id="T_single",
        release_time=2,
        deadline=9,
        weight=4.0,
        duration_by_core={"hp_0": 3, "lp_0": 5},
        power_by_core={"hp_0": 28.0, "lp_0": 15.0},
        core_ids=("hp_0", "lp_0"),
    )

    assert task.task_id == "T_single"
    assert task.release_time == 2
    assert task.deadline == 9
    assert len(task.blocks) == 1
    assert task.blocks[0].block_id == 0
    assert task.blocks[0].release_time == task.release_time
    assert task.blocks[0].deadline == task.deadline
    assert task.blocks[0].weight == task.weight


def test_task_deadline_does_not_need_to_equal_last_block_deadline() -> None:
    """验证任务级 deadline 是权威截止期，不强制等于最后一个任务块 deadline。"""
    task = Task(
        task_id="T1",
        release_time=0,
        deadline=20,
        weight=1.0,
        blocks=[
            _build_block(block_id=0, deadline=8),
            _build_block(block_id=1, predecessor_block_id=0, deadline=12),
        ],
        core_ids=("hp_0", "lp_0"),
    )

    assert task.deadline == 20
    assert task.blocks[-1].deadline == 12


def test_system_state_fields_and_temperature_validation() -> None:
    """验证系统状态字段存储与温度合法性校验。"""
    # 合法输入：SystemState 能正确保存核心状态字段。
    state = SystemState(time_slot=0, energy_joule=8000.0, temperature_celsius=35.0)
    assert state.time_slot == 0
    assert state.energy_joule == 8000.0
    assert state.temperature_celsius == 35.0

    # 低于安全电量阈值 E_min 的判断不在 SystemState 中完成；
    # 只要仍为非负电量，就允许进入后续能量模型并由 energy_violation 标记。
    low_energy_state = SystemState(time_slot=1, energy_joule=1.0, temperature_celsius=35.0)
    assert low_energy_state.energy_joule == 1.0
    assert low_energy_state.energy_violation is False

    with pytest.raises(ValueError, match="energy_joule"):
        SystemState(time_slot=1, energy_joule=-1.0, temperature_celsius=35.0)

    # 非法输入：温度不能是 NaN/Inf。
    with pytest.raises(ValueError, match="finite number"):
        SystemState(time_slot=0, energy_joule=1000.0, temperature_celsius=float("nan"))

    with pytest.raises(ValueError, match="finite number"):
        SystemState(time_slot=0, energy_joule=1000.0, temperature_celsius=float("inf"))

    # 温度不能低于绝对零度。
    with pytest.raises(ValueError, match="-273.15"):
        SystemState(time_slot=0, energy_joule=1000.0, temperature_celsius=-300.0)


def test_build_cores_from_type_counts() -> None:
    """验证按高性能核/低功耗核数量批量生成核心的逻辑。"""
    # 支持根据高性能核/低功耗核数量批量生成核心。
    cores = build_cores_from_type_counts(high_performance_count=2, low_power_count=2)
    assert [core.core_id for core in cores] == ["hp_0", "hp_1", "lp_0", "lp_1"]
    assert [core.core_type for core in cores] == [
        HIGH_PERFORMANCE_CORE_TYPE,
        HIGH_PERFORMANCE_CORE_TYPE,
        LOW_POWER_CORE_TYPE,
        LOW_POWER_CORE_TYPE,
    ]

    with pytest.raises(ValueError, match="non-negative integer"):
        build_cores_from_type_counts(high_performance_count=-1, low_power_count=1)


def test_schedule_decision_supports_run_and_idle() -> None:
    """验证调度决策同时支持空闲、任务级和任务块级三种表达。"""
    idle_decision = ScheduleDecision.idle(time_slot=3, core_id="hp_0")
    assert idle_decision.is_idle is True
    assert idle_decision.task_id is None
    assert idle_decision.block_id is None

    # 任务块级决策。
    run_decision = ScheduleDecision.run_block(
        time_slot=3,
        core_id="lp_0",
        task_id="T1",
        block_id=2,
        frequency_level="high",
    )
    assert run_decision.is_idle is False
    assert run_decision.task_id == "T1"
    assert run_decision.block_id == 2

    # 任务级决策（不携带 block_id）也应合法。
    task_level_decision = ScheduleDecision.run_task(time_slot=4, core_id="lp_0", task_id="T2")
    assert task_level_decision.is_idle is False
    assert task_level_decision.task_id == "T2"
    assert task_level_decision.block_id is None

    with pytest.raises(ValueError, match="must include task_id"):
        ScheduleDecision(time_slot=0, core_id="hp_0", is_idle=False)

    with pytest.raises(ValueError, match="block_id"):
        ScheduleDecision(time_slot=0, core_id="hp_0", is_idle=False, task_id="T1", block_id=-1)
