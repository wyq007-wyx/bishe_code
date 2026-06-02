"""TEBS 项目共享数据模型定义。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

ABSOLUTE_ZERO_CELSIUS = -273.15
HIGH_PERFORMANCE_CORE_TYPE = "high_performance"
LOW_POWER_CORE_TYPE = "low_power"


# dataclass 在 __post_init__ 中复用的基础标量校验函数。
def _require_non_empty_str(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")


def _require_non_negative_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")


def _require_positive_number(value: float, field_name: str) -> None:
    if value <= 0:
        raise ValueError(f"{field_name} must be > 0.")


def build_cores_from_type_counts(
    high_performance_count: int, # 高性能核的数量
    low_power_count: int, # 低性能核的数量
    *, # 表示后面的参数调用时，必须加上参数名称
    high_performance_prefix: str = "hp",
    low_power_prefix: str = "lp",
    high_performance_speed_scale: float = 1.0,
    high_performance_power_scale: float = 1.0,
    low_power_speed_scale: float = 1.0,
    low_power_power_scale: float = 1.0,
) -> list["Core"]:
    """根据两类核心数量生成核心列表。

    该函数面向后续配置文件场景：
    - high_performance_count: 高性能核数量
    - low_power_count: 低功耗核数量
    - 生成 core_id 规则:
      - 高性能核: {high_performance_prefix}_{i}
      - 低功耗核: {low_power_prefix}_{i}
    """

    _require_non_negative_int(high_performance_count, "high_performance_count")
    _require_non_negative_int(low_power_count, "low_power_count")
    _require_non_empty_str(high_performance_prefix, "high_performance_prefix")
    _require_non_empty_str(low_power_prefix, "low_power_prefix")
    _require_positive_number(high_performance_speed_scale, "high_performance_speed_scale")
    _require_positive_number(high_performance_power_scale, "high_performance_power_scale")
    _require_positive_number(low_power_speed_scale, "low_power_speed_scale")
    _require_positive_number(low_power_power_scale, "low_power_power_scale")

    cores: list[Core] = []
    for i in range(high_performance_count):
        cores.append(
            Core(
                core_id=f"{high_performance_prefix}_{i}",
                core_type=HIGH_PERFORMANCE_CORE_TYPE,
                speed_scale=high_performance_speed_scale,
                power_scale=high_performance_power_scale,
            )
        )
    for i in range(low_power_count):
        cores.append(
            Core(
                core_id=f"{low_power_prefix}_{i}",
                core_type=LOW_POWER_CORE_TYPE,
                speed_scale=low_power_speed_scale,
                power_scale=low_power_power_scale,
            )
        )
    return cores


@dataclass(slots=True)
class TaskBlock:
    """任务块：任务内最小可调度单元（块级粒度）。

    时隙字段语义（均采用离散 slot 索引）：
    1. release_time：该块最早允许被调度的时隙（包含该时隙）。
    2. deadline：该块期望完成的最晚时隙（包含该时隙）。

    注意：release_time/deadline 定义的是“时间窗边界”，
    并不单独保证可调度性（例如执行时长过长时仍可能拖期）。
    """

    # 所属任务 ID，用于把任务块归并到同一任务实体。
    task_id: str
    # 任务内部块编号（通常从 0 递增），用于区分同一任务的不同块。
    block_id: int
    # 最早允许调度时隙（包含该时隙），单位：slot。
    release_time: int
    # 期望完成的最晚时隙（包含该时隙），单位：slot。
    deadline: int
    # 权重（优先级/重要性系数），用于加权目标函数。
    weight: float
    # 各核心上的执行时长映射：core_id -> duration_slots。
    duration_by_core: dict[str, int]
    # 各核心上的执行功耗映射：core_id -> power_watts。
    power_by_core: dict[str, float]
    # 前驱任务块编号；无前驱时为 None。
    predecessor_block_id: int | None = None

    def __post_init__(self) -> None:
        # 基础标量与时间窗合法性校验。
        _require_non_empty_str(self.task_id, "task_id")
        _require_non_negative_int(self.block_id, "block_id")
        _require_non_negative_int(self.release_time, "release_time")
        _require_non_negative_int(self.deadline, "deadline")
        _require_positive_number(self.weight, "weight")
        # release_time <= deadline 表示时间窗至少非空：
        # - release_time == deadline：允许“同一时隙释放、同一时隙截止”的紧迫块；
        # - release_time > deadline：时间窗倒置，属于非法输入。
        if self.release_time > self.deadline:
            raise ValueError("release_time must be <= deadline.")

        if not self.duration_by_core:
            raise ValueError("duration_by_core must not be empty.")
        if not self.power_by_core:
            raise ValueError("power_by_core must not be empty.")

        # 核心级时长/功耗映射必须使用同一组核心 ID。
        duration_cores = set(self.duration_by_core.keys())
        power_cores = set(self.power_by_core.keys())
        if duration_cores != power_cores:
            raise ValueError("duration_by_core and power_by_core must share the same core IDs.")

        for core_id, duration in self.duration_by_core.items():
            _require_non_empty_str(core_id, "core_id in duration_by_core")
            _require_non_negative_int(duration, f"duration_by_core[{core_id}]")
            if duration == 0:
                raise ValueError(f"duration_by_core[{core_id}] must be > 0.")

        for core_id, power in self.power_by_core.items():
            _require_non_empty_str(core_id, "core_id in power_by_core")
            if power < 0:
                raise ValueError(f"power_by_core[{core_id}] must be >= 0.")

        if self.predecessor_block_id is not None:
            _require_non_negative_int(self.predecessor_block_id, "predecessor_block_id")
            # 前驱块必须是更早的块，避免出现环依赖。
            if self.predecessor_block_id >= self.block_id:
                raise ValueError("predecessor_block_id must be smaller than block_id.")


@dataclass(slots=True)
class Task:
    """任务：由多个任务块组成的聚合对象。

    时隙字段语义（任务级）：
    1. release_time：任务对外可见的最早释放时隙（包含该时隙）。
    2. deadline：任务级期望完成的最晚时隙（包含该时隙）。

    任务级时间窗用于表达整体时效目标；块级时间窗用于表达块级约束。
    二者可同时存在，后续调度器会结合两层约束进行决策。

    核心适应性语义：
    1. 不同任务块可以拥有不同核心集合（用于多阶段异构适配）。
    2. 若需要所有块核心集合一致，可设置 require_uniform_block_core_ids=True。
    """

    # 任务 ID，全局唯一标识一个任务。
    task_id: str
    # 任务级最早释放时隙（包含该时隙），单位：slot。
    release_time: int
    # 任务级期望最晚完成时隙（包含该时隙），单位：slot。
    deadline: int
    # 任务级权重，反映任务整体优先级或价值。
    weight: float
    # 该任务包含的任务块序列（块级调度对象集合）。
    blocks: list[TaskBlock]
    # 可选核心全集约束；若给定，要求每个任务块可用核心集合是其子集。
    core_ids: tuple[str, ...] = ()
    # 是否要求任务内所有任务块使用完全一致的核心集合（默认 False）。
    require_uniform_block_core_ids: bool = False

    @classmethod
    def single_block(
        cls,
        *,
        task_id: str,
        release_time: int,
        deadline: int,
        weight: float,
        duration_by_core: dict[str, int],
        power_by_core: dict[str, float],
        core_ids: tuple[str, ...] = (),
        block_id: int = 0,
    ) -> "Task":
        """构造不显式划分任务块的单块任务。

        对比方案若只做任务级调度，也统一表示为一个 block_id=0 的任务。
        该单块继承任务级 release_time/deadline/weight，便于后续 simulator
        和 metrics 只处理一套 Task -> TaskBlock 结构。
        """
        block = TaskBlock(
            task_id=task_id,
            block_id=block_id,
            release_time=release_time,
            deadline=deadline,
            weight=weight,
            duration_by_core=duration_by_core,
            power_by_core=power_by_core,
        )
        return cls(
            task_id=task_id,
            release_time=release_time,
            deadline=deadline,
            weight=weight,
            blocks=[block],
            core_ids=core_ids,
            require_uniform_block_core_ids=True,
        )

    def __post_init__(self) -> None:
        # 任务级时间窗与静态约束校验。
        _require_non_empty_str(self.task_id, "task_id")
        _require_non_negative_int(self.release_time, "release_time")
        _require_non_negative_int(self.deadline, "deadline")
        _require_positive_number(self.weight, "weight")
        # 与 TaskBlock 相同，要求时间窗不倒置，确保输入语义一致。
        if self.release_time > self.deadline:
            raise ValueError("release_time must be <= deadline.")
        if not self.blocks:
            raise ValueError("Task must include at least one block.")

        # 保证块归属一致、编号唯一、前驱顺序可拓扑展开。
        block_ids = set()
        for block in self.blocks:
            if block.task_id != self.task_id:
                raise ValueError("All blocks in a Task must share the same task_id.")
            if block.block_id in block_ids:
                raise ValueError("Task blocks must have unique block_id values.")
            block_ids.add(block.block_id)
            if block.predecessor_block_id is not None and block.predecessor_block_id not in block_ids:
                raise ValueError("predecessor_block_id must point to an existing earlier block.")

        # 可选契约：若给定 core_ids，则每个块的可用核心必须是该集合的子集。
        if self.core_ids:
            if len(set(self.core_ids)) != len(self.core_ids):
                raise ValueError("core_ids must be unique.")
            required_cores = set(self.core_ids)
            for core_id in required_cores:
                _require_non_empty_str(core_id, "core_id in core_ids")
            for block in self.blocks:
                block_cores = set(block.duration_by_core.keys())
                if not block_cores.issubset(required_cores):
                    raise ValueError("Each block core set must be a subset of core_ids.")

        # 可选契约：若开启一致性约束，则所有块必须共享相同核心集合。
        if self.require_uniform_block_core_ids and len(self.blocks) > 1:
            first_block_cores = set(self.blocks[0].duration_by_core.keys())
            for block in self.blocks[1:]:
                if set(block.duration_by_core.keys()) != first_block_cores:
                    raise ValueError(
                        "All blocks must share the same core set when "
                        "require_uniform_block_core_ids=True."
                    )


@dataclass(slots=True)
class Core:
    # 核心唯一标识（如 hp_0 / lp_1）。
    core_id: str
    # 核心类型：high_performance 或 low_power。
    core_type: str
    # 速度缩放系数；>1 表示更快，<1 表示更慢（但必须 >0）。
    speed_scale: float = 1.0
    # 功耗缩放系数；>1 表示更耗能，<1 表示更节能（但必须 >0）。
    power_scale: float = 1.0

    def __post_init__(self) -> None:
        _require_non_empty_str(self.core_id, "core_id")
        _require_non_empty_str(self.core_type, "core_type")
        if self.core_type not in {HIGH_PERFORMANCE_CORE_TYPE, LOW_POWER_CORE_TYPE}:
            raise ValueError(
                f"core_type must be one of "
                f"{HIGH_PERFORMANCE_CORE_TYPE!r} or {LOW_POWER_CORE_TYPE!r}."
            )
        _require_positive_number(self.speed_scale, "speed_scale")
        _require_positive_number(self.power_scale, "power_scale")


@dataclass(slots=True)
class SystemState:
    # 当前系统状态所属时隙编号，单位：slot。
    time_slot: int
    # 当前电池剩余能量，单位：焦耳（J）。
    energy_joule: float
    # 当前系统温度，单位：摄氏度（℃）。
    temperature_celsius: float
    # 能量约束违约标记（如低于安全电量阈值）。
    energy_violation: bool = False
    # 热约束违约标记（如超过温度安全阈值）。
    thermal_violation: bool = False

    def __post_init__(self) -> None:
        _require_non_negative_int(self.time_slot, "time_slot")
        # 这里只检查物理合法性：电量不能为负。
        # 安全电量下限 E_min 属于能量模型/配置层约束，后续用 energy_violation 记录。
        if self.energy_joule < 0:
            raise ValueError("energy_joule must be >= 0.")
        # 温度必须是有限值，避免 NaN/Inf 污染后续热模型计算。
        if not math.isfinite(self.temperature_celsius):
            raise ValueError("temperature_celsius must be a finite number.")
        # 物理下界：温度不能低于绝对零度。
        if self.temperature_celsius < ABSOLUTE_ZERO_CELSIUS:
            raise ValueError(f"temperature_celsius must be >= {ABSOLUTE_ZERO_CELSIUS}.")


@dataclass(slots=True)
class ScheduleDecision: # 某时隙某核心的调度决策
    # 决策生效时隙编号，单位：slot。
    time_slot: int
    # 执行该决策的核心 ID。
    core_id: str
    # 是否为空闲决策；True 表示该时隙核心不执行任务。
    is_idle: bool
    # 任务 ID（非空闲决策必填）；任务级和任务块级决策都使用该字段。
    task_id: str | None = None
    # 任务块 ID（可选）；None=任务级决策，整数=任务块级决策。
    block_id: int | None = None
    # 频率档位（可选），用于 DVFS 场景记录如 high/medium/low。
    frequency_level: str | None = None

    def __post_init__(self) -> None:
        _require_non_negative_int(self.time_slot, "time_slot")
        _require_non_empty_str(self.core_id, "core_id")

        # 仅允许两种决策形态：
        #   1) idle：不携带任务字段
        #   2) run：必须提供 task_id，block_id 可选
        #      - block_id is None: 任务级决策（适配 FCFS-DVFS / FCFS-Intermittent）
        #      - block_id is int: 任务块级决策（适配 RHC-MILP）
        if self.is_idle:
            if self.task_id is not None or self.block_id is not None:
                raise ValueError("Idle decision cannot carry task_id or block_id.")
            return

        if self.task_id is None:
            raise ValueError("Non-idle decision must include task_id.")
        _require_non_empty_str(self.task_id, "task_id")
        if self.block_id is not None:
            _require_non_negative_int(self.block_id, "block_id")
        if self.frequency_level is not None:
            _require_non_empty_str(self.frequency_level, "frequency_level")

    @classmethod
    def idle(cls, time_slot: int, core_id: str) -> "ScheduleDecision":
        return cls(time_slot=time_slot, core_id=core_id, is_idle=True)

    @classmethod
    def run(
        cls,
        time_slot: int,
        core_id: str,
        task_id: str,
        block_id: int | None = None,
        frequency_level: str | None = None,
    ) -> "ScheduleDecision":
        return cls(
            time_slot=time_slot,
            core_id=core_id,
            is_idle=False,
            task_id=task_id,
            block_id=block_id,
            frequency_level=frequency_level,
        )

    @classmethod
    def run_task(
        cls,
        time_slot: int,
        core_id: str,
        task_id: str,
        frequency_level: str | None = None,
    ) -> "ScheduleDecision":
        """任务级调度决策（不指定 block_id）。"""
        return cls.run(
            time_slot=time_slot,
            core_id=core_id,
            task_id=task_id,
            block_id=None,
            frequency_level=frequency_level,
        )

    @classmethod
    def run_block(
        cls,
        time_slot: int,
        core_id: str,
        task_id: str,
        block_id: int,
        frequency_level: str | None = None,
    ) -> "ScheduleDecision":
        """任务块级调度决策（显式指定 block_id）。"""
        return cls.run(
            time_slot=time_slot,
            core_id=core_id,
            task_id=task_id,
            block_id=block_id,
            frequency_level=frequency_level,
        )


@dataclass(slots=True)
class ScheduleTrace:
    # 该调度轨迹条目对应的时隙编号，单位：slot。
    time_slot: int
    # 该时隙下所有核心的决策集合（每核心最多一个决策）。
    decisions: list[ScheduleDecision] = field(default_factory=list)

    def __post_init__(self) -> None:
        _require_non_negative_int(self.time_slot, "time_slot")
        seen_cores: set[str] = set()
        # 同一时隙内，每个核心最多只能出现一个决策。
        for decision in self.decisions:
            if decision.time_slot != self.time_slot:
                raise ValueError("All decisions in ScheduleTrace must match time_slot.")
            if decision.core_id in seen_cores:
                raise ValueError("Duplicate core_id in a single ScheduleTrace.")
            seen_cores.add(decision.core_id)


@dataclass(slots=True)
class SolverTrace:
    # 求解记录对应的调度时隙编号。
    time_slot: int
    # 求解器名称（如 gurobi / cbc）。
    solver_name: str
    # 求解状态（如 OPTIMAL / TIME_LIMIT / INFEASIBLE）。
    status: str
    # 本次求解耗时，单位：秒。
    solve_time_sec: float
    # MIP 相对最优间隙；未提供时为 None。
    mip_gap: float | None = None
    # 当前可行解目标值；无可行解或无返回时为 None。
    objective_value: float | None = None
    # 当前最优界（best bound）；无返回时为 None。
    best_bound: float | None = None
    # 是否找到可行解（可与 status 结合判断“可行但未最优”）。
    has_feasible_solution: bool = False
    # 分支定界节点数；非 MIP 求解或求解器未返回时为 None。
    node_count: float | None = None

    def __post_init__(self) -> None:
        _require_non_negative_int(self.time_slot, "time_slot")
        _require_non_empty_str(self.solver_name, "solver_name")
        _require_non_empty_str(self.status, "status")
        if self.solve_time_sec < 0:
            raise ValueError("solve_time_sec must be >= 0.")
        if self.mip_gap is not None and self.mip_gap < 0:
            raise ValueError("mip_gap must be >= 0 when provided.")
        if self.node_count is not None and self.node_count < 0:
            raise ValueError("node_count must be >= 0 when provided.")


__all__ = [
    "HIGH_PERFORMANCE_CORE_TYPE",
    "LOW_POWER_CORE_TYPE",
    "Task",
    "TaskBlock",
    "Core",
    "SystemState",
    "ScheduleDecision",
    "ScheduleTrace",
    "SolverTrace",
    "build_cores_from_type_counts",
]
