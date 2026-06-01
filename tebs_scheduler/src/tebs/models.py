"""TEBS 项目共享数据模型定义。"""

from __future__ import annotations

from dataclasses import dataclass, field


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


@dataclass(slots=True)
class TaskBlock:
    """任务块：任务内最小可调度单元（块级粒度）。

    时隙字段语义（均采用离散 slot 索引）：
    1. release_time：该块最早允许被调度的时隙（包含该时隙）。
    2. deadline：该块期望完成的最晚时隙（包含该时隙）。

    注意：release_time/deadline 定义的是“时间窗边界”，
    并不单独保证可调度性（例如执行时长过长时仍可能拖期）。
    """

    task_id: str
    block_id: int
    release_time: int
    deadline: int
    weight: float
    duration_by_core: dict[str, int]
    power_by_core: dict[str, float]
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
    """

    task_id: str
    release_time: int
    deadline: int
    weight: float
    blocks: list[TaskBlock]
    core_ids: tuple[str, ...] = ()

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

        # 可选契约：若给定 core_ids，则每个块必须覆盖全部核心。
        if self.core_ids:
            if len(set(self.core_ids)) != len(self.core_ids):
                raise ValueError("core_ids must be unique.")
            required_cores = set(self.core_ids)
            for block in self.blocks:
                if set(block.duration_by_core.keys()) != required_cores:
                    raise ValueError("Each block must cover all cores in core_ids.")
                if set(block.power_by_core.keys()) != required_cores:
                    raise ValueError("Each block must cover all cores in core_ids.")


@dataclass(slots=True)
class Core:
    core_id: str
    core_type: str
    speed_scale: float = 1.0
    power_scale: float = 1.0

    def __post_init__(self) -> None:
        _require_non_empty_str(self.core_id, "core_id")
        _require_non_empty_str(self.core_type, "core_type")
        _require_positive_number(self.speed_scale, "speed_scale")
        _require_positive_number(self.power_scale, "power_scale")


@dataclass(slots=True)
class SystemState:
    time_slot: int
    energy_joule: float
    temperature_celsius: float
    energy_violation: bool = False
    thermal_violation: bool = False

    def __post_init__(self) -> None:
        _require_non_negative_int(self.time_slot, "time_slot")
        if self.energy_joule < 0:
            raise ValueError("energy_joule must be >= 0.")


@dataclass(slots=True)
class ScheduleDecision:
    time_slot: int
    core_id: str
    is_idle: bool
    task_id: str | None = None
    block_id: int | None = None
    frequency_level: str | None = None

    def __post_init__(self) -> None:
        _require_non_negative_int(self.time_slot, "time_slot")
        _require_non_empty_str(self.core_id, "core_id")

        # 仅允许两种决策形态：
        #   1) idle：不携带任务字段
        #   2) run：必须同时提供 task_id 和 block_id
        if self.is_idle:
            if self.task_id is not None or self.block_id is not None:
                raise ValueError("Idle decision cannot carry task_id or block_id.")
            return

        if self.task_id is None or self.block_id is None:
            raise ValueError("Non-idle decision must include task_id and block_id.")
        _require_non_empty_str(self.task_id, "task_id")
        _require_non_negative_int(self.block_id, "block_id")

    @classmethod
    def idle(cls, time_slot: int, core_id: str) -> "ScheduleDecision":
        return cls(time_slot=time_slot, core_id=core_id, is_idle=True)

    @classmethod
    def run(
        cls,
        time_slot: int,
        core_id: str,
        task_id: str,
        block_id: int,
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


@dataclass(slots=True)
class ScheduleTrace:
    time_slot: int
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
    time_slot: int
    solver_name: str
    status: str
    solve_time_sec: float
    mip_gap: float | None = None
    objective_value: float | None = None
    best_bound: float | None = None
    has_feasible_solution: bool = False

    def __post_init__(self) -> None:
        _require_non_negative_int(self.time_slot, "time_slot")
        _require_non_empty_str(self.solver_name, "solver_name")
        _require_non_empty_str(self.status, "status")
        if self.solve_time_sec < 0:
            raise ValueError("solve_time_sec must be >= 0.")
        if self.mip_gap is not None and self.mip_gap < 0:
            raise ValueError("mip_gap must be >= 0 when provided.")


__all__ = [
    "Task",
    "TaskBlock",
    "Core",
    "SystemState",
    "ScheduleDecision",
    "ScheduleTrace",
    "SolverTrace",
]
