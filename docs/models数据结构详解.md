# `models.py` 数据结构详解

本文档对 `E:\study\bishe_code\tebs_scheduler\src\tebs\models.py` 中的核心数据结构进行详细说明，覆盖：

1. 各结构的职责定位；
2. 字段语义和单位约定；
3. 构造时校验规则（不变量）；
4. 结构之间的关系；
5. 后续模块如何使用这些结构。

---

## 1. 设计目标与整体思路

`models.py` 是全工程共享的数据契约层（Domain Model Layer）。  
实现策略是“**构造即校验**”：对象一旦创建成功，即满足后续仿真、调度、求解、统计模块的基础约束。

这样做的好处：

- 把错误尽早暴露在数据进入系统时；
- 降低后续模块重复写防御校验的成本；
- 保证不同调度器（规则/Gurobi/学习）使用同一语义。

---

## 2. 公共校验函数

### 2.1 `_require_non_empty_str(value, field_name)`

- 作用：校验字符串非空且非纯空白。
- 用于：`task_id`、`core_id`、`solver_name`、`status` 等标识字段。

### 2.2 `_require_non_negative_int(value, field_name)`

- 作用：校验字段是 `int` 且 `>= 0`。
- 用于：时隙索引、块编号、前驱编号等离散索引字段。

### 2.3 `_require_positive_number(value, field_name)`

- 作用：校验数值 `> 0`。
- 用于：任务权重、核心缩放系数等必须为正的参数。

---

## 3. 核心数据结构

## 3.1 `TaskBlock`

### 职责

表示“任务的最小可调度单元”（块级调度粒度）。

### 字段

- `task_id: str`  
  所属任务 ID。
- `block_id: int`  
  任务内部块编号（建议从 0 递增）。
- `release_time: int`  
  最早可执行时隙索引（包含该时隙）。  
  例：`release_time=5` 表示在 `slot=5` 及之后才允许调度该任务块。
- `deadline: int`  
  期望完成的最晚时隙索引（包含该时隙）。  
  例：`deadline=12` 表示希望不晚于 `slot=12` 完成。
- `weight: float`  
  块/任务优先级权重（用于加权目标）。
- `duration_by_core: dict[str, int]`  
  在不同核心上执行所需时长（单位：slot）。
- `power_by_core: dict[str, float]`  
  在不同核心上执行时计算功耗（单位建议：W）。
- `predecessor_block_id: int | None`  
  前驱块编号；无前驱时为 `None`。

### 关键不变量

1. `release_time <= deadline`；
2. `duration_by_core` 与 `power_by_core` 非空；
3. 二者的核心集合完全一致；
4. 每个核心执行时长 `> 0`；
5. 每个核心功耗 `>= 0`；
6. 若有前驱，必须满足 `predecessor_block_id < block_id`（防止逆序/环）。

### `release_time <= deadline` 详细解释（任务块级）

该约束的核心作用是确保“时间窗不倒置”。

1. `release_time < deadline`：普通时间窗，块有多个可调度时隙；
2. `release_time == deadline`：允许极紧迫块，只有一个边界时隙可用；
3. `release_time > deadline`：逻辑上无可用时隙，属于非法输入，必须在模型层拒绝。

补充说明：  
`release_time <= deadline` 只保证输入语义合法，不保证一定可按时完成。  
是否会拖期还取决于 `duration_by_core`、核心可用性、前驱关系和系统资源状态。

---

## 3.2 `Task`

### 职责

将多个 `TaskBlock` 聚合成完整任务，承载任务级时间窗和权重语义。

### 字段

- `task_id: str`
- `release_time: int`  
  任务级最早释放时隙索引（包含该时隙）。该字段通常用于任务进入队列的时刻判定。
- `deadline: int`  
  任务级最晚完成时隙索引（包含该时隙）。该字段通常用于任务级拖期统计与目标函数计算。
- `weight: float`
- `blocks: list[TaskBlock]`
- `core_ids: tuple[str, ...] = ()`  
  可选的“核心全集契约”。若给定，要求每个块都覆盖这些核心。

### 关键不变量

1. 任务级 `release_time <= deadline`；
2. `blocks` 至少包含一个块；
3. 所有块的 `task_id` 必须与任务 `task_id` 一致；
4. 任务内 `block_id` 必须唯一；
5. 若块声明前驱，则前驱必须指向“已出现的更早块”；
6. 若给定 `core_ids`，每个块都必须完整覆盖该核心集合（时长映射和功耗映射都要覆盖）。

### `release_time <= deadline` 详细解释（任务级）

任务级该约束与块级含义一致，确保任务的整体时间窗合法：

1. 若不满足该约束，任务在语义上“尚未释放就已截止”，无法参与任何合理调度；
2. 满足该约束后，任务至少具有一个合法边界时隙；
3. 任务级时间窗是宏观时效目标，块级时间窗是细粒度执行约束，调度时两者会共同生效。

---

## 3.3 `Core`

### 职责

描述异构核心的静态属性，供后续调度和资源建模引用。

### 字段

- `core_id: str`：核心标识；
- `core_type: str`：核心类型（如 `big`/`little`）；
- `speed_scale: float = 1.0`：速度缩放系数（>0）；
- `power_scale: float = 1.0`：功耗缩放系数（>0）。

### 关键不变量

- `core_id`、`core_type` 非空；
- `speed_scale > 0`；
- `power_scale > 0`。

---

## 3.4 `SystemState`

### 职责

表示单个时隙下系统状态（被仿真器逐时更新）。

### 字段

- `time_slot: int`：当前时隙；
- `energy_joule: float`：当前电量（单位：J）；
- `temperature_celsius: float`：当前温度（单位：℃）；
- `energy_violation: bool = False`：是否发生能量违约；
- `thermal_violation: bool = False`：是否发生热违约。

### 关键不变量

- `time_slot >= 0`；
- `energy_joule >= 0`。

---

## 3.5 `ScheduleDecision`

### 职责

统一表达“某时隙某核心的调度决策”。  
这是此前 `ScheduleAction` 与 `ScheduleDecision` 术语冲突后的统一版本。

### 字段

- `time_slot: int`
- `core_id: str`
- `is_idle: bool`
- `task_id: str | None`
- `block_id: int | None`
- `frequency_level: str | None`（为后续 DVFS 预留）

### 两种合法形态

1. **空闲决策**：`is_idle=True`，且 `task_id/block_id` 必须为 `None`；
2. **执行决策**：`is_idle=False`，且必须给出 `task_id + block_id`。

### 工厂方法

- `ScheduleDecision.idle(time_slot, core_id)`
- `ScheduleDecision.run(time_slot, core_id, task_id, block_id, frequency_level=None)`

---

## 3.6 `ScheduleTrace`

### 职责

记录“某个时隙”的全核决策集合。

### 字段

- `time_slot: int`
- `decisions: list[ScheduleDecision]`

### 关键不变量

1. 所有 `decision.time_slot` 必须与 `ScheduleTrace.time_slot` 一致；
2. 同一时隙内，每个 `core_id` 最多出现一次（避免重复分配冲突）。

---

## 3.7 `SolverTrace`

### 职责

统一记录求解器（尤其 Gurobi）的运行状态，供指标统计模块使用。

### 字段

- `time_slot: int`  
  对应的调度时隙编号，表示这条求解记录属于哪个滚动窗口起点/仿真时刻。
- `solver_name: str`  
  求解器名称，例如 `gurobi`、`cbc`，用于多求解器对比统计与日志过滤。
- `status: str`  
  求解状态码（字符串化），例如 `OPTIMAL`、`TIME_LIMIT`、`INFEASIBLE`，用于判断该次求解结果类型。
- `solve_time_sec: float`  
  本次求解耗时（秒），用于统计平均求解时间、P95/P99 时延、超时比例等性能指标。
- `mip_gap: float | None`  
  MIP 相对最优间隙；当求解器提供该值时用于衡量最优性质量。  
  `None` 表示该次求解未返回 gap（例如非 MIP 场景或异常终止）。
- `objective_value: float | None`  
  当前可行解目标值（若存在）。  
  `None` 通常表示没有可行解或求解未产生目标值。
- `best_bound: float | None`  
  分支定界过程中当前最优界（best bound），用于与 `objective_value` 联合分析收敛程度。  
  `None` 表示该次求解未返回 bound。
- `has_feasible_solution: bool = False`  
  是否拿到了可行解。该字段用于区分“有可行但未最优”（如超时）与“完全不可行/无解”两类情况。

### 关键不变量

- `solve_time_sec >= 0`；
- 若提供 `mip_gap`，则 `mip_gap >= 0`。

---

## 4. 结构关系（从任务到仿真）

1. `Task` 由多个 `TaskBlock` 构成；
2. 调度器在每个时隙、每个核心输出 `ScheduleDecision`；
3. 同一时隙的多个决策打包为 `ScheduleTrace`；
4. 仿真器依据决策更新 `SystemState`；
5. 若使用求解器，同时记录 `SolverTrace`；
6. `metrics.py` 最终消费 `ScheduleTrace + SystemState + SolverTrace` 统计指标。

---

## 5. 单位与命名约定

- 时间离散化单位：`slot`；
- 电量单位：`J`（焦耳）；
- 温度单位：`℃`；
- 功耗建议单位：`W`；
- 统一动作术语：`ScheduleDecision`（不再使用 `ScheduleAction`）。

---

## 6. 后续扩展建议

1. 若未来加入抢占式调度，可在 `ScheduleDecision` 中增加 `remaining_work` 或 `resume_token`；
2. 若支持更复杂 DAG，可将 `predecessor_block_id` 升级为 `predecessor_block_ids: list[int]`；
3. 若需更严格类型约束，可引入 `pydantic` 或 `attrs` 做更细粒度验证与序列化。

---

## 7. 最小示例

```python
from tebs.models import TaskBlock, Task, ScheduleDecision

b0 = TaskBlock(
    task_id="T1",
    block_id=0,
    release_time=0,
    deadline=10,
    weight=2.0,
    duration_by_core={"c0": 3, "c1": 5},
    power_by_core={"c0": 20.0, "c1": 12.0},
)

task = Task(
    task_id="T1",
    release_time=0,
    deadline=10,
    weight=2.0,
    blocks=[b0],
    core_ids=("c0", "c1"),
)

decision = ScheduleDecision.run(
    time_slot=0,
    core_id="c0",
    task_id="T1",
    block_id=0,
)
```

如果上述对象创建成功，说明最基本的数据契约满足，可直接进入后续仿真流程。
