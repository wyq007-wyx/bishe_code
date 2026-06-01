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
  任务唯一标识，用于把多个 `TaskBlock` 归并到同一任务，并在调度轨迹和指标统计中关联同一任务。
- `release_time: int`  
  任务级最早释放时隙索引（包含该时隙）。该字段通常用于任务进入队列的时刻判定。
- `deadline: int`  
  任务级最晚完成时隙索引（包含该时隙）。该字段通常用于任务级拖期统计与目标函数计算。
- `weight: float`  
  任务级权重，用于表达任务重要性；常用于加权响应时间、加权拖期等目标或指标。
- `blocks: list[TaskBlock]`  
  任务包含的任务块集合；每个元素对应一个块级调度单元，顺序与前驱约束共同定义任务内部结构。
- `core_ids: tuple[str, ...] = ()`  
  可选的“任务可用核心全集”。若给定，要求每个任务块的核心集合是其子集。
- `require_uniform_block_core_ids: bool = False`  
  是否要求任务内所有任务块使用完全一致的核心集合。  
  `False`：允许多阶段任务使用不同核心集合；`True`：要求每个块核心集合一致。

### 关键不变量

1. 任务级 `release_time <= deadline`；
2. `blocks` 至少包含一个块；
3. 所有块的 `task_id` 必须与任务 `task_id` 一致；
4. 任务内 `block_id` 必须唯一；
5. 若块声明前驱，则前驱必须指向“已出现的更早块”；
6. 若给定 `core_ids`，每个块的核心集合必须是 `core_ids` 的子集；
7. 若 `require_uniform_block_core_ids=True`，所有块核心集合必须一致。

### `release_time <= deadline` 详细解释（任务级）

任务级该约束与块级含义一致，确保任务的整体时间窗合法：

1. 若不满足该约束，任务在语义上“尚未释放就已截止”，无法参与任何合理调度；
2. 满足该约束后，任务至少具有一个合法边界时隙；
3. 任务级时间窗是宏观时效目标，块级时间窗是细粒度执行约束，调度时两者会共同生效。

### 任务级截止期与任务块截止期

任务级 `deadline` 是任务整体的权威截止期，用于计算任务级拖期和论文指标。  
任务块的 `deadline` 是块级约束或辅助时间窗，不要求最后一个任务块的 `deadline` 等于任务级 `deadline`。

推荐处理方式：

1. 主方法若采用任务块调度：调度器根据任务块执行顺序计算任务完成时间 `C_j`，最终拖期仍使用任务级 `Task.deadline`；
2. 若任务块没有独立截止期：可直接令每个块的 `deadline = Task.deadline`；
3. 对比方案若不细分任务块：使用单块任务表示，即 `blocks=[block_id=0]`。

### 任务块核心适应性说明

1. 默认模式（`require_uniform_block_core_ids=False`）：  
   不同任务块可使用不同核心集合，适合多阶段任务（如前处理偏低功耗核、推理偏高性能核）。
2. 一致模式（`require_uniform_block_core_ids=True`）：  
   任务内所有块必须使用相同核心集合，适合“每个块核心适配一致”的任务类型。

### 单块任务构造方法

`Task.single_block(...)` 用于构造不显式划分任务块的任务。  
该方法会自动创建一个 `block_id=0` 的 `TaskBlock`，并让该任务块继承任务级：

- `release_time`
- `deadline`
- `weight`
- `duration_by_core`
- `power_by_core`

这样 FCFS-DVFS、FCFS-Intermittent 等任务级基线也能复用统一的 `Task -> TaskBlock` 数据结构。

---

## 3.3 `Core`

### 职责

描述异构核心的静态属性，供后续调度和资源建模引用。

### 字段

- `core_id: str`：核心标识；
- `core_type: str`：核心类型，当前约束为：
  - `high_performance`（高性能核）
  - `low_power`（低功耗核）
- `speed_scale: float = 1.0`：速度缩放系数（>0）；
- `power_scale: float = 1.0`：功耗缩放系数（>0）。

### 关键不变量

- `core_id`、`core_type` 非空；
- `core_type` 必须是 `high_performance` 或 `low_power`；
- `speed_scale > 0`；
- `power_scale > 0`。

### `core_id` 与“按数量配置”的映射方式

为支持“配置文件按类型给数量”的需求，`models.py` 增加了：

- `build_cores_from_type_counts(high_performance_count, low_power_count, ...)`

默认生成规则：

- 高性能核：`hp_0`, `hp_1`, ...；
- 低功耗核：`lp_0`, `lp_1`, ...。

这样后续 `config.py` 读取配置（例如 `2` 个高性能核、`2` 个低功耗核）后，可直接调用该函数生成 `Core` 列表，保证 `core_id` 命名稳定且可追踪。

推荐配置键：

- `simulation.high_performance_core_count`
- `simulation.low_power_core_count`

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
- `temperature_celsius` 必须是有限数值（不能是 `NaN/Inf`）；
- `temperature_celsius >= -273.15`（绝对零度下界）。

注意：`SystemState` 不检查电池最低安全电量 `E_min`。  
`E_min` 属于后续 `EnergyConfig` / `energy_model.py` / `simulator.py` 的运行安全约束。  
仿真中允许 `energy_joule < E_min`，并通过 `energy_violation=True` 记录违约，以便统计策略表现。

### 温度校验说明

1. 为什么要检查有限数值：  
   若温度出现 `NaN/Inf`，会直接污染热模型迭代、约束判断和指标统计，错误会在后续阶段被放大。
2. 为什么只加下界不加硬上界：  
   温度上界通常是任务/平台配置（如 `T_max`）决定的运行约束，而不是数据结构层面的“非法输入”。  
   仿真中允许温度超过 `T_max`，并通过 `thermal_violation` 记录违约，便于评估调度策略鲁棒性。

---

## 3.5 `ScheduleDecision`

### 职责

统一表达“某时隙某核心的调度决策”。  
这是此前 `ScheduleAction` 与 `ScheduleDecision` 术语冲突后的统一版本。

### 字段

- `time_slot: int`  
  决策生效的时隙编号，表示该动作发生在仿真时间轴的哪个 slot。
- `core_id: str`  
  执行该决策的核心 ID，用于将决策绑定到具体资源。
- `is_idle: bool`  
  是否为空闲动作：`True` 表示该核心在当前时隙不执行任务，`False` 表示执行任务。
- `task_id: str | None`  
  非空闲决策必须提供；表示该核心正在执行哪个任务。
- `block_id: int | None`  
  可选；当为 `None` 时表示“任务级决策”，当为整数时表示“任务块级决策”。
- `frequency_level: str | None`  
  可选频率档位标识（如 `high/medium/low`），为 DVFS 或能耗对比实验预留。

### 两种合法形态

1. **空闲决策**：`is_idle=True`，且 `task_id/block_id` 必须为 `None`；
2. **执行决策**：`is_idle=False`，且必须给出 `task_id`；`block_id` 可选。

执行决策按粒度分为：

- **任务级**：`task_id` 有值，`block_id=None`（适用于 FCFS-DVFS、FCFS-Intermittent 等不显式调度块的基线）；
- **任务块级**：`task_id` 有值，`block_id` 为非负整数（适用于 RHC-MILP 等块级调度方法）。

### 工厂方法

- `ScheduleDecision.idle(time_slot, core_id)`
- `ScheduleDecision.run(time_slot, core_id, task_id, block_id=None, frequency_level=None)`（通用入口）
- `ScheduleDecision.run_task(time_slot, core_id, task_id, frequency_level=None)`（任务级）
- `ScheduleDecision.run_block(time_slot, core_id, task_id, block_id, frequency_level=None)`（任务块级）

---

## 3.6 `ScheduleTrace`

### 职责

记录“某个时隙”的全核决策集合。

### 字段

- `time_slot: int`  
  该轨迹条目对应的时隙编号。
- `decisions: list[ScheduleDecision]`  
  当前时隙的全核决策集合；每个核心最多一条决策，可包含空闲、任务级或任务块级动作。

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
2. 调度器在每个时隙、每个核心输出 `ScheduleDecision`（可为任务级或任务块级）；
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
    duration_by_core={"hp_0": 3, "lp_0": 5},
    power_by_core={"hp_0": 20.0, "lp_0": 12.0},
)

task = Task(
    task_id="T1",
    release_time=0,
    deadline=10,
    weight=2.0,
    blocks=[b0],
    core_ids=("hp_0", "lp_0"),
)

decision = ScheduleDecision.run(
    time_slot=0,
    core_id="hp_0",
    task_id="T1",
    block_id=0,
)
```

如果上述对象创建成功，说明最基本的数据契约满足，可直接进入后续仿真流程。

任务级基线可使用单块任务：

```python
task = Task.single_block(
    task_id="T2",
    release_time=0,
    deadline=10,
    weight=1.0,
    duration_by_core={"hp_0": 2, "lp_0": 4},
    power_by_core={"hp_0": 25.0, "lp_0": 14.0},
    core_ids=("hp_0", "lp_0"),
)
```

---

## 8. 字段对照表（代码字段 vs 论文符号）

下表用于把 `models.py` 字段和论文中的数学记号快速对应起来。  
说明：论文符号可能因章节记法不同略有变化，以下采用当前方案文档中的主记法。

| 代码字段 | 所属结构 | 论文符号（建议） | 单位 | 约束/说明 |
|---|---|---|---|---|
| `task_id` | `Task` / `TaskBlock` / `ScheduleDecision` | `j` 或 `J_j` | - | 任务标识；字符串，非空 |
| `block_id` | `TaskBlock` / `ScheduleDecision` | `b` 或 `(j,b)` | - | 任务块标识；块级决策时使用，`>=0` |
| `release_time` | `Task` / `TaskBlock` | `r_j`（或 `r_{j,b}`） | slot | 最早释放时隙（含边界） |
| `deadline` | `Task` / `TaskBlock` | `d_j`（或 `d_{j,b}`） | slot | 最晚完成时隙（含边界） |
| `weight` | `Task` / `TaskBlock` | `w_j`（或 `w_{j,b}`） | - | 权重系数，`>0` |
| `duration_by_core[c]` | `TaskBlock` | `l_{j,b,c}` | slot | 某核心上执行时长，`>0` |
| `power_by_core[c]` | `TaskBlock` | `p_{j,b,c}` | W | 某核心上执行功耗，`>=0` |
| `predecessor_block_id` | `TaskBlock` | `pred(j,b)` | - | 前驱块编号；若存在则 `< block_id` |
| `core_ids` | `Task` | `C` | - | 可选任务可用核心全集；声明后每块核心集合必须为其子集 |
| `require_uniform_block_core_ids` | `Task` | - | bool | 是否要求任务内所有块使用相同核心集合 |
| `core_id` | `Core` / `ScheduleDecision` | `c` | - | 核心标识，如 `hp_0`、`lp_1` |
| `core_type` | `Core` | 核类型集合 | - | `high_performance` / `low_power` |
| `speed_scale` | `Core` | `\gamma_c` | - | 速度缩放系数，`>0` |
| `power_scale` | `Core` | `\eta_c` | - | 功耗缩放系数，`>0` |
| `time_slot` | `SystemState` / `ScheduleDecision` / `ScheduleTrace` / `SolverTrace` | `t` | slot | 离散时隙索引，`>=0` |
| `energy_joule` | `SystemState` | `E(t)` | J | 当前电量，`>=0` |
| `temperature_celsius` | `SystemState` | `T(t)` | ℃ | 有限数值且 `>= -273.15` |
| `energy_violation` | `SystemState` | 能量违约指示 | bool | 是否触发能量约束违约 |
| `thermal_violation` | `SystemState` | 热违约指示 | bool | 是否触发温度约束违约 |
| `is_idle` | `ScheduleDecision` | `idle_{c,t}` | bool | 是否空闲决策 |
| `frequency_level` | `ScheduleDecision` | 频率档位变量 | 离散档位 | DVFS 预留字段，可选 |
| `decisions` | `ScheduleTrace` | `\{a_{c,t}\}` | - | 单时隙全核决策集合 |
| `solver_name` | `SolverTrace` | 求解器标签 | - | 如 `gurobi` |
| `status` | `SolverTrace` | 求解状态 | - | 如 `OPTIMAL` / `TIME_LIMIT` |
| `solve_time_sec` | `SolverTrace` | `\tau_t` | s | 本次求解耗时，`>=0` |
| `mip_gap` | `SolverTrace` | `gap_t` | - | 相对最优间隙；若有则 `>=0` |
| `objective_value` | `SolverTrace` | `z_t` | 与目标函数一致 | 当前可行解目标值，可空 |
| `best_bound` | `SolverTrace` | `\underline{z}_t` / `\bar{z}_t` | 与目标函数一致 | 当前最优界，可空 |
| `has_feasible_solution` | `SolverTrace` | 可行解指示 | bool | 是否获得可行解 |
