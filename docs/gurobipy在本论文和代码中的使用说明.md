# gurobipy 在本论文和代码中的使用说明

## 1. 文档目的

本文档说明 `gurobipy` 在本论文实验和当前代码工程中的作用、安装配置方式、调用链路、测试命令、输出指标以及论文写作中的表述建议。

当前项目将 Gurobi 作为混合整数线性规划求解器使用，Python 侧通过 `gurobipy` 建模和求解。Gurobi 相关代码是可选依赖：没有安装 `gurobipy` 或没有可用 license 时，非 Gurobi 模块仍可运行，Gurobi 相关测试会按设计跳过或返回安全 idle fallback。

## 2. 本论文为什么使用 Gurobi

本论文研究的是星载多核计算平台上的热-电协同感知任务调度问题。调度决策同时涉及：

1. 任务块是否在某个时隙执行。
2. 任务块分配到哪个异构核心。
3. 当前时隙和滚动窗口内的能量约束。
4. 当前时隙和滚动窗口内的热约束。
5. 任务 release time、deadline、前驱约束和核心容量约束。
6. 响应时间、拖期和剩余工作量等多目标权衡。

这些变量和约束天然适合用 MILP 表达。Gurobi 在本工程中承担两个角色：

1. 作为主方法 `RHC-MILP-Gurobi` 的求解器，用于滚动时域内的任务重排、核心匹配和热-电约束优化。
2. 作为强基线 `FCFS-Gurobi` 的求解器，在固定 FCFS 队列顺序下优化当前任务块的核心分配，用于区分“异构核心匹配收益”和“任务重排收益”。

论文中可以将 `gurobipy` 描述为 Gurobi Optimizer 的 Python API。本代码通过该 API 创建变量、约束和目标函数，并将求解状态、求解时间、MIPGap、目标值等统计信息统一记录到 `SolverTrace`。

## 3. 代码中哪些模块使用 gurobipy

### 3.1 Gurobi 适配层

文件：

```text
tebs_scheduler/src/tebs/gurobi_adapter.py
```

该模块封装所有通用 Gurobi 调用细节，避免各个调度器直接散落大量求解器代码。

主要接口：

| 接口 | 作用 |
|---|---|
| `is_gurobi_available()` | 检查当前 Python 环境是否可导入 `gurobipy`。 |
| `require_gurobi()` | 导入并返回 `gurobipy`，如果未安装则抛出 `GurobiUnavailableError`。 |
| `GurobiSolveParams` | 统一保存 `TimeLimit`、`MIPGap`、`Threads`、`OutputFlag` 等求解参数。 |
| `apply_gurobi_params(...)` | 把项目标准参数写入 Gurobi model。 |
| `optimize_gurobi_model(...)` | 对已构建的 Gurobi model 执行 `model.optimize()`。 |
| `extract_solve_result(...)` | 从 Gurobi model 中提取状态、目标值、MIPGap、节点数和运行时间。 |
| `GurobiSolveResult.to_solver_trace(...)` | 将求解结果转换为全工程统一的 `SolverTrace`。 |

状态映射：

```text
2  -> OPTIMAL
3  -> INFEASIBLE
4  -> INF_OR_UNBD
5  -> UNBOUNDED
9  -> TIME_LIMIT
11 -> INTERRUPTED
13 -> SUBOPTIMAL
```

这些状态后续会被 `solver_monitor.py` 和 `metrics.py` 汇总为：

```text
optimal_slot_ratio
feasible_slot_ratio
timeout_ratio
infeasible_ratio
avg_solve_time_sec
p50_solve_time_sec
p95_solve_time_sec
avg_mip_gap
p95_mip_gap
avg_node_count
avg_objective_value
```

### 3.2 FCFS-Gurobi 强基线

文件：

```text
tebs_scheduler/src/tebs/baseline_fcfs_gurobi.py
```

该模块实现固定队列强基线。它不允许任务重排，只从传入的 FCFS 队列头部选择当前可执行任务块，然后使用 Gurobi 在兼容核心中选择一个核心。

当前模型的核心变量：

```text
x[core_id] in {0, 1}
```

变量含义：

```text
x[core_id] = 1 表示当前 FCFS 队首任务块分配给该核心执行
```

主要约束：

```text
sum_c x[c] == 1
compute_power * delta_t <= available_energy_j
compute_power * delta_t <= thermal_budget_j
```

目标函数：

```text
min duration_on_core + 0.001 * compute_power_on_core
```

该目标优先选择执行时间较短的核心，并用一个很小的功耗惩罚在相同或接近执行时间时偏向低功耗核心。

输出：

1. 当前时隙的 `ScheduleDecision`。
2. 当前窗口计划 `GurobiWindowAssignment`。
3. 求解统计 `SolverTrace`。

### 3.3 RHC-MILP-Gurobi 主方法

文件：

```text
tebs_scheduler/src/tebs/rhc_milp_gurobi.py
```

该模块实现论文主方法。每个当前时隙 `t0` 都建立一个滚动窗口 MILP，并只执行窗口中当前时隙的动作。

当前模型的核心变量：

```text
x[block_index, core_id, slot] in {0, 1}
```

变量含义：

```text
x[j, c, t] = 1 表示候选任务块 j 在窗口时隙 t 分配到核心 c 执行
```

候选任务块筛选条件：

1. 任务未完成。
2. 任务 release time 落在当前滚动窗口内。
3. 任务块未完成。
4. 任务块 release time 落在当前滚动窗口内。
5. 前驱任务块已经完成。

主要约束：

```text
每个核心每个时隙最多执行一个任务块
每个任务块每个时隙最多分配到一个核心
窗口内至少服务一个可行任务块
任务和任务块 release time 之前不能调度
每个窗口时隙满足热预算约束
从 t0 到窗口内任意时隙的累计能量不低于 E_min
```

目标函数形式：

```text
min alpha * weighted_response_proxy
  + beta  * weighted_tardiness_proxy
  - mu    * served_work_reward
  + receding_penalty
```

其中：

```text
weighted_response_proxy = w_j * max(0, completion_proxy - release_time)
weighted_tardiness_proxy = w_j * max(0, completion_proxy - deadline)
served_work_reward = w_j
receding_penalty = 0.01 * max(0, slot - current_time)
```

该实现与当前仿真器的 slot 级推进保持一致：MILP 规划整个窗口，但仿真器每轮只执行当前时隙动作，下一时隙重新观测系统状态并重新求解。

## 4. Gurobi 参数配置

默认配置在：

```text
tebs_scheduler/configs/default_sim.yaml
tebs_scheduler/configs/gurobi.yaml
```

当前代码实际读取的参数字段包括：

```yaml
gurobi:
  gurobi_time_limit: 5.0
  gurobi_mip_gap: 0.01
  gurobi_threads: 0
  gurobi_verbose: false
```

含义：

| 参数 | 含义 | 默认值 |
|---|---|---:|
| `gurobi_time_limit` | 单次滚动窗口最大求解时间，单位秒 | `5.0` |
| `gurobi_mip_gap` | 相对 MIPGap 容忍度 | `0.01` |
| `gurobi_threads` | Gurobi 线程数，`0` 表示自动选择 | `0` |
| `gurobi_verbose` | 是否输出 Gurobi 求解日志 | `false` |

敏感性实验可使用如下参数组：

```text
TimeLimit in {1s, 3s, 5s, 10s}
MIPGap in {0, 0.01, 0.03, 0.05}
H in {6, 8, 12, 16, 24}
```

论文中可报告不同参数下的平均求解时间、P95 求解时间、最优解比例、可行解比例、超时比例和调度质量变化。

## 5. 安装与 license 配置

### 5.1 conda 环境 bishe

当前机器上已经在 `bishe` conda 环境中安装过 Gurobi：

```text
E:\conda\envs\bishe
```

验证命令：

```bat
conda activate bishe
python -c "import gurobipy as gp; print(gp.gurobi.version())"
gurobi_cl --version
grbgetkey --help
```

当前本机验证到的版本：

```text
gurobipy 13.0.2
Gurobi Optimizer 13.0.2
```

license 文件：

```text
E:\conda\envs\bishe\gurobi.lic
```

当前 license 验证结果：

```text
Restricted license - for non-production use only - expires 2027-11-29
```

### 5.2 工作区 .venv 沙箱环境

为了在工作区沙箱环境中运行测试，当前也在以下虚拟环境中安装了 `gurobipy`：

```text
E:\study\bishe_code\.venv
```

安装包：

```text
gurobipy 13.0.2
pytest 9.0.3
PyYAML 6.0.3
```

验证命令：

```powershell
E:\study\bishe_code\.venv\Scripts\python.exe -c "import gurobipy as gp; print(gp.gurobi.version())"
```

最小求解验证：

```powershell
E:\study\bishe_code\.venv\Scripts\python.exe -c "import gurobipy as gp; m=gp.Model(); x=m.addVar(lb=0); m.setObjective(x, gp.GRB.MAXIMIZE); m.addConstr(x<=1); m.optimize(); print('status=', m.Status, 'obj=', m.ObjVal)"
```

预期结果：

```text
status= 2 obj= 1.0
```

其中 `status=2` 对应 Gurobi 的 `OPTIMAL`。

注意：

1. `.venv/` 已被 `.gitignore` 忽略，不会提交到 GitHub。
2. `gurobi.lic` 不应提交到仓库。
3. 如果迁移到其他机器，需要重新安装 Gurobi 并配置该机器对应的 license。

## 6. 测试命令

### 6.1 使用 conda 环境

```bat
conda activate bishe
cd /d E:\study\bishe_code\tebs_scheduler
python -m pytest tests/test_gurobi_adapter.py tests/test_baseline_fcfs_gurobi.py tests/test_rhc_milp_gurobi_small.py
```

### 6.2 使用工作区 .venv 环境

```powershell
cd E:\study\bishe_code\tebs_scheduler
E:\study\bishe_code\.venv\Scripts\python.exe -m pytest tests/test_gurobi_adapter.py tests/test_baseline_fcfs_gurobi.py tests/test_rhc_milp_gurobi_small.py
```

### 6.3 全量测试

```powershell
cd E:\study\bishe_code\tebs_scheduler
E:\study\bishe_code\.venv\Scripts\python.exe -m pytest
```

当前工作区 `.venv` 的全量测试结果：

```text
74 passed
```

这说明 Gurobi 相关测试不再 skip，而是真实执行了求解。

## 7. 在批量实验中启用 Gurobi 方法

第17轮批量运行器文件：

```text
tebs_scheduler/src/tebs/experiment_runner.py
```

可以在 `methods` 中加入：

```text
fcfs_gurobi
rhc_milp_gurobi
```

示例：

```python
from tebs.experiment_runner import ExperimentBatchConfig, run_experiment_batch

result = run_experiment_batch(
    ExperimentBatchConfig(
        scenarios=("S1_light",),
        methods=("fcfs_dvfs", "fcfs_intermittent", "fcfs_gurobi", "rhc_milp_gurobi"),
        random_seeds=(0, 1, 2),
        output_dir="outputs",
        num_tasks=8,
        simulation_slots=60,
        generate_figures=True,
    )
)

print(result.summary_csv_path)
print(result.solver_summary_csv_path)
```

输出位置：

```text
outputs/
  metrics/
    summary.csv
    solver_summary.csv
  schedules/
    *_schedule.csv
  states/
    *_state.csv
  solver/
    *_solver_trace.csv
  figures/
    ...
```

其中 `solver_summary.csv` 用于支撑论文中的 Gurobi 求解速度和最优性分析。

## 8. 论文写作中的建议表述

可以在方法实现部分写：

```text
本文使用 Gurobi Optimizer 作为混合整数线性规划求解器，并通过其 Python API gurobipy 完成模型构建与求解。对于每个滚动时域窗口，代码根据当前系统状态、任务队列、未来环境输入和热-电约束创建 MILP 模型。模型求解后仅执行当前时隙动作，并在下一时隙重新观测系统状态后再次求解。
```

可以在实验设置部分写：

```text
Gurobi 单次滚动窗口默认 TimeLimit 设置为 5 s，默认 MIPGap 设置为 0.01，线程数由 Gurobi 自动选择。实验记录每个求解时隙的状态、运行时间、MIPGap、目标函数值和可行性，并统计最优解比例、可行解比例、超时比例、平均求解时间和 P95 求解时间。
```

可以在对比方法部分写：

```text
FCFS-Gurobi 固定任务队列顺序，仅使用 Gurobi 优化队首任务块的核心分配；RHC-MILP-Gurobi 则允许在滚动窗口内进行任务重排、核心匹配和热-电前瞻约束优化。因此二者的对比可以区分异构核心匹配带来的收益与任务级重排带来的额外收益。
```

可以在可复现性部分写：

```text
若实验环境未安装 gurobipy 或无可用 Gurobi license，代码中的非 Gurobi 基线仍可运行；Gurobi 相关测试会自动跳过或由调度器返回安全 idle。安装 Gurobi license 后，相关测试将执行真实 MILP 求解。
```

## 9. 常见问题

### 9.1 为什么以前 Gurobi 测试会 skipped

因为当时当前 Python 环境没有可用 `gurobipy` 或 license。测试文件使用了自动跳过机制，以保证基础仿真和规则基线不被可选商业求解器阻塞。

安装并配置 license 后，Gurobi 相关测试会真实执行，并且不再 skipped。

### 9.2 为什么要保留 gurobi_adapter.py

直接在每个调度器里写 `import gurobipy` 会导致：

1. 无 Gurobi 环境下整个包导入失败。
2. Gurobi 参数和状态解析分散。
3. 求解统计字段不统一。

因此项目使用 `gurobi_adapter.py` 统一封装可选依赖导入、参数设置、状态映射和结果提取。

### 9.3 license 文件能不能提交

不能。`gurobi.lic` 是机器和用户相关的授权文件，不应提交到 GitHub，也不应写入论文仓库。论文或 README 中只需说明如何获取和配置 license。

### 9.4 Gurobi 结果如何进入指标表

流程如下：

```text
Gurobi model.optimize()
  -> GurobiSolveResult
  -> SolverTrace
  -> compute_solver_metrics(...)
  -> summary.csv / solver_summary.csv
```

对应文件：

```text
tebs_scheduler/src/tebs/gurobi_adapter.py
tebs_scheduler/src/tebs/models.py
tebs_scheduler/src/tebs/metrics.py
tebs_scheduler/src/tebs/solver_monitor.py
tebs_scheduler/src/tebs/experiment_runner.py
```

## 10. 当前代码状态说明

当前真实 Gurobi 环境下全量测试已经通过：

```text
74 passed
```

真实运行 Gurobi 测试后，修复了 `FCFS-Gurobi` 的一个队列语义问题：

```text
tebs_scheduler/src/tebs/baseline_fcfs_gurobi.py
```

修复点：

```text
FCFS-Gurobi 现在尊重传入任务列表的 FCFS 队列顺序，不再在调度器内部按 release time 重新排序。
```

该行为更符合固定 FCFS 队列基线的定义：如果队首任务尚未释放，则不跳过队首任务去执行后续任务。
