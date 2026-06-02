"""Batch experiment runner for TEBS schedulers."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .baseline_dvfs import FcfsDvfsScheduler
from .baseline_fcfs_gurobi import FcfsGurobiScheduler
from .baseline_intermittent import FcfsIntermittentScheduler
from .config import ConfigBundle, load_config
from .environment import OrbitEnvironment, environment_from_config
from .metrics import compute_metrics
from .models import Core, SolverTrace, build_cores_from_type_counts
from .rhc_milp_gurobi import RhcMilpGurobiScheduler
from .scheduler_base import BaseScheduler, SafeIdleScheduler
from .simulator import SimulationResult, initial_state_from_config, run_simulation
from .solver_monitor import solver_summary_dict
from .task_factory import generate_task_set, load_scenario_specs, normalize_scenario_name
from .visualization import FigureOutputs, generate_experiment_figures


class ExperimentRunnerError(ValueError):
    """Raised when a batch experiment configuration is invalid."""


@dataclass(slots=True)
class ExperimentBatchConfig:
    """Configuration for a scenario x method x seed experiment batch."""

    scenarios: Sequence[str] = ("S1_light",)
    methods: Sequence[str] = ("fcfs_dvfs", "fcfs_intermittent")
    random_seeds: Sequence[int] = (0,)
    output_dir: str | Path = Path("outputs")
    num_tasks: int | None = None
    simulation_slots: int | None = None
    generate_figures: bool = True
    config: ConfigBundle | None = None
    scenarios_path: str | Path | None = None

    def __post_init__(self) -> None:
        if not self.scenarios:
            raise ExperimentRunnerError("scenarios must not be empty.")
        if not self.methods:
            raise ExperimentRunnerError("methods must not be empty.")
        if not self.random_seeds:
            raise ExperimentRunnerError("random_seeds must not be empty.")
        if self.num_tasks is not None and self.num_tasks <= 0:
            raise ExperimentRunnerError("num_tasks must be positive when provided.")
        if self.simulation_slots is not None and self.simulation_slots <= 0:
            raise ExperimentRunnerError("simulation_slots must be positive when provided.")


@dataclass(frozen=True, slots=True)
class ExperimentRunResult:
    """Output for one scenario/method/seed run."""

    scenario_name: str
    method_name: str
    random_seed: int
    summary: Mapping[str, float | int | str | None]
    output_paths: Mapping[str, Path]
    simulation_result: SimulationResult
    figure_outputs: FigureOutputs | None = None


@dataclass(frozen=True, slots=True)
class ExperimentBatchResult:
    """Aggregate output for a full batch run."""

    run_results: tuple[ExperimentRunResult, ...]
    summary_csv_path: Path
    solver_summary_csv_path: Path

    @property
    def summary_rows(self) -> tuple[Mapping[str, float | int | str | None], ...]:
        return tuple(result.summary for result in self.run_results)


SchedulerFactory = Callable[[ConfigBundle], BaseScheduler]


def available_method_names() -> tuple[str, ...]:
    """Return scheduler method names understood by the runner."""

    return tuple(sorted(_METHOD_FACTORIES))


def create_scheduler(method_name: str, config: ConfigBundle | None = None) -> BaseScheduler:
    """Create a fresh scheduler instance for one run."""

    normalized = normalize_method_name(method_name)
    factory = _METHOD_FACTORIES.get(normalized)
    if factory is None:
        available = ", ".join(available_method_names())
        raise ExperimentRunnerError(
            f"Unknown method_name '{method_name}'. Available methods: {available}"
        )
    return factory(config or load_config())


def normalize_method_name(method_name: str) -> str:
    """Normalize common method aliases to runner registry keys."""

    if not isinstance(method_name, str) or not method_name.strip():
        raise ExperimentRunnerError("method_name must be a non-empty string.")
    key = method_name.strip().lower().replace("-", "_")
    aliases = {
        "idle": "safe_idle",
        "safeidle": "safe_idle",
        "fcfsdvfs": "fcfs_dvfs",
        "fcfsintermittent": "fcfs_intermittent",
        "fcfsgurobi": "fcfs_gurobi",
        "rhc": "rhc_milp_gurobi",
        "rhc_gurobi": "rhc_milp_gurobi",
        "rhcmilpgurobi": "rhc_milp_gurobi",
    }
    return aliases.get(key, key)


def run_experiment_batch(batch_config: ExperimentBatchConfig) -> ExperimentBatchResult:
    """Run all scenario, random seed and scheduler method combinations."""

    cfg = batch_config.config or load_config()
    output_root = Path(batch_config.output_dir)
    dirs = _prepare_output_dirs(output_root)
    cores = _cores_from_config(cfg)
    run_results: list[ExperimentRunResult] = []
    solver_summary_rows: list[dict[str, float | int | str | None]] = []

    for scenario_name in batch_config.scenarios:
        canonical_scenario = normalize_scenario_name(scenario_name)
        simulation_slots = _resolve_simulation_slots(
            canonical_scenario,
            batch_config.simulation_slots,
            batch_config.scenarios_path,
        )
        environment = environment_from_config(cfg, simulation_slots=simulation_slots)
        for seed in batch_config.random_seeds:
            tasks = generate_task_set(
                scenario_name=canonical_scenario,
                num_tasks=batch_config.num_tasks,
                random_seed=seed,
                core_specs=cores,
                config=cfg,
                scenarios_path=batch_config.scenarios_path,
            )
            for method_name in batch_config.methods:
                run_result = _run_one_method(
                    scenario_name=canonical_scenario,
                    method_name=method_name,
                    random_seed=int(seed),
                    tasks=tasks,
                    cores=cores,
                    environment=environment,
                    simulation_slots=simulation_slots,
                    config=cfg,
                    dirs=dirs,
                    generate_figures=batch_config.generate_figures,
                )
                run_results.append(run_result)
                solver_summary_rows.append(
                    _solver_summary_row(
                        scenario_name=canonical_scenario,
                        method_name=run_result.method_name,
                        random_seed=int(seed),
                        solver_trace=run_result.simulation_result.solver_trace,
                    )
                )

    summary_csv = dirs["metrics"] / "summary.csv"
    solver_summary_csv = dirs["metrics"] / "solver_summary.csv"
    _write_rows_csv(summary_csv, [dict(result.summary) for result in run_results])
    _write_rows_csv(solver_summary_csv, solver_summary_rows)
    return ExperimentBatchResult(
        run_results=tuple(run_results),
        summary_csv_path=summary_csv,
        solver_summary_csv_path=solver_summary_csv,
    )


def run_single_experiment(
    *,
    scenario_name: str,
    method_name: str,
    random_seed: int,
    output_dir: str | Path,
    num_tasks: int | None = None,
    simulation_slots: int | None = None,
    generate_figures: bool = True,
    config: ConfigBundle | None = None,
    scenarios_path: str | Path | None = None,
) -> ExperimentRunResult:
    """Run one scenario/method/seed combination and write per-run artifacts."""

    batch_result = run_experiment_batch(
        ExperimentBatchConfig(
            scenarios=(scenario_name,),
            methods=(method_name,),
            random_seeds=(random_seed,),
            output_dir=output_dir,
            num_tasks=num_tasks,
            simulation_slots=simulation_slots,
            generate_figures=generate_figures,
            config=config,
            scenarios_path=scenarios_path,
        )
    )
    return batch_result.run_results[0]


def _run_one_method(
    *,
    scenario_name: str,
    method_name: str,
    random_seed: int,
    tasks,
    cores: Sequence[Core],
    environment: OrbitEnvironment,
    simulation_slots: int,
    config: ConfigBundle,
    dirs: Mapping[str, Path],
    generate_figures: bool,
) -> ExperimentRunResult:
    normalized_method = normalize_method_name(method_name)
    scheduler = create_scheduler(normalized_method, config)
    simulation_result = run_simulation(
        scheduler=scheduler,
        tasks=tasks,
        cores=cores,
        environment=environment,
        initial_system_state=initial_state_from_config(config),
        simulation_slots=simulation_slots,
        config=config,
    )
    summary = compute_metrics(
        tasks=tasks,
        schedule_trace=simulation_result.schedule_trace,
        state_trace=simulation_result.state_trace,
        solver_trace=simulation_result.solver_trace,
        completion_records=simulation_result.task_completion_records,
    )
    summary_row: dict[str, float | int | str | None] = {
        "scenario_name": scenario_name,
        "method_name": normalized_method,
        "random_seed": random_seed,
        "simulation_slots": simulation_slots,
    }
    summary_row.update(summary)

    artifact_prefix = _run_file_prefix(scenario_name, normalized_method, random_seed)
    output_paths = {
        "schedule_csv": dirs["schedules"] / f"{artifact_prefix}_schedule.csv",
        "state_csv": dirs["states"] / f"{artifact_prefix}_state.csv",
        "solver_trace_csv": dirs["solver"] / f"{artifact_prefix}_solver_trace.csv",
    }
    _write_schedule_csv(output_paths["schedule_csv"], simulation_result.schedule_trace)
    _write_state_csv(output_paths["state_csv"], simulation_result.state_trace)
    _write_solver_trace_csv(output_paths["solver_trace_csv"], simulation_result.solver_trace)

    figure_outputs = None
    if generate_figures:
        figure_outputs = generate_experiment_figures(
            schedule_trace=simulation_result.schedule_trace,
            state_trace=simulation_result.state_trace,
            metrics_rows=[summary_row],
            solver_trace=simulation_result.solver_trace,
            output_dir=dirs["figures"] / artifact_prefix,
            method_name=normalized_method,
            tasks=tasks,
            environment_slots=environment.slots,
        )
        output_paths = {
            **output_paths,
            **{
                key: value
                for key, value in figure_outputs.as_dict().items()
                if value is not None
            },
        }

    return ExperimentRunResult(
        scenario_name=scenario_name,
        method_name=normalized_method,
        random_seed=random_seed,
        summary=summary_row,
        output_paths=output_paths,
        simulation_result=simulation_result,
        figure_outputs=figure_outputs,
    )


def _solver_summary_row(
    *,
    scenario_name: str,
    method_name: str,
    random_seed: int,
    solver_trace: Sequence[SolverTrace],
) -> dict[str, float | int | str | None]:
    row: dict[str, float | int | str | None] = {
        "scenario_name": scenario_name,
        "method_name": method_name,
        "random_seed": random_seed,
    }
    row.update(solver_summary_dict(solver_trace))
    return row


def _prepare_output_dirs(output_root: Path) -> dict[str, Path]:
    dirs = {
        "root": output_root,
        "metrics": output_root / "metrics",
        "schedules": output_root / "schedules",
        "states": output_root / "states",
        "solver": output_root / "solver",
        "figures": output_root / "figures",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _cores_from_config(config: ConfigBundle) -> tuple[Core, ...]:
    return tuple(
        build_cores_from_type_counts(
            config.simulation.high_performance_core_count,
            config.simulation.low_power_core_count,
        )
    )


def _resolve_simulation_slots(
    scenario_name: str,
    override_slots: int | None,
    scenarios_path: str | Path | None,
) -> int:
    if override_slots is not None:
        return override_slots
    specs = load_scenario_specs(scenarios_path)
    normalized = normalize_scenario_name(scenario_name)
    if normalized not in specs:
        available = ", ".join(sorted(specs))
        raise ExperimentRunnerError(
            f"Unknown scenario_name '{scenario_name}'. Available scenarios: {available}"
        )
    return specs[normalized].simulation_slots


def _write_schedule_csv(path: Path, schedule_trace) -> None:
    rows = []
    for trace in schedule_trace:
        for decision in trace.decisions:
            rows.append(
                {
                    "time_slot": trace.time_slot,
                    "core_id": decision.core_id,
                    "is_idle": decision.is_idle,
                    "task_id": decision.task_id,
                    "block_id": decision.block_id,
                    "frequency_level": decision.frequency_level,
                }
            )
    _write_rows_csv(
        path,
        rows,
        fieldnames=("time_slot", "core_id", "is_idle", "task_id", "block_id", "frequency_level"),
    )


def _write_state_csv(path: Path, state_trace) -> None:
    rows = [
        {
            "time_slot": state.time_slot,
            "energy_joule": state.energy_joule,
            "energy_wh": state.energy_joule / 3600.0,
            "temperature_celsius": state.temperature_celsius,
            "energy_violation": state.energy_violation,
            "thermal_violation": state.thermal_violation,
        }
        for state in state_trace
    ]
    _write_rows_csv(
        path,
        rows,
        fieldnames=(
            "time_slot",
            "energy_joule",
            "energy_wh",
            "temperature_celsius",
            "energy_violation",
            "thermal_violation",
        ),
    )


def _write_solver_trace_csv(path: Path, solver_trace: Sequence[SolverTrace]) -> None:
    rows = [
        {
            "time_slot": trace.time_slot,
            "solver_name": trace.solver_name,
            "status": trace.status,
            "solve_time_sec": trace.solve_time_sec,
            "mip_gap": trace.mip_gap,
            "objective_value": trace.objective_value,
            "best_bound": trace.best_bound,
            "has_feasible_solution": trace.has_feasible_solution,
            "node_count": trace.node_count,
        }
        for trace in solver_trace
    ]
    _write_rows_csv(
        path,
        rows,
        fieldnames=(
            "time_slot",
            "solver_name",
            "status",
            "solve_time_sec",
            "mip_gap",
            "objective_value",
            "best_bound",
            "has_feasible_solution",
            "node_count",
        ),
    )


def _write_rows_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        identifier_fields = ("scenario_name", "method_name", "random_seed", "simulation_slots")
        discovered = {key for row in rows for key in row}
        ordered = [field for field in identifier_fields if field in discovered]
        ordered.extend(sorted(discovered - set(ordered)))
        fieldnames = tuple(ordered)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _run_file_prefix(scenario_name: str, method_name: str, random_seed: int) -> str:
    return f"{_safe_file_part(scenario_name)}_seed{random_seed}_{_safe_file_part(method_name)}"


def _safe_file_part(value: str) -> str:
    safe = "".join(char.lower() if char.isalnum() else "_" for char in value.strip())
    safe = "_".join(part for part in safe.split("_") if part)
    return safe or "run"


_METHOD_FACTORIES: dict[str, SchedulerFactory] = {
    "safe_idle": lambda cfg: SafeIdleScheduler(),
    "fcfs_dvfs": lambda cfg: FcfsDvfsScheduler(),
    "fcfs_intermittent": lambda cfg: FcfsIntermittentScheduler(),
    "fcfs_gurobi": lambda cfg: FcfsGurobiScheduler(config=cfg),
    "rhc_milp_gurobi": lambda cfg: RhcMilpGurobiScheduler(config=cfg),
}


__all__ = [
    "ExperimentBatchConfig",
    "ExperimentBatchResult",
    "ExperimentRunResult",
    "ExperimentRunnerError",
    "available_method_names",
    "create_scheduler",
    "normalize_method_name",
    "run_experiment_batch",
    "run_single_experiment",
]
