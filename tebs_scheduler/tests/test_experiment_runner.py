from __future__ import annotations

import csv

import pytest

from tebs.experiment_runner import (
    ExperimentBatchConfig,
    ExperimentRunnerError,
    available_method_names,
    normalize_method_name,
    run_experiment_batch,
)


def _read_csv_rows(path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_experiment_runner_runs_minimal_batch_and_writes_summaries(tmp_path) -> None:
    result = run_experiment_batch(
        ExperimentBatchConfig(
            scenarios=("S1_light",),
            methods=("safe_idle", "fcfs_dvfs"),
            random_seeds=(7,),
            output_dir=tmp_path,
            num_tasks=2,
            simulation_slots=8,
            generate_figures=True,
        )
    )

    assert len(result.run_results) == 2
    assert result.summary_csv_path.exists()
    assert result.solver_summary_csv_path.exists()

    summary_rows = _read_csv_rows(result.summary_csv_path)
    solver_rows = _read_csv_rows(result.solver_summary_csv_path)

    assert {row["method_name"] for row in summary_rows} == {"safe_idle", "fcfs_dvfs"}
    assert {row["method_name"] for row in solver_rows} == {"safe_idle", "fcfs_dvfs"}
    assert all(row["scenario_name"] == "S1_light" for row in summary_rows)
    assert all(row["random_seed"] == "7" for row in summary_rows)

    for run_result in result.run_results:
        assert run_result.output_paths["schedule_csv"].exists()
        assert run_result.output_paths["state_csv"].exists()
        assert run_result.output_paths["solver_trace_csv"].exists()
        assert run_result.figure_outputs is not None
        assert run_result.figure_outputs.gantt_chart.exists()
        assert run_result.figure_outputs.energy_temperature_curve.exists()
        assert run_result.figure_outputs.power_curve.exists()


def test_experiment_runner_gives_each_method_independent_schedule_file(tmp_path) -> None:
    result = run_experiment_batch(
        ExperimentBatchConfig(
            scenarios=("S1_light",),
            methods=("safe_idle", "fcfs_intermittent"),
            random_seeds=(1,),
            output_dir=tmp_path,
            num_tasks=2,
            simulation_slots=6,
            generate_figures=False,
        )
    )

    paths = [run.output_paths["schedule_csv"] for run in result.run_results]

    assert len(paths) == 2
    assert len(set(paths)) == 2
    assert all(path.exists() for path in paths)
    assert {run.method_name for run in result.run_results} == {"safe_idle", "fcfs_intermittent"}


def test_experiment_runner_fixed_seed_is_reproducible(tmp_path) -> None:
    config = ExperimentBatchConfig(
        scenarios=("S1_light",),
        methods=("safe_idle", "fcfs_dvfs"),
        random_seeds=(11,),
        output_dir=tmp_path / "run_a",
        num_tasks=3,
        simulation_slots=10,
        generate_figures=False,
    )
    first = run_experiment_batch(config)
    second = run_experiment_batch(
        ExperimentBatchConfig(
            scenarios=config.scenarios,
            methods=config.methods,
            random_seeds=config.random_seeds,
            output_dir=tmp_path / "run_b",
            num_tasks=config.num_tasks,
            simulation_slots=config.simulation_slots,
            generate_figures=False,
        )
    )

    assert _read_csv_rows(first.summary_csv_path) == _read_csv_rows(second.summary_csv_path)


def test_experiment_runner_method_registry_and_aliases() -> None:
    assert "fcfs_dvfs" in available_method_names()
    assert "rhc_milp_gurobi" in available_method_names()
    assert normalize_method_name("FCFS-DVFS") == "fcfs_dvfs"
    assert normalize_method_name("rhc") == "rhc_milp_gurobi"


def test_unknown_method_raises_clear_error(tmp_path) -> None:
    with pytest.raises(ExperimentRunnerError, match="Unknown method_name"):
        run_experiment_batch(
            ExperimentBatchConfig(
                scenarios=("S1_light",),
                methods=("missing_method",),
                random_seeds=(0,),
                output_dir=tmp_path,
                num_tasks=1,
                simulation_slots=4,
                generate_figures=False,
            )
        )
