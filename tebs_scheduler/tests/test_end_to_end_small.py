from __future__ import annotations

import csv

from scripts.run_small_experiment import run_small_experiment


def _read_rows(path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_run_small_experiment_writes_end_to_end_outputs(tmp_path) -> None:
    result = run_small_experiment(
        output_dir=tmp_path,
        random_seed=3,
        num_tasks=2,
        simulation_slots=6,
        generate_figures=False,
        include_gurobi=False,
    )

    rows = _read_rows(result.summary_csv_path)
    methods = {row["method_name"] for row in rows}

    assert len(result.run_results) == 4
    assert {"safe_idle", "fcfs_dvfs", "tcn_scheduler", "hybrid_gurobi_tcn"} <= methods
    assert result.summary_csv_path.exists()
    assert result.solver_summary_csv_path.exists()
    assert all(run.output_paths["schedule_csv"].exists() for run in result.run_results)
    assert all(run.output_paths["state_csv"].exists() for run in result.run_results)
