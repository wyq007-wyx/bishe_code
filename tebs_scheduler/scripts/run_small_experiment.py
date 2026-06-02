"""Run a minimal TEBS end-to-end experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tebs.experiment_runner import ExperimentBatchConfig, ExperimentBatchResult, run_experiment_batch
from tebs.gurobi_adapter import is_gurobi_available


def run_small_experiment(
    *,
    output_dir: str | Path,
    scenario_name: str = "S1_light",
    random_seed: int = 0,
    num_tasks: int = 2,
    simulation_slots: int = 8,
    generate_figures: bool = False,
    include_gurobi: bool = True,
) -> ExperimentBatchResult:
    """Run one small scenario through baseline and learning schedulers."""

    methods = ["safe_idle", "fcfs_dvfs", "tcn_scheduler", "hybrid_gurobi_tcn"]
    if include_gurobi and is_gurobi_available():
        methods.extend(["fcfs_gurobi", "rhc_milp_gurobi"])

    return run_experiment_batch(
        ExperimentBatchConfig(
            scenarios=(scenario_name,),
            methods=tuple(methods),
            random_seeds=(random_seed,),
            output_dir=output_dir,
            num_tasks=num_tasks,
            simulation_slots=simulation_slots,
            generate_figures=generate_figures,
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/small_experiment")
    parser.add_argument("--scenario-name", default="S1_light")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--num-tasks", type=int, default=2)
    parser.add_argument("--simulation-slots", type=int, default=8)
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--skip-gurobi", action="store_true")
    args = parser.parse_args(argv)

    result = run_small_experiment(
        output_dir=args.output_dir,
        scenario_name=args.scenario_name,
        random_seed=args.random_seed,
        num_tasks=args.num_tasks,
        simulation_slots=args.simulation_slots,
        generate_figures=args.figures,
        include_gurobi=not args.skip_gurobi,
    )
    print(f"summary_csv={result.summary_csv_path}")
    print(f"solver_summary_csv={result.solver_summary_csv_path}")
    print(f"runs={len(result.run_results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
