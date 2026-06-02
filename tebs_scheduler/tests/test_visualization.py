from __future__ import annotations

import pytest

from tebs.environment import generate_orbit_environment
from tebs.models import ScheduleDecision, ScheduleTrace, SolverTrace, SystemState, Task
from tebs.visualization import (
    VisualizationError,
    generate_experiment_figures,
    power_rows_from_schedule,
    render_energy_temperature_curve,
    render_gantt_chart,
    render_power_curve,
    render_solver_statistics,
    state_trace_to_energy_temperature_rows,
)


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _task() -> Task:
    return Task.single_block(
        task_id="T0",
        release_time=0,
        deadline=5,
        weight=1.0,
        duration_by_core={"hp_0": 2, "lp_0": 3},
        power_by_core={"hp_0": 10.0, "lp_0": 5.0},
        core_ids=("hp_0", "lp_0"),
    )


def _schedule_trace() -> list[ScheduleTrace]:
    return [
        ScheduleTrace(
            time_slot=0,
            decisions=[
                ScheduleDecision.run_block(0, "hp_0", "T0", 0),
                ScheduleDecision.idle(0, "lp_0"),
            ],
        ),
        ScheduleTrace(
            time_slot=1,
            decisions=[
                ScheduleDecision.run_block(1, "hp_0", "T0", 0),
                ScheduleDecision.idle(1, "lp_0"),
            ],
        ),
        ScheduleTrace(
            time_slot=2,
            decisions=[
                ScheduleDecision.idle(2, "hp_0"),
                ScheduleDecision.idle(2, "lp_0"),
            ],
        ),
    ]


def _state_trace() -> list[SystemState]:
    return [
        SystemState(time_slot=0, energy_joule=3600.0, temperature_celsius=25.0),
        SystemState(
            time_slot=1,
            energy_joule=7200.0,
            temperature_celsius=30.0,
            energy_violation=True,
        ),
        SystemState(
            time_slot=2,
            energy_joule=5400.0,
            temperature_celsius=35.0,
            thermal_violation=True,
        ),
    ]


def _solver_trace() -> list[SolverTrace]:
    return [
        SolverTrace(
            time_slot=0,
            solver_name="gurobi",
            status="OPTIMAL",
            solve_time_sec=0.5,
            mip_gap=0.0,
            objective_value=10.0,
            has_feasible_solution=True,
        ),
        SolverTrace(
            time_slot=1,
            solver_name="gurobi",
            status="TIME_LIMIT",
            solve_time_sec=1.5,
            mip_gap=0.05,
            objective_value=12.0,
            has_feasible_solution=True,
        ),
    ]


def _assert_png(path) -> None:
    assert path.exists()
    assert path.stat().st_size > 100
    assert path.read_bytes()[:8] == PNG_SIGNATURE


def test_render_visualization_png_files(tmp_path) -> None:
    env = generate_orbit_environment(
        simulation_slots=3,
        delta_t_seconds=60,
        sunlight_duration_slots=2,
        eclipse_duration_slots=1,
        max_solar_power_w=8.0,
        base_power_w=1.0,
        solar_power_profile="constant",
    )

    gantt = render_gantt_chart(
        schedule_trace=_schedule_trace(),
        output_path=tmp_path / "gantt.png",
    )
    energy = render_energy_temperature_curve(
        state_trace=_state_trace(),
        output_path=tmp_path / "energy_temperature.png",
    )
    power = render_power_curve(
        schedule_trace=_schedule_trace(),
        output_path=tmp_path / "power.png",
        tasks=[_task()],
        environment_slots=env.slots,
    )
    solver = render_solver_statistics(
        solver_trace=_solver_trace(),
        output_path=tmp_path / "solver.png",
    )

    for path in (gantt, energy, power, solver):
        _assert_png(path)


def test_energy_plot_rows_keep_joule_and_add_wh() -> None:
    rows = state_trace_to_energy_temperature_rows(_state_trace())

    assert rows[0]["energy_j"] == 3600.0
    assert rows[0]["energy_wh"] == 1.0
    assert rows[1]["energy_j"] == 7200.0
    assert rows[1]["energy_wh"] == 2.0


def test_power_rows_reconstruct_compute_base_total_and_harvested_power() -> None:
    env = generate_orbit_environment(
        simulation_slots=3,
        delta_t_seconds=60,
        sunlight_duration_slots=2,
        eclipse_duration_slots=1,
        max_solar_power_w=8.0,
        base_power_w=1.0,
        solar_power_profile="constant",
    )

    rows = power_rows_from_schedule(
        schedule_trace=_schedule_trace(),
        tasks=[_task()],
        environment_slots=env.slots,
    )

    assert rows[0]["compute_power_w"] == 10.0
    assert rows[0]["base_power_w"] == 1.0
    assert rows[0]["total_power_w"] == 11.0
    assert rows[0]["harvested_power_w"] == 8.0


def test_generate_experiment_figures_returns_standard_paths(tmp_path) -> None:
    outputs = generate_experiment_figures(
        schedule_trace=_schedule_trace(),
        state_trace=_state_trace(),
        metrics_rows=[{"method_name": "demo", "completed_task_count": 1}],
        solver_trace=_solver_trace(),
        output_dir=tmp_path,
        method_name="demo",
        tasks=[_task()],
    )

    for path in outputs.as_dict().values():
        assert path is not None
        _assert_png(path)


def test_empty_visualization_inputs_raise_clear_errors(tmp_path) -> None:
    with pytest.raises(VisualizationError, match="schedule_trace"):
        render_gantt_chart(schedule_trace=[], output_path=tmp_path / "gantt.png")

    with pytest.raises(VisualizationError, match="state_trace"):
        render_energy_temperature_curve(state_trace=[], output_path=tmp_path / "energy.png")

    with pytest.raises(VisualizationError, match="solver_trace"):
        render_solver_statistics(solver_trace=[], output_path=tmp_path / "solver.png")
