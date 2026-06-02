from __future__ import annotations

from tebs.environment import generate_orbit_environment
from tebs.models import SystemState, build_cores_from_type_counts
from tebs.scheduler_base import SafeIdleScheduler
from tebs.simulator import run_simulation


def test_idle_scheduler_runs_complete_simulation_cycle() -> None:
    cores = tuple(build_cores_from_type_counts(1, 1))
    env = generate_orbit_environment(
        simulation_slots=6,
        delta_t_seconds=60,
        sunlight_duration_slots=3,
        eclipse_duration_slots=3,
        max_solar_power_w=6.0,
        base_power_w=1.0,
        solar_power_profile="constant",
    )
    initial_state = SystemState(time_slot=0, energy_joule=129600.0, temperature_celsius=25.0)

    result = run_simulation(
        scheduler=SafeIdleScheduler(),
        tasks=[],
        cores=cores,
        environment=env,
        initial_system_state=initial_state,
        simulation_slots=6,
    )

    assert len(result.schedule_trace) == 6
    assert len(result.state_trace) == 6
    assert len(result.task_completion_records) == 0
    assert all(len(trace.decisions) == len(cores) for trace in result.schedule_trace)
    assert all(
        len({decision.core_id for decision in trace.decisions}) == len(trace.decisions)
        for trace in result.schedule_trace
    )
    assert result.state_trace[0].energy_joule != initial_state.energy_joule
