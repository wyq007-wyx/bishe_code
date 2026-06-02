from __future__ import annotations

import pytest

from tebs.environment import EnvironmentError, generate_orbit_environment


def test_orbit_environment_repeats_sunlight_and_eclipse_cycle() -> None:
    env = generate_orbit_environment(
        simulation_slots=10,
        delta_t_seconds=60,
        sunlight_duration_slots=3,
        eclipse_duration_slots=2,
        max_solar_power_w=12.0,
        base_power_w=3.0,
        solar_power_profile="constant",
    )

    assert len(env) == 10
    assert env.is_sunlight_profile == (
        True,
        True,
        True,
        False,
        False,
        True,
        True,
        True,
        False,
        False,
    )
    assert all(power > 0 for power in env.harvested_power_profile_w[:3])
    assert env.harvested_power_profile_w[3] == 0.0
    assert env.harvested_power_profile_w[4] == 0.0
    assert all(power > 0 for power in env.base_power_profile_w)


def test_half_sine_solar_profile_is_positive_only_in_sunlight() -> None:
    env = generate_orbit_environment(
        simulation_slots=6,
        delta_t_seconds=60,
        sunlight_duration_slots=4,
        eclipse_duration_slots=2,
        max_solar_power_w=10.0,
        base_power_w=2.0,
        solar_power_profile="half_sine_orbit",
    )

    assert all(slot.harvested_power_w > 0 for slot in env.slots[:4])
    assert [slot.harvested_power_w for slot in env.slots[4:]] == [0.0, 0.0]
    assert max(env.harvested_power_profile_w) <= 10.0


def test_environment_window_is_bounded_by_available_slots() -> None:
    env = generate_orbit_environment(
        simulation_slots=5,
        delta_t_seconds=60,
        sunlight_duration_slots=2,
        eclipse_duration_slots=1,
        max_solar_power_w=8.0,
        base_power_w=2.5,
    )

    window = env.window(start_slot=3, horizon_slots=5)

    assert [slot.time_slot for slot in window] == [3, 4]


def test_invalid_orbit_period_raises_clear_error() -> None:
    with pytest.raises(EnvironmentError, match="must be > 0"):
        generate_orbit_environment(
            simulation_slots=5,
            delta_t_seconds=60,
            sunlight_duration_slots=0,
            eclipse_duration_slots=0,
            max_solar_power_w=8.0,
            base_power_w=2.5,
        )
