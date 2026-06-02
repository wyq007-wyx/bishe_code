from __future__ import annotations

from tebs.thermal_model import compute_thermal_budget_j, update_temperature


def test_high_compute_power_makes_temperature_rise() -> None:
    first = update_temperature(
        temperature_prev_celsius=25.0,
        compute_power_w=20.0,
        base_power_w=2.0,
        delta_t_seconds=60,
        temperature_max_celsius=70.0,
        thermal_capacity_j_per_c=3000.0,
        cooling_coeff=0.01,
        ambient_temperature_celsius=25.0,
    )
    second = update_temperature(
        temperature_prev_celsius=first.temperature_next_celsius,
        compute_power_w=20.0,
        base_power_w=2.0,
        delta_t_seconds=60,
        temperature_max_celsius=70.0,
        thermal_capacity_j_per_c=3000.0,
        cooling_coeff=0.01,
        ambient_temperature_celsius=25.0,
    )

    assert first.temperature_next_celsius > 25.0
    assert second.temperature_next_celsius > first.temperature_next_celsius


def test_idle_slot_can_cool_when_base_power_is_zero() -> None:
    result = update_temperature(
        temperature_prev_celsius=50.0,
        compute_power_w=0.0,
        base_power_w=0.0,
        delta_t_seconds=60,
        temperature_max_celsius=70.0,
        thermal_capacity_j_per_c=3000.0,
        cooling_coeff=1.0,
        ambient_temperature_celsius=25.0,
    )

    assert result.temperature_next_celsius < 50.0


def test_thermal_violation_and_compute_heat_are_reported() -> None:
    result = update_temperature(
        temperature_prev_celsius=69.0,
        compute_power_w=200.0,
        base_power_w=10.0,
        delta_t_seconds=60,
        temperature_max_celsius=70.0,
        thermal_capacity_j_per_c=3000.0,
        cooling_coeff=0.0,
        ambient_temperature_celsius=25.0,
    )

    assert result.compute_heat_j == 12000.0
    assert result.thermal_violation is True


def test_thermal_budget_is_non_negative() -> None:
    budget = compute_thermal_budget_j(
        temperature_prev_celsius=69.5,
        base_power_w=1.0,
        delta_t_seconds=60,
        temperature_max_celsius=70.0,
        thermal_capacity_j_per_c=3000.0,
        cooling_coeff=0.01,
        ambient_temperature_celsius=25.0,
    )

    assert budget >= 0.0
