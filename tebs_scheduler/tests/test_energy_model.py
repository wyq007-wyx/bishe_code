from __future__ import annotations

from tebs.energy_model import is_energy_feasible, update_energy


def test_energy_load_and_harvest_are_computed_in_joules() -> None:
    result = update_energy(
        energy_prev_j=100.0,
        harvested_power_w=2.0,
        base_power_w=1.0,
        compute_power_w=4.0,
        delta_t_seconds=10,
        energy_min_j=20.0,
        energy_max_j=200.0,
    )

    assert result.energy_load_j == 50.0
    assert result.energy_harvested_j == 20.0
    assert result.energy_unclipped_j == 70.0
    assert result.energy_next_j == 70.0
    assert result.energy_violation is False


def test_higher_compute_power_reduces_next_energy() -> None:
    low_power = update_energy(
        energy_prev_j=100.0,
        harvested_power_w=0.0,
        base_power_w=1.0,
        compute_power_w=1.0,
        delta_t_seconds=10,
        energy_min_j=0.0,
        energy_max_j=200.0,
    )
    high_power = update_energy(
        energy_prev_j=100.0,
        harvested_power_w=0.0,
        base_power_w=1.0,
        compute_power_w=5.0,
        delta_t_seconds=10,
        energy_min_j=0.0,
        energy_max_j=200.0,
    )

    assert high_power.energy_next_j < low_power.energy_next_j


def test_energy_next_is_clipped_to_bounds_and_flags_violation() -> None:
    full = update_energy(
        energy_prev_j=190.0,
        harvested_power_w=5.0,
        base_power_w=0.0,
        compute_power_w=0.0,
        delta_t_seconds=10,
        energy_min_j=20.0,
        energy_max_j=200.0,
    )
    depleted = update_energy(
        energy_prev_j=30.0,
        harvested_power_w=0.0,
        base_power_w=1.0,
        compute_power_w=5.0,
        delta_t_seconds=10,
        energy_min_j=20.0,
        energy_max_j=200.0,
    )

    assert full.energy_next_j == 200.0
    assert depleted.energy_unclipped_j == -30.0
    assert depleted.energy_next_j == 20.0
    assert depleted.energy_violation is True


def test_energy_feasibility_check_uses_unclipped_energy() -> None:
    assert is_energy_feasible(
        energy_prev_j=100.0,
        harvested_power_w=0.0,
        base_power_w=1.0,
        compute_power_w=2.0,
        delta_t_seconds=10,
        energy_min_j=20.0,
    )
    assert not is_energy_feasible(
        energy_prev_j=40.0,
        harvested_power_w=0.0,
        base_power_w=1.0,
        compute_power_w=2.0,
        delta_t_seconds=10,
        energy_min_j=20.0,
    )
