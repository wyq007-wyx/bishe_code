"""Battery-energy update model using Joule as the internal unit."""

from __future__ import annotations

from dataclasses import dataclass

from .config import ConfigBundle, load_config


class EnergyModelError(ValueError):
    """Raised when energy-model inputs are invalid."""


@dataclass(frozen=True, slots=True)
class EnergyStepResult:
    """Energy accounting for one simulation slot."""

    energy_load_j: float
    energy_harvested_j: float
    energy_unclipped_j: float
    energy_next_j: float
    energy_violation: bool


def update_energy_from_config(
    *,
    energy_prev_j: float,
    harvested_power_w: float,
    base_power_w: float,
    compute_power_w: float,
    config: ConfigBundle | None = None,
) -> EnergyStepResult:
    """Update battery energy using values from the project configuration."""

    cfg = config or load_config()
    return update_energy(
        energy_prev_j=energy_prev_j,
        harvested_power_w=harvested_power_w,
        base_power_w=base_power_w,
        compute_power_w=compute_power_w,
        delta_t_seconds=cfg.simulation.delta_t_seconds,
        energy_min_j=cfg.energy.e_min_j,
        energy_max_j=cfg.energy.e_max_j,
    )


def update_energy(
    *,
    energy_prev_j: float,
    harvested_power_w: float,
    base_power_w: float,
    compute_power_w: float,
    delta_t_seconds: int,
    energy_min_j: float,
    energy_max_j: float,
) -> EnergyStepResult:
    """Update battery energy for one slot.

    The returned ``energy_next_j`` is clipped to ``[energy_min_j, energy_max_j]``.
    The unclipped value is preserved so callers can inspect the deficit.
    """

    _require_non_negative_number(energy_prev_j, "energy_prev_j")
    _require_non_negative_number(harvested_power_w, "harvested_power_w")
    _require_non_negative_number(base_power_w, "base_power_w")
    _require_non_negative_number(compute_power_w, "compute_power_w")
    _require_positive_int(delta_t_seconds, "delta_t_seconds")
    _require_non_negative_number(energy_min_j, "energy_min_j")
    _require_positive_number(energy_max_j, "energy_max_j")
    if energy_min_j >= energy_max_j:
        raise EnergyModelError("energy_min_j must be smaller than energy_max_j.")

    energy_load_j = (base_power_w + compute_power_w) * delta_t_seconds
    energy_harvested_j = harvested_power_w * delta_t_seconds
    energy_unclipped_j = energy_prev_j + energy_harvested_j - energy_load_j
    energy_violation = energy_unclipped_j < energy_min_j
    energy_next_j = min(energy_max_j, max(energy_min_j, energy_unclipped_j))
    return EnergyStepResult(
        energy_load_j=energy_load_j,
        energy_harvested_j=energy_harvested_j,
        energy_unclipped_j=energy_unclipped_j,
        energy_next_j=energy_next_j,
        energy_violation=energy_violation,
    )


def is_energy_feasible(
    *,
    energy_prev_j: float,
    harvested_power_w: float,
    base_power_w: float,
    compute_power_w: float,
    delta_t_seconds: int,
    energy_min_j: float,
) -> bool:
    """Return whether one slot can run without crossing the energy floor."""

    _require_non_negative_number(energy_min_j, "energy_min_j")
    energy_load_j = (base_power_w + compute_power_w) * delta_t_seconds
    energy_harvested_j = harvested_power_w * delta_t_seconds
    return energy_prev_j + energy_harvested_j - energy_load_j >= energy_min_j


def _require_positive_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise EnergyModelError(f"{field_name} must be a positive integer.")


def _require_positive_number(value: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise EnergyModelError(f"{field_name} must be > 0.")


def _require_non_negative_number(value: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise EnergyModelError(f"{field_name} must be >= 0.")


__all__ = [
    "EnergyModelError",
    "EnergyStepResult",
    "is_energy_feasible",
    "update_energy",
    "update_energy_from_config",
]
