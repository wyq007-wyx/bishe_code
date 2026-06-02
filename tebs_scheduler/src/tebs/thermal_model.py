"""Thermal update model for TEBS simulations."""

from __future__ import annotations

from dataclasses import dataclass

from .config import ConfigBundle, load_config


class ThermalModelError(ValueError):
    """Raised when thermal-model inputs are invalid."""


@dataclass(frozen=True, slots=True)
class ThermalStepResult:
    """Thermal accounting for one simulation slot."""

    temperature_next_celsius: float
    compute_heat_j: float
    base_heat_j: float
    cooling_heat_j: float
    net_heat_j: float
    thermal_budget_j: float
    thermal_violation: bool


def update_temperature_from_config(
    *,
    temperature_prev_celsius: float,
    compute_power_w: float,
    base_power_w: float,
    config: ConfigBundle | None = None,
    ambient_temperature_celsius: float | None = None,
) -> ThermalStepResult:
    """Update temperature using values from the project configuration."""

    cfg = config or load_config()
    return update_temperature(
        temperature_prev_celsius=temperature_prev_celsius,
        compute_power_w=compute_power_w,
        base_power_w=base_power_w,
        delta_t_seconds=cfg.simulation.delta_t_seconds,
        temperature_max_celsius=cfg.thermal.t_max_celsius,
        thermal_capacity_j_per_c=cfg.thermal.thermal_capacity_j_per_c,
        cooling_coeff=cfg.thermal.cooling_coeff,
        ambient_temperature_celsius=(
            cfg.thermal.ambient_temperature_celsius
            if ambient_temperature_celsius is None
            else ambient_temperature_celsius
        ),
    )


def update_temperature(
    *,
    temperature_prev_celsius: float,
    compute_power_w: float,
    base_power_w: float,
    delta_t_seconds: int,
    temperature_max_celsius: float,
    thermal_capacity_j_per_c: float,
    cooling_coeff: float,
    ambient_temperature_celsius: float,
) -> ThermalStepResult:
    """Update temperature with a simple lumped RC thermal model."""

    _require_number(temperature_prev_celsius, "temperature_prev_celsius")
    _require_non_negative_number(compute_power_w, "compute_power_w")
    _require_non_negative_number(base_power_w, "base_power_w")
    _require_positive_int(delta_t_seconds, "delta_t_seconds")
    _require_number(temperature_max_celsius, "temperature_max_celsius")
    _require_positive_number(thermal_capacity_j_per_c, "thermal_capacity_j_per_c")
    _require_non_negative_number(cooling_coeff, "cooling_coeff")
    _require_number(ambient_temperature_celsius, "ambient_temperature_celsius")
    if temperature_max_celsius <= ambient_temperature_celsius:
        raise ThermalModelError(
            "temperature_max_celsius must be greater than ambient_temperature_celsius."
        )

    compute_heat_j = compute_power_w * delta_t_seconds
    base_heat_j = base_power_w * delta_t_seconds
    cooling_heat_j = _cooling_heat_j(
        temperature_prev_celsius=temperature_prev_celsius,
        ambient_temperature_celsius=ambient_temperature_celsius,
        cooling_coeff=cooling_coeff,
        delta_t_seconds=delta_t_seconds,
    )
    net_heat_j = compute_heat_j + base_heat_j - cooling_heat_j
    temperature_next_celsius = temperature_prev_celsius + net_heat_j / thermal_capacity_j_per_c
    thermal_budget_j = compute_thermal_budget_j(
        temperature_prev_celsius=temperature_prev_celsius,
        base_power_w=base_power_w,
        delta_t_seconds=delta_t_seconds,
        temperature_max_celsius=temperature_max_celsius,
        thermal_capacity_j_per_c=thermal_capacity_j_per_c,
        cooling_coeff=cooling_coeff,
        ambient_temperature_celsius=ambient_temperature_celsius,
    )
    return ThermalStepResult(
        temperature_next_celsius=temperature_next_celsius,
        compute_heat_j=compute_heat_j,
        base_heat_j=base_heat_j,
        cooling_heat_j=cooling_heat_j,
        net_heat_j=net_heat_j,
        thermal_budget_j=thermal_budget_j,
        thermal_violation=temperature_next_celsius > temperature_max_celsius,
    )


def compute_thermal_budget_j(
    *,
    temperature_prev_celsius: float,
    base_power_w: float,
    delta_t_seconds: int,
    temperature_max_celsius: float,
    thermal_capacity_j_per_c: float,
    cooling_coeff: float,
    ambient_temperature_celsius: float,
) -> float:
    """Return compute heat that can still be absorbed in the next slot."""

    cooling_heat_j = _cooling_heat_j(
        temperature_prev_celsius=temperature_prev_celsius,
        ambient_temperature_celsius=ambient_temperature_celsius,
        cooling_coeff=cooling_coeff,
        delta_t_seconds=delta_t_seconds,
    )
    headroom_j = (temperature_max_celsius - temperature_prev_celsius) * thermal_capacity_j_per_c
    base_heat_j = base_power_w * delta_t_seconds
    return max(0.0, headroom_j + cooling_heat_j - base_heat_j)


def is_thermal_feasible(
    *,
    temperature_prev_celsius: float,
    compute_power_w: float,
    base_power_w: float,
    delta_t_seconds: int,
    temperature_max_celsius: float,
    thermal_capacity_j_per_c: float,
    cooling_coeff: float,
    ambient_temperature_celsius: float,
) -> bool:
    """Return whether one slot can run without exceeding the thermal limit."""

    result = update_temperature(
        temperature_prev_celsius=temperature_prev_celsius,
        compute_power_w=compute_power_w,
        base_power_w=base_power_w,
        delta_t_seconds=delta_t_seconds,
        temperature_max_celsius=temperature_max_celsius,
        thermal_capacity_j_per_c=thermal_capacity_j_per_c,
        cooling_coeff=cooling_coeff,
        ambient_temperature_celsius=ambient_temperature_celsius,
    )
    return not result.thermal_violation


def _cooling_heat_j(
    *,
    temperature_prev_celsius: float,
    ambient_temperature_celsius: float,
    cooling_coeff: float,
    delta_t_seconds: int,
) -> float:
    return cooling_coeff * max(0.0, temperature_prev_celsius - ambient_temperature_celsius) * delta_t_seconds


def _require_number(value: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ThermalModelError(f"{field_name} must be a number.")


def _require_positive_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ThermalModelError(f"{field_name} must be a positive integer.")


def _require_positive_number(value: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ThermalModelError(f"{field_name} must be > 0.")


def _require_non_negative_number(value: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ThermalModelError(f"{field_name} must be >= 0.")


__all__ = [
    "ThermalModelError",
    "ThermalStepResult",
    "compute_thermal_budget_j",
    "is_thermal_feasible",
    "update_temperature",
    "update_temperature_from_config",
]
