"""Orbit environment generation for TEBS simulations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, Sequence

from .config import ConfigBundle, load_config


class EnvironmentError(ValueError):
    """Raised when orbit-environment inputs are invalid."""


@dataclass(frozen=True, slots=True)
class EnvironmentSlot:
    """Resource values available during one simulation slot."""

    time_slot: int
    is_sunlight: bool
    harvested_power_w: float
    base_power_w: float
    ambient_temperature_celsius: float

    def __post_init__(self) -> None:
        _require_non_negative_int(self.time_slot, "time_slot")
        _require_non_negative_number(self.harvested_power_w, "harvested_power_w")
        _require_positive_number(self.base_power_w, "base_power_w")
        if not math.isfinite(self.ambient_temperature_celsius):
            raise EnvironmentError("ambient_temperature_celsius must be finite.")


@dataclass(frozen=True, slots=True)
class OrbitEnvironment:
    """Discrete orbit resource profile."""

    delta_t_seconds: int
    slots: tuple[EnvironmentSlot, ...]

    def __post_init__(self) -> None:
        _require_positive_int(self.delta_t_seconds, "delta_t_seconds")
        if not self.slots:
            raise EnvironmentError("slots must not be empty.")
        expected_time = self.slots[0].time_slot
        for slot in self.slots:
            if slot.time_slot != expected_time:
                raise EnvironmentError("Environment slots must use contiguous time_slot values.")
            expected_time += 1

    def __len__(self) -> int:
        return len(self.slots)

    def __iter__(self) -> Iterator[EnvironmentSlot]:
        return iter(self.slots)

    def __getitem__(self, index: int) -> EnvironmentSlot:
        return self.slots[index]

    def window(self, start_slot: int, horizon_slots: int) -> tuple[EnvironmentSlot, ...]:
        """Return a bounded look-ahead window."""

        _require_non_negative_int(start_slot, "start_slot")
        _require_positive_int(horizon_slots, "horizon_slots")
        return self.slots[start_slot : start_slot + horizon_slots]

    @property
    def harvested_power_profile_w(self) -> tuple[float, ...]:
        return tuple(slot.harvested_power_w for slot in self.slots)

    @property
    def base_power_profile_w(self) -> tuple[float, ...]:
        return tuple(slot.base_power_w for slot in self.slots)

    @property
    def is_sunlight_profile(self) -> tuple[bool, ...]:
        return tuple(slot.is_sunlight for slot in self.slots)

    @property
    def thermal_env_profile_celsius(self) -> tuple[float, ...]:
        return tuple(slot.ambient_temperature_celsius for slot in self.slots)


def environment_from_config(
    config: ConfigBundle | None = None,
    *,
    simulation_slots: int | None = None,
) -> OrbitEnvironment:
    """Generate an orbit environment from the project configuration."""

    cfg = config or load_config()
    return generate_orbit_environment(
        simulation_slots=simulation_slots or cfg.simulation.simulation_slots,
        delta_t_seconds=cfg.simulation.delta_t_seconds,
        sunlight_duration_slots=cfg.simulation.sunlight_duration_slots,
        eclipse_duration_slots=cfg.simulation.eclipse_duration_slots,
        max_solar_power_w=cfg.energy.default_max_solar_power_w,
        base_power_w=cfg.energy.default_base_power_w,
        solar_power_profile=cfg.energy.solar_power_profile,
        solar_power_min_ratio=cfg.energy.solar_power_min_ratio,
        ambient_temperature_celsius=cfg.thermal.ambient_temperature_celsius,
    )


def generate_orbit_environment(
    *,
    simulation_slots: int,
    delta_t_seconds: int,
    sunlight_duration_slots: int,
    eclipse_duration_slots: int,
    max_solar_power_w: float,
    base_power_w: float,
    solar_power_profile: str = "half_sine_orbit",
    solar_power_min_ratio: float = 0.0,
    ambient_temperature_celsius: float = 25.0,
) -> OrbitEnvironment:
    """Build a repeatable sunlight/eclipse resource profile."""

    _require_positive_int(simulation_slots, "simulation_slots")
    _require_positive_int(delta_t_seconds, "delta_t_seconds")
    _require_non_negative_int(sunlight_duration_slots, "sunlight_duration_slots")
    _require_non_negative_int(eclipse_duration_slots, "eclipse_duration_slots")
    if sunlight_duration_slots + eclipse_duration_slots <= 0:
        raise EnvironmentError("sunlight_duration_slots + eclipse_duration_slots must be > 0.")
    _require_non_negative_number(max_solar_power_w, "max_solar_power_w")
    _require_positive_number(base_power_w, "base_power_w")
    _require_ratio(solar_power_min_ratio, "solar_power_min_ratio")
    if solar_power_profile not in {"constant", "half_sine_orbit"}:
        raise EnvironmentError("solar_power_profile must be 'constant' or 'half_sine_orbit'.")
    if not math.isfinite(ambient_temperature_celsius):
        raise EnvironmentError("ambient_temperature_celsius must be finite.")

    orbit_period_slots = sunlight_duration_slots + eclipse_duration_slots
    slots: list[EnvironmentSlot] = []
    for time_slot in range(simulation_slots):
        phase = time_slot % orbit_period_slots
        is_sunlight = sunlight_duration_slots > 0 and phase < sunlight_duration_slots
        harvested_power_w = _solar_power_for_phase(
            phase=phase,
            is_sunlight=is_sunlight,
            sunlight_duration_slots=sunlight_duration_slots,
            max_solar_power_w=max_solar_power_w,
            solar_power_profile=solar_power_profile,
            solar_power_min_ratio=solar_power_min_ratio,
        )
        slots.append(
            EnvironmentSlot(
                time_slot=time_slot,
                is_sunlight=is_sunlight,
                harvested_power_w=harvested_power_w,
                base_power_w=base_power_w,
                ambient_temperature_celsius=ambient_temperature_celsius,
            )
        )

    return OrbitEnvironment(delta_t_seconds=delta_t_seconds, slots=tuple(slots))


def ensure_environment_slots(
    environment: OrbitEnvironment | Sequence[EnvironmentSlot],
) -> tuple[EnvironmentSlot, ...]:
    """Normalize environment-like inputs to a slot tuple."""

    if isinstance(environment, OrbitEnvironment):
        return environment.slots
    slots = tuple(environment)
    if not slots:
        raise EnvironmentError("environment must include at least one slot.")
    return slots


def _solar_power_for_phase(
    *,
    phase: int,
    is_sunlight: bool,
    sunlight_duration_slots: int,
    max_solar_power_w: float,
    solar_power_profile: str,
    solar_power_min_ratio: float,
) -> float:
    if not is_sunlight or max_solar_power_w == 0:
        return 0.0
    if solar_power_profile == "constant":
        return max_solar_power_w

    daylight_position = (phase + 0.5) / sunlight_duration_slots
    sine_ratio = math.sin(math.pi * daylight_position)
    ratio = max(solar_power_min_ratio, sine_ratio)
    return max_solar_power_w * ratio


def _require_positive_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise EnvironmentError(f"{field_name} must be a positive integer.")


def _require_non_negative_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EnvironmentError(f"{field_name} must be a non-negative integer.")


def _require_positive_number(value: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise EnvironmentError(f"{field_name} must be > 0.")


def _require_non_negative_number(value: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise EnvironmentError(f"{field_name} must be >= 0.")


def _require_ratio(value: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise EnvironmentError(f"{field_name} must be within [0, 1].")


__all__ = [
    "EnvironmentError",
    "EnvironmentSlot",
    "OrbitEnvironment",
    "ensure_environment_slots",
    "environment_from_config",
    "generate_orbit_environment",
]
