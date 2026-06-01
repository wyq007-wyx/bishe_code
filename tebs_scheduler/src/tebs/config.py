"""Configuration loading utilities for TEBS experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class ConfigError(ValueError):
    """Raised when configuration content is missing or invalid."""


@dataclass(slots=True, frozen=True)
class SimulationConfig:
    """Global simulation settings."""

    delta_t_seconds: int = 60
    horizon_slots: int = 12
    simulation_slots: int = 120
    core_count: int = 4
    sunlight_duration_slots: int = 6
    eclipse_duration_slots: int = 6
    energy_unit: str = "J"

    def __post_init__(self) -> None:
        _require_positive_int(self.delta_t_seconds, "delta_t_seconds")
        _require_positive_int(self.horizon_slots, "horizon_slots")
        _require_positive_int(self.simulation_slots, "simulation_slots")
        _require_positive_int(self.core_count, "core_count")
        _require_non_negative_int(self.sunlight_duration_slots, "sunlight_duration_slots")
        _require_non_negative_int(self.eclipse_duration_slots, "eclipse_duration_slots")
        if self.sunlight_duration_slots + self.eclipse_duration_slots <= 0:
            raise ConfigError(
                "sunlight_duration_slots + eclipse_duration_slots must be > 0."
            )
        if self.energy_unit != "J":
            raise ConfigError("energy_unit must be 'J' for internal consistency.")


@dataclass(slots=True, frozen=True)
class GurobiConfig:
    """Gurobi solver settings."""

    gurobi_time_limit: float = 5.0
    gurobi_mip_gap: float = 0.01
    gurobi_threads: int = 0
    gurobi_verbose: bool = False

    def __post_init__(self) -> None:
        _require_positive_number(self.gurobi_time_limit, "gurobi_time_limit")
        _require_non_negative_number(self.gurobi_mip_gap, "gurobi_mip_gap")
        _require_non_negative_int(self.gurobi_threads, "gurobi_threads")


@dataclass(slots=True, frozen=True)
class ThermalConfig:
    """Thermal model parameters."""

    initial_temperature_celsius: float = 35.0
    t_max_celsius: float = 75.0
    ambient_temperature_celsius: float = 25.0
    thermal_capacity_j_per_c: float = 1200.0
    cooling_coeff: float = 0.01

    def __post_init__(self) -> None:
        _require_positive_number(self.thermal_capacity_j_per_c, "thermal_capacity_j_per_c")
        _require_non_negative_number(self.cooling_coeff, "cooling_coeff")
        if self.t_max_celsius <= self.ambient_temperature_celsius:
            raise ConfigError("t_max_celsius must be greater than ambient_temperature_celsius.")


@dataclass(slots=True, frozen=True)
class EnergyConfig:
    """Energy model parameters. Internal unit is Joule."""

    e_max_j: float = 180000.0
    e_min_j: float = 36000.0
    initial_energy_j: float = 126000.0
    default_base_power_w: float = 25.0
    default_max_solar_power_w: float = 100.0

    def __post_init__(self) -> None:
        _require_positive_number(self.e_max_j, "e_max_j")
        _require_non_negative_number(self.e_min_j, "e_min_j")
        _require_non_negative_number(self.initial_energy_j, "initial_energy_j")
        _require_positive_number(self.default_base_power_w, "default_base_power_w")
        _require_non_negative_number(self.default_max_solar_power_w, "default_max_solar_power_w")
        if self.e_min_j >= self.e_max_j:
            raise ConfigError("e_min_j must be smaller than e_max_j.")
        if not (self.e_min_j <= self.initial_energy_j <= self.e_max_j):
            raise ConfigError("initial_energy_j must be within [e_min_j, e_max_j].")


@dataclass(slots=True, frozen=True)
class ConfigBundle:
    """Container for module-specific configuration objects."""

    simulation: SimulationConfig
    gurobi: GurobiConfig
    thermal: ThermalConfig
    energy: EnergyConfig


def default_config_path() -> Path:
    """Return the default project config path."""
    return Path(__file__).resolve().parents[2] / "configs" / "default_sim.yaml"


def load_config(config_path: str | Path | None = None) -> ConfigBundle:
    """Load TEBS configuration from YAML, falling back to defaults for missing sections."""
    path = Path(config_path) if config_path is not None else default_config_path()
    payload = _load_yaml_mapping(path)
    return load_config_from_mapping(payload)


def load_config_from_mapping(payload: Mapping[str, Any] | None) -> ConfigBundle:
    """Build configuration objects from a mapping payload."""
    data = dict(payload or {})
    simulation_data = _as_mapping(data.get("simulation"), section_name="simulation")
    gurobi_data = _as_mapping(data.get("gurobi"), section_name="gurobi")
    thermal_data = _as_mapping(data.get("thermal"), section_name="thermal")
    energy_data = _as_mapping(data.get("energy"), section_name="energy")

    simulation = SimulationConfig(
        delta_t_seconds=_read_int(simulation_data, "delta_t_seconds", 60),
        horizon_slots=_read_int(simulation_data, "horizon_slots", 12),
        simulation_slots=_read_int(simulation_data, "simulation_slots", 120),
        core_count=_read_int(simulation_data, "core_count", 4),
        sunlight_duration_slots=_read_int(simulation_data, "sunlight_duration_slots", 6),
        eclipse_duration_slots=_read_int(simulation_data, "eclipse_duration_slots", 6),
        energy_unit=_read_str(simulation_data, "energy_unit", "J").upper(),
    )
    gurobi = GurobiConfig(
        gurobi_time_limit=_read_float(gurobi_data, "gurobi_time_limit", 5.0),
        gurobi_mip_gap=_read_float(gurobi_data, "gurobi_mip_gap", 0.01),
        gurobi_threads=_read_int(gurobi_data, "gurobi_threads", 0),
        gurobi_verbose=_read_bool(gurobi_data, "gurobi_verbose", False),
    )
    thermal = ThermalConfig(
        initial_temperature_celsius=_read_float(
            thermal_data, "initial_temperature_celsius", 35.0
        ),
        t_max_celsius=_read_float(thermal_data, "t_max_celsius", 75.0),
        ambient_temperature_celsius=_read_float(
            thermal_data, "ambient_temperature_celsius", 25.0
        ),
        thermal_capacity_j_per_c=_read_float(thermal_data, "thermal_capacity_j_per_c", 1200.0),
        cooling_coeff=_read_float(thermal_data, "cooling_coeff", 0.01),
    )
    energy = EnergyConfig(
        e_max_j=_read_float(energy_data, "e_max_j", 180000.0),
        e_min_j=_read_float(energy_data, "e_min_j", 36000.0),
        initial_energy_j=_read_float(energy_data, "initial_energy_j", 126000.0),
        default_base_power_w=_read_float(energy_data, "default_base_power_w", 25.0),
        default_max_solar_power_w=_read_float(
            energy_data, "default_max_solar_power_w", 100.0
        ),
    )
    return ConfigBundle(
        simulation=simulation,
        gurobi=gurobi,
        thermal=thermal,
        energy=energy,
    )


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Configuration file not found: {path}")
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ConfigError(
            "PyYAML is required to load YAML configuration. Install with: pip install pyyaml"
        ) from exc

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ConfigError(f"Configuration root must be a mapping in: {path}")
    return payload


def _as_mapping(value: Any, *, section_name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"Section '{section_name}' must be a mapping.")
    return value


def _read_int(section: Mapping[str, Any], key: str, default: int) -> int:
    raw = section.get(key, default)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ConfigError(f"{key} must be an integer.")
    return raw


def _read_float(section: Mapping[str, Any], key: str, default: float) -> float:
    raw = section.get(key, default)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ConfigError(f"{key} must be a number.")
    return float(raw)


def _read_str(section: Mapping[str, Any], key: str, default: str) -> str:
    raw = section.get(key, default)
    if not isinstance(raw, str) or not raw.strip():
        raise ConfigError(f"{key} must be a non-empty string.")
    return raw


def _read_bool(section: Mapping[str, Any], key: str, default: bool) -> bool:
    raw = section.get(key, default)
    if not isinstance(raw, bool):
        raise ConfigError(f"{key} must be a boolean.")
    return raw


def _require_positive_int(value: int, field_name: str) -> None:
    if value <= 0:
        raise ConfigError(f"{field_name} must be > 0.")


def _require_non_negative_int(value: int, field_name: str) -> None:
    if value < 0:
        raise ConfigError(f"{field_name} must be >= 0.")


def _require_positive_number(value: float, field_name: str) -> None:
    if value <= 0:
        raise ConfigError(f"{field_name} must be > 0.")


def _require_non_negative_number(value: float, field_name: str) -> None:
    if value < 0:
        raise ConfigError(f"{field_name} must be >= 0.")


__all__ = [
    "ConfigBundle",
    "ConfigError",
    "EnergyConfig",
    "GurobiConfig",
    "SimulationConfig",
    "ThermalConfig",
    "default_config_path",
    "load_config",
    "load_config_from_mapping",
]
