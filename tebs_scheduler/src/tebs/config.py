"""TEBS 实验配置加载工具。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class ConfigError(ValueError):
    """配置内容缺失或非法时抛出的异常。"""


@dataclass(slots=True, frozen=True)
class SimulationConfig:
    """全局仿真参数。"""

    delta_t_seconds: int = 60
    horizon_slots: int = 12
    simulation_slots: int = 194
    high_performance_core_count: int = 2
    low_power_core_count: int = 2
    sunlight_duration_slots: int = 61
    eclipse_duration_slots: int = 36
    energy_unit: str = "J"

    def __post_init__(self) -> None:
        _require_positive_int(self.delta_t_seconds, "delta_t_seconds")
        _require_positive_int(self.horizon_slots, "horizon_slots")
        _require_positive_int(self.simulation_slots, "simulation_slots")
        _require_non_negative_int(self.high_performance_core_count, "high_performance_core_count")
        _require_non_negative_int(self.low_power_core_count, "low_power_core_count")
        if self.high_performance_core_count + self.low_power_core_count <= 0:
            raise ConfigError(
                "high_performance_core_count + low_power_core_count must be > 0."
            )
        _require_non_negative_int(self.sunlight_duration_slots, "sunlight_duration_slots")
        _require_non_negative_int(self.eclipse_duration_slots, "eclipse_duration_slots")
        if self.sunlight_duration_slots + self.eclipse_duration_slots <= 0:
            raise ConfigError(
                "sunlight_duration_slots + eclipse_duration_slots must be > 0."
            )
        if self.energy_unit != "J":
            raise ConfigError("energy_unit must be 'J' for internal consistency.")

    @property
    def core_count(self) -> int:
        """总核心数量（兼容旧代码访问）。"""
        return self.high_performance_core_count + self.low_power_core_count


@dataclass(slots=True, frozen=True)
class GurobiConfig:
    """Gurobi 求解器参数。"""

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
    """热模型参数。"""

    initial_temperature_celsius: float = 25.0
    t_max_celsius: float = 70.0
    ambient_temperature_celsius: float = 25.0
    thermal_capacity_j_per_c: float = 3000.0
    cooling_coeff: float = 0.01

    def __post_init__(self) -> None:
        _require_positive_number(self.thermal_capacity_j_per_c, "thermal_capacity_j_per_c")
        _require_non_negative_number(self.cooling_coeff, "cooling_coeff")
        if self.t_max_celsius <= self.ambient_temperature_celsius:
            raise ConfigError("t_max_celsius must be greater than ambient_temperature_celsius.")


@dataclass(slots=True, frozen=True)
class EnergyConfig:
    """能量模型参数，内部统一使用焦耳。"""

    e_max_j: float = 144000.0
    e_min_j: float = 115200.0
    initial_energy_j: float = 129600.0
    default_base_power_w: float = 3.6
    default_max_solar_power_w: float = 14.0
    solar_power_profile: str = "half_sine_orbit"
    solar_power_min_ratio: float = 0.0

    def __post_init__(self) -> None:
        _require_positive_number(self.e_max_j, "e_max_j")
        _require_non_negative_number(self.e_min_j, "e_min_j")
        _require_non_negative_number(self.initial_energy_j, "initial_energy_j")
        _require_positive_number(self.default_base_power_w, "default_base_power_w")
        _require_non_negative_number(self.default_max_solar_power_w, "default_max_solar_power_w")
        if self.solar_power_profile not in {"constant", "half_sine_orbit"}:
            raise ConfigError("solar_power_profile must be 'constant' or 'half_sine_orbit'.")
        _require_ratio(self.solar_power_min_ratio, "solar_power_min_ratio")
        if self.e_min_j >= self.e_max_j:
            raise ConfigError("e_min_j must be smaller than e_max_j.")
        if not (self.e_min_j <= self.initial_energy_j <= self.e_max_j):
            raise ConfigError("initial_energy_j must be within [e_min_j, e_max_j].")


@dataclass(slots=True, frozen=True)
class ConfigBundle:
    """按模块聚合后的配置对象。"""

    simulation: SimulationConfig
    gurobi: GurobiConfig
    thermal: ThermalConfig
    energy: EnergyConfig


def default_config_path() -> Path:
    """返回项目默认配置文件路径。"""
    return Path(__file__).resolve().parents[2] / "configs" / "default_sim.yaml"


def load_config(config_path: str | Path | None = None) -> ConfigBundle:
    """从 YAML 加载配置，缺失字段自动回退到默认值。"""
    path = Path(config_path) if config_path is not None else default_config_path()
    payload = _load_yaml_mapping(path)
    return load_config_from_mapping(payload)


def load_config_from_mapping(payload: Mapping[str, Any] | None) -> ConfigBundle:
    """从字典结构构造配置对象。"""
    data = dict(payload or {})
    simulation_data = _as_mapping(data.get("simulation"), section_name="simulation")
    gurobi_data = _as_mapping(data.get("gurobi"), section_name="gurobi")
    thermal_data = _as_mapping(data.get("thermal"), section_name="thermal")
    energy_data = _as_mapping(data.get("energy"), section_name="energy")

    high_performance_core_count, low_power_core_count = _resolve_core_type_counts(simulation_data)

    simulation = SimulationConfig(
        delta_t_seconds=_read_int(simulation_data, "delta_t_seconds", 60),
        horizon_slots=_read_int(simulation_data, "horizon_slots", 12),
        simulation_slots=_read_int(simulation_data, "simulation_slots", 194),
        high_performance_core_count=high_performance_core_count,
        low_power_core_count=low_power_core_count,
        sunlight_duration_slots=_read_int(simulation_data, "sunlight_duration_slots", 61),
        eclipse_duration_slots=_read_int(simulation_data, "eclipse_duration_slots", 36),
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
            thermal_data, "initial_temperature_celsius", 25.0
        ),
        t_max_celsius=_read_float(thermal_data, "t_max_celsius", 70.0),
        ambient_temperature_celsius=_read_float(
            thermal_data, "ambient_temperature_celsius", 25.0
        ),
        thermal_capacity_j_per_c=_read_float(thermal_data, "thermal_capacity_j_per_c", 3000.0),
        cooling_coeff=_read_float(thermal_data, "cooling_coeff", 0.01),
    )
    energy = EnergyConfig(
        e_max_j=_read_float(energy_data, "e_max_j", 144000.0),
        e_min_j=_read_float(energy_data, "e_min_j", 115200.0),
        initial_energy_j=_read_float(energy_data, "initial_energy_j", 129600.0),
        default_base_power_w=_read_float(energy_data, "default_base_power_w", 3.6),
        default_max_solar_power_w=_read_float(
            energy_data, "default_max_solar_power_w", 14.0
        ),
        solar_power_profile=_read_str(
            energy_data, "solar_power_profile", "half_sine_orbit"
        ),
        solar_power_min_ratio=_read_float(energy_data, "solar_power_min_ratio", 0.0),
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


def _resolve_core_type_counts(simulation_section: Mapping[str, Any]) -> tuple[int, int]:
    """解析两类核心数量，并兼容旧版 core_count 配置。"""
    has_high_key = "high_performance_core_count" in simulation_section
    has_low_key = "low_power_core_count" in simulation_section

    if has_high_key or has_low_key:
        high = _read_int(simulation_section, "high_performance_core_count", 2)
        low = _read_int(simulation_section, "low_power_core_count", 2)
        return high, low

    legacy_core_count = _read_int(simulation_section, "core_count", 4)
    high = legacy_core_count // 2
    low = legacy_core_count - high
    return high, low


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


def _require_ratio(value: float, field_name: str) -> None:
    if not 0 <= value <= 1:
        raise ConfigError(f"{field_name} must be within [0, 1].")


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
