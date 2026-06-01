from __future__ import annotations

import pytest

from tebs.config import ConfigBundle, ConfigError, load_config


def test_round2_default_values() -> None:
    """验证默认配置文件可加载，且关键默认值符合预期。"""
    cfg = load_config()

    assert isinstance(cfg, ConfigBundle)
    assert cfg.simulation.delta_t_seconds == 60
    assert cfg.simulation.horizon_slots == 12
    assert cfg.simulation.high_performance_core_count == 2
    assert cfg.simulation.low_power_core_count == 2
    assert cfg.simulation.core_count == 4
    assert cfg.gurobi.gurobi_time_limit == 5.0
    assert cfg.simulation.energy_unit == "J"


def test_missing_sections_fallback_to_reasonable_defaults(tmp_path) -> None:
    """验证配置缺失部分 section 时，系统会回退到合理默认值。"""
    cfg_path = tmp_path / "partial_config.yaml"
    cfg_path.write_text(
        "simulation:\n  simulation_slots: 180\n",
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)
    assert cfg.simulation.simulation_slots == 180
    assert cfg.simulation.delta_t_seconds == 60
    assert cfg.gurobi.gurobi_time_limit == 5.0
    assert cfg.energy.e_min_j == 36000.0


def test_missing_file_raises_clear_error(tmp_path) -> None:
    """验证配置文件不存在时，会抛出清晰的错误信息。"""
    missing = tmp_path / "not_found.yaml"
    with pytest.raises(ConfigError, match="Configuration file not found"):
        load_config(missing)


def test_invalid_values_raise_clear_error(tmp_path) -> None:
    """验证配置取值非法时，会抛出可定位字段的错误信息。"""
    cfg_path = tmp_path / "bad_config.yaml"
    cfg_path.write_text(
        "simulation:\n  delta_t_seconds: -1\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="delta_t_seconds must be > 0"):
        load_config(cfg_path)


def test_core_count_config_supports_new_and_legacy_keys(tmp_path) -> None:
    """验证核心数量配置同时兼容新版分类型字段与旧版 core_count 字段。"""
    # 新版：显式配置两类核心数量。
    new_cfg = tmp_path / "new_core_layout.yaml"
    new_cfg.write_text(
        "simulation:\n"
        "  high_performance_core_count: 3\n"
        "  low_power_core_count: 1\n",
        encoding="utf-8",
    )
    cfg_new = load_config(new_cfg)
    assert cfg_new.simulation.high_performance_core_count == 3
    assert cfg_new.simulation.low_power_core_count == 1
    assert cfg_new.simulation.core_count == 4

    # 旧版：仅配置 core_count 时仍可兼容解析。
    legacy_cfg = tmp_path / "legacy_core_layout.yaml"
    legacy_cfg.write_text(
        "simulation:\n"
        "  core_count: 5\n",
        encoding="utf-8",
    )
    cfg_legacy = load_config(legacy_cfg)
    assert cfg_legacy.simulation.high_performance_core_count == 2
    assert cfg_legacy.simulation.low_power_core_count == 3
    assert cfg_legacy.simulation.core_count == 5
