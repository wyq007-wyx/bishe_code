from __future__ import annotations

import pytest

from tebs.config import ConfigBundle, ConfigError, load_config


def test_round2_default_values() -> None:
    cfg = load_config()

    assert isinstance(cfg, ConfigBundle)
    assert cfg.simulation.delta_t_seconds == 60
    assert cfg.simulation.horizon_slots == 12
    assert cfg.gurobi.gurobi_time_limit == 5.0
    assert cfg.simulation.energy_unit == "J"


def test_missing_sections_fallback_to_reasonable_defaults(tmp_path) -> None:
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
    missing = tmp_path / "not_found.yaml"
    with pytest.raises(ConfigError, match="Configuration file not found"):
        load_config(missing)


def test_invalid_values_raise_clear_error(tmp_path) -> None:
    cfg_path = tmp_path / "bad_config.yaml"
    cfg_path.write_text(
        "simulation:\n  delta_t_seconds: -1\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="delta_t_seconds must be > 0"):
        load_config(cfg_path)
