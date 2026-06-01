"""Task-set generation utilities for TEBS experiments."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import ConfigBundle, ConfigError, load_config
from .models import (
    HIGH_PERFORMANCE_CORE_TYPE,
    LOW_POWER_CORE_TYPE,
    Core,
    Task,
    TaskBlock,
    build_cores_from_type_counts,
)


class TaskFactoryError(ValueError):
    """Raised when task-set generation inputs are invalid."""


def _require_non_empty_str(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise TaskFactoryError(f"{field_name} must be a non-empty string.")


def _require_positive_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TaskFactoryError(f"{field_name} must be a positive integer.")


def _require_positive_number(value: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise TaskFactoryError(f"{field_name} must be > 0.")


def _require_ratio(value: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise TaskFactoryError(f"{field_name} must be within [0, 1].")


def _require_int_range(
    value: tuple[int, int],
    field_name: str,
    *,
    minimum: int,
) -> None:
    start, end = value
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
        or start < minimum
        or end < start
    ):
        raise TaskFactoryError(
            f"{field_name} must be an integer range with {minimum} <= min <= max."
        )


def _require_number_range(
    value: tuple[float, float],
    field_name: str,
    *,
    minimum: float,
) -> None:
    start, end = value
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, (int, float))
        or not isinstance(end, (int, float))
        or start <= minimum
        or end < start
    ):
        raise TaskFactoryError(
            f"{field_name} must be a numeric range with {minimum} < min <= max."
        )


@dataclass(frozen=True, slots=True)
class TaskTypeSpec:
    """Static profile for one onboard payload-processing task type."""

    task_type_id: str
    name: str
    block_count_range: tuple[int, int]
    weight: float
    high_power: bool
    hp_duration_range: tuple[int, int]
    hp_power_range_w: tuple[float, float]
    low_power_duration_multiplier: float
    low_power_power_multiplier: float
    deadline_slack_range: tuple[int, int]

    def __post_init__(self) -> None:
        _require_non_empty_str(self.task_type_id, "task_type_id")
        _require_non_empty_str(self.name, "name")
        _require_int_range(self.block_count_range, "block_count_range", minimum=1)
        _require_positive_number(self.weight, "weight")
        _require_int_range(self.hp_duration_range, "hp_duration_range", minimum=1)
        _require_number_range(self.hp_power_range_w, "hp_power_range_w", minimum=0.0)
        _require_positive_number(
            self.low_power_duration_multiplier, "low_power_duration_multiplier"
        )
        _require_positive_number(self.low_power_power_multiplier, "low_power_power_multiplier")
        _require_int_range(self.deadline_slack_range, "deadline_slack_range", minimum=0)


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    """Scenario-level task count, pressure and type-weight settings."""

    scenario_name: str
    simulation_slots: int
    task_count_range: tuple[int, int]
    high_power_task_ratio: float
    description: str = ""
    task_type_weights: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty_str(self.scenario_name, "scenario_name")
        _require_positive_int(self.simulation_slots, "simulation_slots")
        _require_int_range(self.task_count_range, "task_count_range", minimum=1)
        _require_ratio(self.high_power_task_ratio, "high_power_task_ratio")
        for task_type_id, weight in self.task_type_weights:
            _require_non_empty_str(task_type_id, "task_type_id in task_type_weights")
            _require_positive_number(weight, f"task_type_weights[{task_type_id}]")

    @property
    def weight_by_task_type(self) -> dict[str, float]:
        """Return scenario-specific sampling weights keyed by task type ID."""
        return dict(self.task_type_weights)


TASK_TYPE_SPECS: Mapping[str, TaskTypeSpec] = {
    "T1": TaskTypeSpec(
        task_type_id="T1",
        name="cloud_detection",
        block_count_range=(4, 8),
        weight=2.0,
        high_power=False,
        hp_duration_range=(2, 4),
        hp_power_range_w=(7.0, 10.0),
        low_power_duration_multiplier=1.6,
        low_power_power_multiplier=0.65,
        deadline_slack_range=(8, 20),
    ),
    "T2": TaskTypeSpec(
        task_type_id="T2",
        name="cloud_removal_and_compression",
        block_count_range=(4, 4),
        weight=2.0,
        high_power=False,
        hp_duration_range=(2, 5),
        hp_power_range_w=(6.0, 11.0),
        low_power_duration_multiplier=1.65,
        low_power_power_multiplier=0.62,
        deadline_slack_range=(8, 18),
    ),
    "T3": TaskTypeSpec(
        task_type_id="T3",
        name="sar_vessel_detection",
        block_count_range=(6, 10),
        weight=5.0,
        high_power=True,
        hp_duration_range=(4, 7),
        hp_power_range_w=(16.0, 24.0),
        low_power_duration_multiplier=1.9,
        low_power_power_multiplier=0.68,
        deadline_slack_range=(6, 14),
    ),
    "T4": TaskTypeSpec(
        task_type_id="T4",
        name="optical_object_detection",
        block_count_range=(4, 8),
        weight=3.0,
        high_power=False,
        hp_duration_range=(3, 5),
        hp_power_range_w=(10.0, 15.0),
        low_power_duration_multiplier=1.7,
        low_power_power_multiplier=0.66,
        deadline_slack_range=(7, 16),
    ),
    "T5": TaskTypeSpec(
        task_type_id="T5",
        name="change_detection",
        block_count_range=(4, 6),
        weight=5.0,
        high_power=True,
        hp_duration_range=(3, 6),
        hp_power_range_w=(15.0, 22.0),
        low_power_duration_multiplier=1.85,
        low_power_power_multiplier=0.67,
        deadline_slack_range=(5, 12),
    ),
    "T6": TaskTypeSpec(
        task_type_id="T6",
        name="image_compression",
        block_count_range=(3, 6),
        weight=1.0,
        high_power=False,
        hp_duration_range=(1, 3),
        hp_power_range_w=(4.0, 8.0),
        low_power_duration_multiplier=1.55,
        low_power_power_multiplier=0.58,
        deadline_slack_range=(12, 26),
    ),
    "T7": TaskTypeSpec(
        task_type_id="T7",
        name="sar_backend_processing",
        block_count_range=(6, 10),
        weight=4.0,
        high_power=True,
        hp_duration_range=(5, 8),
        hp_power_range_w=(18.0, 26.0),
        low_power_duration_multiplier=1.95,
        low_power_power_multiplier=0.7,
        deadline_slack_range=(8, 16),
    ),
}

HIGH_POWER_TASK_TYPE_IDS = frozenset(
    task_type_id for task_type_id, spec in TASK_TYPE_SPECS.items() if spec.high_power
)


def default_task_type_specs() -> dict[str, TaskTypeSpec]:
    """Return the built-in seven task profiles keyed by task type ID."""
    return dict(TASK_TYPE_SPECS)


def default_scenarios_path() -> Path:
    """Return the default YAML path that stores scenario generation settings."""
    return Path(__file__).resolve().parents[2] / "configs" / "scenarios.yaml"


def load_scenario_specs(scenarios_path: str | Path | None = None) -> dict[str, ScenarioSpec]:
    """Load scenario profiles, merging YAML values onto built-in defaults."""
    scenarios = _builtin_scenario_specs()
    path = Path(scenarios_path) if scenarios_path is not None else default_scenarios_path()
    if not path.exists():
        return scenarios

    payload = _load_yaml_mapping(path)
    raw_scenarios = payload.get("scenarios", {})
    if not isinstance(raw_scenarios, Mapping):
        raise TaskFactoryError("Section 'scenarios' must be a mapping.")

    for scenario_name, raw_data in raw_scenarios.items():
        _require_non_empty_str(str(scenario_name), "scenario_name")
        if not isinstance(raw_data, Mapping):
            raise TaskFactoryError(f"Scenario '{scenario_name}' must be a mapping.")

        base = scenarios.get(str(scenario_name))
        generic = base or ScenarioSpec(
            scenario_name=str(scenario_name),
            simulation_slots=180,
            task_count_range=(16, 28),
            high_power_task_ratio=0.45,
            description="Generated from scenarios.yaml.",
            task_type_weights=_mixed_type_weights(),
        )
        scenarios[str(scenario_name)] = ScenarioSpec(
            scenario_name=str(scenario_name),
            simulation_slots=_read_int(raw_data, "simulation_slots", generic.simulation_slots),
            task_count_range=_read_int_pair(
                raw_data, "task_count_range", generic.task_count_range, minimum=1
            ),
            high_power_task_ratio=_read_float(
                raw_data, "high_power_task_ratio", generic.high_power_task_ratio
            ),
            description=_read_str(raw_data, "description", generic.description),
            task_type_weights=generic.task_type_weights,
        )
    return scenarios


def normalize_scenario_name(scenario_name: str) -> str:
    """Normalize short aliases such as 'S1' or 'stress' to canonical names."""
    _require_non_empty_str(scenario_name, "scenario_name")
    name = scenario_name.strip()
    alias = {
        "s1": "S1_light",
        "light": "S1_light",
        "s1_light": "S1_light",
        "s2": "S2_mixed",
        "mixed": "S2_mixed",
        "s2_mixed": "S2_mixed",
        "s3": "S3_stress",
        "stress": "S3_stress",
        "high_pressure": "S3_stress",
        "s3_stress": "S3_stress",
        "s5": "S5_large",
        "large": "S5_large",
        "s5_large": "S5_large",
    }
    return alias.get(name.lower(), name)


def generate_task_set(
    scenario_name: str = "S2_mixed",
    num_tasks: int | None = None,
    random_seed: int | None = None,
    core_specs: Sequence[Core] | None = None,
    config: ConfigBundle | None = None,
    scenarios_path: str | Path | None = None,
) -> list[Task]:
    """Generate a reproducible list of TEBS tasks for one experiment scenario."""
    rng = random.Random(random_seed)
    scenario = _resolve_scenario(scenario_name, scenarios_path)
    task_count = _resolve_task_count(num_tasks, scenario, rng)
    cores = _resolve_cores(core_specs, config)
    task_type_ids = _draw_task_type_ids(task_count, scenario, rng)

    tasks = [
        _build_task(
            task_index=index,
            task_type_spec=TASK_TYPE_SPECS[task_type_id],
            scenario=scenario,
            cores=cores,
            rng=rng,
        )
        for index, task_type_id in enumerate(task_type_ids)
    ]
    return sorted(tasks, key=lambda task: (task.release_time, task.deadline, task.task_id))


def task_type_id_from_task_id(task_id: str) -> str:
    """Extract the task type ID from task IDs emitted by this factory."""
    _require_non_empty_str(task_id, "task_id")
    task_type_id = task_id.split("_", 1)[0]
    if task_type_id not in TASK_TYPE_SPECS:
        raise TaskFactoryError(f"Unknown task type ID in task_id: {task_id}")
    return task_type_id


def task_type_id_from_task(task: Task) -> str:
    """Extract the task type ID from a generated Task object."""
    return task_type_id_from_task_id(task.task_id)


def is_high_power_task_type(task_type_id: str) -> bool:
    """Return whether a task type belongs to the high-power task group."""
    if task_type_id not in TASK_TYPE_SPECS:
        raise TaskFactoryError(f"Unknown task type ID: {task_type_id}")
    return TASK_TYPE_SPECS[task_type_id].high_power


def count_high_power_tasks(tasks: Sequence[Task]) -> int:
    """Count generated tasks whose type profile is high-power/high-thermal."""
    return sum(1 for task in tasks if is_high_power_task_type(task_type_id_from_task(task)))


def high_power_task_ratio(tasks: Sequence[Task]) -> float:
    """Return the observed high-power task ratio for a generated task set."""
    if not tasks:
        return 0.0
    return count_high_power_tasks(tasks) / len(tasks)


def _builtin_scenario_specs() -> dict[str, ScenarioSpec]:
    return {
        "S1_light": ScenarioSpec(
            scenario_name="S1_light",
            simulation_slots=120,
            description="Light-load smoke and low-pressure scenario.",
            task_count_range=(8, 16),
            high_power_task_ratio=0.2,
            task_type_weights=(
                ("T1", 4.0),
                ("T2", 1.0),
                ("T3", 1.0),
                ("T4", 2.0),
                ("T5", 1.0),
                ("T6", 4.0),
                ("T7", 0.5),
            ),
        ),
        "S2_mixed": ScenarioSpec(
            scenario_name="S2_mixed",
            simulation_slots=180,
            description="Default mixed-load comparison scenario.",
            task_count_range=(16, 28),
            high_power_task_ratio=0.45,
            task_type_weights=_mixed_type_weights(),
        ),
        "S3_stress": ScenarioSpec(
            scenario_name="S3_stress",
            simulation_slots=240,
            description="High-pressure scenario with more SAR/change workloads.",
            task_count_range=(28, 40),
            high_power_task_ratio=0.7,
            task_type_weights=(
                ("T1", 1.0),
                ("T2", 1.0),
                ("T3", 3.0),
                ("T4", 1.0),
                ("T5", 3.0),
                ("T6", 0.8),
                ("T7", 3.0),
            ),
        ),
        "S5_large": ScenarioSpec(
            scenario_name="S5_large",
            simulation_slots=480,
            description="Large-scale task set for later learning-based experiments.",
            task_count_range=(100, 200),
            high_power_task_ratio=0.5,
            task_type_weights=_mixed_type_weights(),
        ),
    }


def _mixed_type_weights() -> tuple[tuple[str, float], ...]:
    return (
        ("T1", 1.0),
        ("T2", 1.0),
        ("T3", 1.0),
        ("T4", 1.0),
        ("T5", 1.0),
        ("T6", 1.0),
        ("T7", 1.0),
    )


def _resolve_scenario(
    scenario_name: str,
    scenarios_path: str | Path | None,
) -> ScenarioSpec:
    normalized_name = normalize_scenario_name(scenario_name)
    scenarios = load_scenario_specs(scenarios_path)
    if normalized_name not in scenarios:
        available = ", ".join(sorted(scenarios))
        raise TaskFactoryError(
            f"Unknown scenario_name '{scenario_name}'. Available scenarios: {available}"
        )
    return scenarios[normalized_name]


def _resolve_task_count(
    num_tasks: int | None,
    scenario: ScenarioSpec,
    rng: random.Random,
) -> int:
    if num_tasks is not None:
        if isinstance(num_tasks, bool) or not isinstance(num_tasks, int) or num_tasks <= 0:
            raise TaskFactoryError("num_tasks must be a positive integer.")
        return num_tasks
    return rng.randint(scenario.task_count_range[0], scenario.task_count_range[1])


def _resolve_cores(
    core_specs: Sequence[Core] | None,
    config: ConfigBundle | None,
) -> tuple[Core, ...]:
    if core_specs is not None:
        cores = tuple(core_specs)
    else:
        cfg = config
        if cfg is None:
            try:
                cfg = load_config()
            except ConfigError as exc:
                raise TaskFactoryError(str(exc)) from exc
        cores = tuple(
            build_cores_from_type_counts(
                cfg.simulation.high_performance_core_count,
                cfg.simulation.low_power_core_count,
                high_performance_speed_scale=1.0,
                high_performance_power_scale=1.0,
                low_power_speed_scale=1.0,
                low_power_power_scale=1.0,
            )
        )

    if not cores:
        raise TaskFactoryError("core_specs must include at least one Core.")
    core_ids = [core.core_id for core in cores]
    if len(set(core_ids)) != len(core_ids):
        raise TaskFactoryError("core_specs must contain unique core_id values.")
    return cores


def _draw_task_type_ids(
    task_count: int,
    scenario: ScenarioSpec,
    rng: random.Random,
) -> list[str]:
    high_count = int(task_count * scenario.high_power_task_ratio + 0.5)
    high_count = max(0, min(task_count, high_count))
    regular_count = task_count - high_count

    high_type_ids = [task_type_id for task_type_id in TASK_TYPE_SPECS if task_type_id in HIGH_POWER_TASK_TYPE_IDS]
    regular_type_ids = [
        task_type_id for task_type_id in TASK_TYPE_SPECS if task_type_id not in HIGH_POWER_TASK_TYPE_IDS
    ]
    task_type_ids = [
        _weighted_choice(high_type_ids, scenario, rng) for _ in range(high_count)
    ]
    task_type_ids.extend(
        _weighted_choice(regular_type_ids, scenario, rng) for _ in range(regular_count)
    )
    rng.shuffle(task_type_ids)
    return task_type_ids


def _weighted_choice(
    candidate_type_ids: Sequence[str],
    scenario: ScenarioSpec,
    rng: random.Random,
) -> str:
    weights_by_type = scenario.weight_by_task_type
    weighted_candidates = [
        (task_type_id, weights_by_type.get(task_type_id, 1.0))
        for task_type_id in candidate_type_ids
    ]
    total_weight = sum(weight for _, weight in weighted_candidates)
    if total_weight <= 0:
        raise TaskFactoryError("Task type weights must sum to a positive value.")

    threshold = rng.uniform(0.0, total_weight)
    cumulative = 0.0
    for task_type_id, weight in weighted_candidates:
        cumulative += weight
        if threshold <= cumulative:
            return task_type_id
    return weighted_candidates[-1][0]


def _build_task(
    *,
    task_index: int,
    task_type_spec: TaskTypeSpec,
    scenario: ScenarioSpec,
    cores: Sequence[Core],
    rng: random.Random,
) -> Task:
    task_id = f"{task_type_spec.task_type_id}_{task_index:04d}"
    release_time = rng.randint(0, max(0, scenario.simulation_slots - 1))
    block_count = rng.randint(
        task_type_spec.block_count_range[0],
        task_type_spec.block_count_range[1],
    )
    duration_maps: list[dict[str, int]] = []
    power_maps: list[dict[str, float]] = []
    for _ in range(block_count):
        duration_by_core, power_by_core = _draw_block_core_profiles(task_type_spec, cores, rng)
        duration_maps.append(duration_by_core)
        power_maps.append(power_by_core)

    fastest_total_duration = sum(min(duration_by_core.values()) for duration_by_core in duration_maps)
    slack = rng.randint(
        task_type_spec.deadline_slack_range[0],
        task_type_spec.deadline_slack_range[1],
    )
    deadline = min(scenario.simulation_slots - 1, release_time + fastest_total_duration + slack)
    deadline = max(release_time, deadline)
    core_ids = tuple(core.core_id for core in cores)

    blocks = [
        TaskBlock(
            task_id=task_id,
            block_id=block_id,
            release_time=release_time,
            deadline=deadline,
            weight=task_type_spec.weight,
            duration_by_core=duration_maps[block_id],
            power_by_core=power_maps[block_id],
            predecessor_block_id=block_id - 1 if block_id > 0 else None,
        )
        for block_id in range(block_count)
    ]
    return Task(
        task_id=task_id,
        release_time=release_time,
        deadline=deadline,
        weight=task_type_spec.weight,
        blocks=blocks,
        core_ids=core_ids,
        require_uniform_block_core_ids=True,
    )


def _draw_block_core_profiles(
    task_type_spec: TaskTypeSpec,
    cores: Sequence[Core],
    rng: random.Random,
) -> tuple[dict[str, int], dict[str, float]]:
    base_duration = rng.randint(
        task_type_spec.hp_duration_range[0],
        task_type_spec.hp_duration_range[1],
    )
    base_power = rng.uniform(
        task_type_spec.hp_power_range_w[0],
        task_type_spec.hp_power_range_w[1],
    )
    stage_jitter = rng.uniform(0.85, 1.15)
    duration_by_core: dict[str, int] = {}
    power_by_core: dict[str, float] = {}

    for core in cores:
        if core.core_type == HIGH_PERFORMANCE_CORE_TYPE:
            duration_multiplier = 1.0
            power_multiplier = 1.0
        elif core.core_type == LOW_POWER_CORE_TYPE:
            duration_multiplier = task_type_spec.low_power_duration_multiplier
            power_multiplier = task_type_spec.low_power_power_multiplier
        else:
            raise TaskFactoryError(f"Unsupported core_type: {core.core_type}")

        duration = math.ceil(base_duration * stage_jitter * duration_multiplier / core.speed_scale)
        power = base_power * power_multiplier * core.power_scale
        duration_by_core[core.core_id] = max(1, duration)
        power_by_core[core.core_id] = round(max(0.001, power), 6)

    return duration_by_core, power_by_core


def _load_yaml_mapping(path: Path) -> Mapping[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise TaskFactoryError(
            "PyYAML is required to load scenario configuration. Install with: pip install pyyaml"
        ) from exc

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise TaskFactoryError(f"Scenario configuration root must be a mapping in: {path}")
    return payload


def _read_int(section: Mapping[str, Any], key: str, default: int) -> int:
    raw = section.get(key, default)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise TaskFactoryError(f"{key} must be an integer.")
    return raw


def _read_float(section: Mapping[str, Any], key: str, default: float) -> float:
    raw = section.get(key, default)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise TaskFactoryError(f"{key} must be a number.")
    return float(raw)


def _read_str(section: Mapping[str, Any], key: str, default: str) -> str:
    raw = section.get(key, default)
    if not isinstance(raw, str):
        raise TaskFactoryError(f"{key} must be a string.")
    return raw


def _read_int_pair(
    section: Mapping[str, Any],
    key: str,
    default: tuple[int, int],
    *,
    minimum: int,
) -> tuple[int, int]:
    raw = section.get(key, default)
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise TaskFactoryError(f"{key} must be a two-item sequence.")
    pair = (raw[0], raw[1])
    _require_int_range(pair, key, minimum=minimum)
    return pair


__all__ = [
    "HIGH_POWER_TASK_TYPE_IDS",
    "TASK_TYPE_SPECS",
    "ScenarioSpec",
    "TaskFactoryError",
    "TaskTypeSpec",
    "count_high_power_tasks",
    "default_scenarios_path",
    "default_task_type_specs",
    "generate_task_set",
    "high_power_task_ratio",
    "is_high_power_task_type",
    "load_scenario_specs",
    "normalize_scenario_name",
    "task_type_id_from_task",
    "task_type_id_from_task_id",
]
