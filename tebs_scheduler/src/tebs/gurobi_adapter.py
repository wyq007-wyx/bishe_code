"""Optional Gurobi adapter for TEBS MILP schedulers."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable

from .config import ConfigBundle, GurobiConfig, load_config
from .models import SolverTrace


class GurobiAdapterError(RuntimeError):
    """Raised when Gurobi setup or solve fails."""


class GurobiUnavailableError(GurobiAdapterError):
    """Raised when gurobipy is not installed in the current environment."""


STATUS_OPTIMAL = "OPTIMAL"
STATUS_TIME_LIMIT = "TIME_LIMIT"
STATUS_INFEASIBLE = "INFEASIBLE"
STATUS_INF_OR_UNBD = "INF_OR_UNBD"
STATUS_UNBOUNDED = "UNBOUNDED"
STATUS_SUBOPTIMAL = "SUBOPTIMAL"
STATUS_INTERRUPTED = "INTERRUPTED"
STATUS_LOADED = "LOADED"
STATUS_UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class GurobiSolveParams:
    """Small, explicit subset of Gurobi parameters used by the project."""

    time_limit_sec: float = 5.0
    mip_gap: float = 0.01
    threads: int = 0
    verbose: bool = False

    @classmethod
    def from_config(cls, config: ConfigBundle | GurobiConfig | None = None) -> "GurobiSolveParams":
        """Create solver parameters from the project config bundle or section."""

        if config is None:
            gurobi_cfg = load_config().gurobi
        elif isinstance(config, ConfigBundle):
            gurobi_cfg = config.gurobi
        else:
            gurobi_cfg = config
        return cls(
            time_limit_sec=gurobi_cfg.gurobi_time_limit,
            mip_gap=gurobi_cfg.gurobi_mip_gap,
            threads=gurobi_cfg.gurobi_threads,
            verbose=gurobi_cfg.gurobi_verbose,
        )

    def __post_init__(self) -> None:
        if self.time_limit_sec <= 0:
            raise GurobiAdapterError("time_limit_sec must be > 0.")
        if self.mip_gap < 0:
            raise GurobiAdapterError("mip_gap must be >= 0.")
        if self.threads < 0:
            raise GurobiAdapterError("threads must be >= 0.")


@dataclass(frozen=True, slots=True)
class GurobiSolveResult:
    """Normalized result returned after optimizing one Gurobi model."""

    status: str
    raw_status: int | None
    solve_time_sec: float
    mip_gap: float | None = None
    objective_value: float | None = None
    best_bound: float | None = None
    node_count: float | None = None
    has_feasible_solution: bool = False
    solver_name: str = "gurobi"
    model_name: str | None = None

    def to_solver_trace(self, *, time_slot: int) -> SolverTrace:
        """Convert the normalized result into the shared SolverTrace model."""

        return SolverTrace(
            time_slot=time_slot,
            solver_name=self.solver_name,
            status=self.status,
            solve_time_sec=self.solve_time_sec,
            mip_gap=self.mip_gap,
            objective_value=self.objective_value,
            best_bound=self.best_bound,
            has_feasible_solution=self.has_feasible_solution,
            node_count=self.node_count,
        )


ModelBuilder = Callable[[Any], Any]


def is_gurobi_available() -> bool:
    """Return True when gurobipy can be imported."""

    return importlib.util.find_spec("gurobipy") is not None


def require_gurobi() -> Any:
    """Import and return gurobipy, raising a project-specific error when absent."""

    try:
        return importlib.import_module("gurobipy")
    except ModuleNotFoundError as exc:
        raise GurobiUnavailableError(
            "gurobipy is not installed; install Gurobi and gurobipy to run MILP schedulers."
        ) from exc


def apply_gurobi_params(model: Any, params: GurobiSolveParams) -> None:
    """Apply the project standard Gurobi parameters to a model."""

    model.setParam("TimeLimit", params.time_limit_sec)
    model.setParam("MIPGap", params.mip_gap)
    model.setParam("OutputFlag", 1 if params.verbose else 0)
    if params.threads > 0:
        model.setParam("Threads", params.threads)


def solve_gurobi_model(
    model_builder: ModelBuilder,
    *,
    params: GurobiSolveParams | None = None,
    model_name: str | None = None,
) -> GurobiSolveResult:
    """Build and optimize a model using a callback that receives gurobipy."""

    gp = require_gurobi()
    try:
        model = model_builder(gp)
        return optimize_gurobi_model(model, params=params, model_name=model_name)
    except GurobiAdapterError:
        raise
    except Exception as exc:
        gurobi_error = getattr(gp, "GurobiError", None)
        if gurobi_error is not None and isinstance(exc, gurobi_error):
            raise GurobiAdapterError(str(exc)) from exc
        raise


def optimize_gurobi_model(
    model: Any,
    *,
    params: GurobiSolveParams | None = None,
    model_name: str | None = None,
) -> GurobiSolveResult:
    """Apply parameters, optimize an existing model and normalize its result."""

    solve_params = params or GurobiSolveParams()
    apply_gurobi_params(model, solve_params)
    try:
        model.optimize()
    except Exception as exc:
        raise GurobiAdapterError(str(exc)) from exc
    return extract_solve_result(model, model_name=model_name)


def extract_solve_result(model: Any, *, model_name: str | None = None) -> GurobiSolveResult:
    """Extract normalized status and scalar solve statistics from a model."""

    raw_status = _safe_attr(model, "Status")
    status = map_gurobi_status(raw_status)
    sol_count = int(_safe_attr(model, "SolCount", 0) or 0)
    has_feasible_solution = status == STATUS_OPTIMAL or sol_count > 0
    objective_value = _safe_float_attr(model, "ObjVal") if has_feasible_solution else None
    mip_gap = _safe_float_attr(model, "MIPGap") if has_feasible_solution else None
    best_bound = _safe_float_attr(model, "ObjBound")
    node_count = _safe_float_attr(model, "NodeCount")
    solve_time = _safe_float_attr(model, "Runtime") or 0.0
    return GurobiSolveResult(
        status=status,
        raw_status=int(raw_status) if raw_status is not None else None,
        solve_time_sec=solve_time,
        mip_gap=mip_gap,
        objective_value=objective_value,
        best_bound=best_bound,
        node_count=node_count,
        has_feasible_solution=has_feasible_solution,
        model_name=model_name,
    )


def map_gurobi_status(status_code: int | None) -> str:
    """Map Gurobi numeric status codes to stable project status strings."""

    if status_code is None:
        return STATUS_UNKNOWN
    return _STATUS_BY_CODE.get(int(status_code), STATUS_UNKNOWN)


def _safe_attr(model: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(model, name)
    except Exception:
        return default


def _safe_float_attr(model: Any, name: str) -> float | None:
    value = _safe_attr(model, name)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


_STATUS_BY_CODE = {
    1: STATUS_LOADED,
    2: STATUS_OPTIMAL,
    3: STATUS_INFEASIBLE,
    4: STATUS_INF_OR_UNBD,
    5: STATUS_UNBOUNDED,
    9: STATUS_TIME_LIMIT,
    11: STATUS_INTERRUPTED,
    13: STATUS_SUBOPTIMAL,
}


__all__ = [
    "GurobiAdapterError",
    "GurobiSolveParams",
    "GurobiSolveResult",
    "GurobiUnavailableError",
    "STATUS_INF_OR_UNBD",
    "STATUS_INFEASIBLE",
    "STATUS_INTERRUPTED",
    "STATUS_LOADED",
    "STATUS_OPTIMAL",
    "STATUS_SUBOPTIMAL",
    "STATUS_TIME_LIMIT",
    "STATUS_UNKNOWN",
    "STATUS_UNBOUNDED",
    "apply_gurobi_params",
    "extract_solve_result",
    "is_gurobi_available",
    "map_gurobi_status",
    "optimize_gurobi_model",
    "require_gurobi",
    "solve_gurobi_model",
]
