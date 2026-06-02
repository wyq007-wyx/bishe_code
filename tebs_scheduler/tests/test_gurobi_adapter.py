from __future__ import annotations

import pytest

from tebs.gurobi_adapter import (
    GurobiAdapterError,
    GurobiSolveParams,
    STATUS_INFEASIBLE,
    STATUS_OPTIMAL,
    STATUS_TIME_LIMIT,
    apply_gurobi_params,
    map_gurobi_status,
    solve_gurobi_model,
)


class _FakeModel:
    def __init__(self) -> None:
        self.params: dict[str, float | int] = {}

    def setParam(self, name: str, value: float | int) -> None:
        self.params[name] = value


def test_apply_gurobi_params_sets_time_limit_and_gap() -> None:
    model = _FakeModel()

    apply_gurobi_params(
        model,
        GurobiSolveParams(time_limit_sec=3.5, mip_gap=0.02, threads=2, verbose=True),
    )

    assert model.params["TimeLimit"] == 3.5
    assert model.params["MIPGap"] == 0.02
    assert model.params["Threads"] == 2
    assert model.params["OutputFlag"] == 1


def test_status_mapping_covers_core_gurobi_statuses() -> None:
    assert map_gurobi_status(2) == STATUS_OPTIMAL
    assert map_gurobi_status(3) == STATUS_INFEASIBLE
    assert map_gurobi_status(9) == STATUS_TIME_LIMIT


def test_toy_milp_solves_when_gurobi_is_available() -> None:
    pytest.importorskip("gurobipy")

    def _builder(gp):
        model = gp.Model("adapter_toy_milp")
        x = model.addVar(vtype=gp.GRB.BINARY, name="x")
        model.addConstr(x >= 1, name="force_x")
        model.setObjective(x, gp.GRB.MINIMIZE)
        return model

    try:
        result = solve_gurobi_model(
            _builder,
            params=GurobiSolveParams(time_limit_sec=5.0, mip_gap=0.0, verbose=False),
            model_name="adapter_toy_milp",
        )
    except GurobiAdapterError as exc:
        pytest.skip(f"Gurobi is installed but not usable in this environment: {exc}")

    assert result.status == STATUS_OPTIMAL
    assert result.has_feasible_solution is True
    assert result.objective_value == pytest.approx(1.0)
