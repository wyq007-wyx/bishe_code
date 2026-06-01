from __future__ import annotations

import importlib
import pkgutil

import tebs


def _iter_tebs_modules() -> list[str]:
    modules = [tebs.__name__]
    modules.extend(
        module.name
        for module in pkgutil.walk_packages(tebs.__path__, prefix=f"{tebs.__name__}.")
    )
    return modules


def test_tebs_modules_are_importable() -> None:
    for module_name in _iter_tebs_modules():
        importlib.import_module(module_name)
