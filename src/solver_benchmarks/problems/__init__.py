from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass, field
from typing import Callable

import cvxpy as cp
from .finance_NCM import NearestCorrelationMatrix

@dataclass
class ProblemSpec:
    name: str
    func: Callable[[int], cp.Problem]
    tags: list[str] = field(default_factory=list)
    description: str = ""


_REGISTRY: dict[str, ProblemSpec] = {}
_discovered = False


def register_problem(
    name: str,
    tags: list[str] | None = None,
    description: str = "",
):
    """Decorator to register a problem factory function.

    The decorated function must accept a single `seed` argument and return
    a ``cp.Problem``.
    """

    def decorator(func: Callable[[int], cp.Problem]) -> Callable[[int], cp.Problem]:
        spec = ProblemSpec(
            name=name,
            func=func,
            tags=tags or [],
            description=description,
        )
        _REGISTRY[name] = spec
        return func

    return decorator


def _discover():
    global _discovered
    if _discovered:
        return
    # Import all sub-modules of this package so decorators execute.
    package = importlib.import_module(__name__)
    for info in pkgutil.walk_packages(package.__path__, prefix=package.__name__ + "."):
        importlib.import_module(info.name)
    _discovered = True


def list_problems(tag: str | None = None) -> list[ProblemSpec]:
    """Return registered problems, optionally filtered by tag."""
    _discover()
    specs = list(_REGISTRY.values())
    if tag is not None:
        specs = [s for s in specs if tag in s.tags]
    return sorted(specs, key=lambda s: s.name)


def get_problem(name: str) -> ProblemSpec:
    """Look up a problem by name."""
    _discover()
    return _REGISTRY[name]

