"""Classify a CVXPY problem by its cone type."""

from __future__ import annotations

import cvxpy as cp
from cvxpy.constraints.exponential import ExpCone
from cvxpy.constraints.psd import PSD
from cvxpy.constraints.second_order import SOC


def classify_problem(problem: cp.Problem) -> str:
    """Return a short label for the problem type.

    Returns one of: ``"MIP"``, ``"SDP"``, ``"ECP"``, ``"SOCP"``, ``"QP"``,
    ``"LP"``.
    """
    if problem.is_mixed_integer():
        return "MIP"

    cone_types = {type(c) for c in problem.constraints}

    if PSD in cone_types:
        return "SDP"
    if ExpCone in cone_types:
        return "ECP"
    if SOC in cone_types:
        return "SOCP"
    obj = problem.objective.expr
    if obj.is_quadratic() and not obj.is_affine():
        return "QP"
    return "LP"
