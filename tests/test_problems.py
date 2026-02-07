"""Validate that all registered problems are well-formed."""

from __future__ import annotations

import cvxpy as cp

from solver_benchmarks.problems import list_problems


def test_all_problems_return_cp_problem():
    """Every registered problem factory must return a cp.Problem."""
    problems = list_problems()
    assert len(problems) > 0, "No problems registered"
    for spec in problems:
        problem = spec.func(0)
        assert isinstance(problem, cp.Problem), (
            f"{spec.name} returned {type(problem)}, expected cp.Problem"
        )


def test_all_problems_have_tags():
    """Every problem should have at least one tag."""
    for spec in list_problems():
        assert len(spec.tags) > 0, f"{spec.name} has no tags"


def test_problem_names_unique():
    """Problem names must be unique."""
    names = [spec.name for spec in list_problems()]
    assert len(names) == len(set(names)), f"Duplicate problem names: {names}"


def test_problem_names_have_type_prefix():
    """Problem names should follow the type/name convention."""
    for spec in list_problems():
        assert "/" in spec.name, f"{spec.name} missing type prefix (expected 'type/name')"


def test_problems_have_type_tag():
    """Each problem should have a type tag matching its prefix."""
    valid_types = {"lp", "qp", "socp", "sdp", "mip"}
    for spec in list_problems():
        prefix = spec.name.split("/")[0]
        assert prefix in valid_types, f"{spec.name} has unknown prefix '{prefix}'"
        assert prefix in spec.tags, f"{spec.name} missing type tag '{prefix}'"


def test_problems_are_reproducible():
    """Calling with the same seed should produce the same problem size."""
    for spec in list_problems():
        p1 = spec.func(0)
        p2 = spec.func(0)
        assert (
            p1.size_metrics.num_scalar_variables
            == p2.size_metrics.num_scalar_variables
        ), f"{spec.name} not reproducible"
