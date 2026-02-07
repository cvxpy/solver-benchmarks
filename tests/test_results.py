"""Validate result serialization and .jsonl files."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from solver_benchmarks.results import BenchmarkResult, load_results, save_results, load_all_results


def test_round_trip():
    """Results survive serialization and deserialization."""
    r = BenchmarkResult(
        problem_name="test/foo",
        solver_name="SCS",
        solve_time=1.23,
        status="optimal",
        objective_value=42.0,
        problem_type="LP",
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.jsonl"
        save_results([r], path)
        loaded = load_results(path)
        assert len(loaded) == 1
        assert loaded[0].problem_name == "test/foo"
        assert loaded[0].solver_name == "SCS"
        assert loaded[0].solve_time == 1.23
        assert loaded[0].status == "optimal"
        assert loaded[0].objective_value == 42.0


def test_load_all_results():
    """load_all_results finds all .jsonl files in a directory."""
    r1 = BenchmarkResult(problem_name="a", solver_name="X", status="optimal")
    r2 = BenchmarkResult(problem_name="b", solver_name="Y", status="optimal")
    with tempfile.TemporaryDirectory() as tmpdir:
        save_results([r1], Path(tmpdir) / "file1.jsonl")
        save_results([r2], Path(tmpdir) / "file2.jsonl")
        all_results = load_all_results(tmpdir)
        assert len(all_results) == 2
        names = {r.problem_name for r in all_results}
        assert names == {"a", "b"}


def test_jsonl_format():
    """Each line in the file is valid JSON."""
    r = BenchmarkResult(problem_name="test/bar", solver_name="OSQP")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.jsonl"
        save_results([r], path)
        with open(path) as f:
            for line in f:
                obj = json.loads(line)
                assert isinstance(obj, dict)
                assert "problem_name" in obj


def test_unknown_fields_ignored():
    """Extra fields in JSON should be ignored gracefully."""
    data = {"problem_name": "x", "solver_name": "Y", "unknown_field": 999}
    r = BenchmarkResult.from_dict(data)
    assert r.problem_name == "x"
    assert r.solver_name == "Y"


def test_existing_jsonl_files():
    """All .jsonl files in results/ should parse correctly."""
    results_dir = Path(__file__).parent.parent / "results"
    if not results_dir.exists():
        return
    for path in results_dir.glob("*.jsonl"):
        results = load_results(path)
        for r in results:
            assert r.problem_name, f"Empty problem_name in {path}"
            assert r.solver_name, f"Empty solver_name in {path}"
