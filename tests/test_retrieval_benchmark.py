"""Tests for the local FTS5 retrieval benchmark and dense-pilot decision gate."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from local_ai_agent.config import load_settings
from local_ai_agent.evaluation.retrieval_benchmark import (
    FTS5_BENCHMARK_CASES,
    RetrievalBenchmarkThresholds,
    assess_fts5_benchmark,
    run_fts5_benchmark,
)


def test_fts5_benchmark_uses_sanitized_fixed_cases_and_reports_quality_latency_and_memory(
    tmp_path: Path,
) -> None:
    result = run_fts5_benchmark(tmp_path / "benchmark.db")

    assert result.case_count == len(FTS5_BENCHMARK_CASES)
    assert result.precision_at_k == 1.0
    assert result.recall_at_k == 1.0
    assert result.mean_latency_ms >= 0.0
    assert result.peak_memory_bytes > 0
    assert all(case.query and case.relevant_keys for case in FTS5_BENCHMARK_CASES)
    assert "secret" not in str(FTS5_BENCHMARK_CASES).lower()


def test_fts5_benchmark_decision_is_configuration_driven(tmp_path: Path) -> None:
    result = run_fts5_benchmark(tmp_path / "benchmark.db")
    thresholds = RetrievalBenchmarkThresholds.from_settings(
        replace(
            load_settings(),
            retrieval_benchmark_min_precision_at_k=1.0,
            retrieval_benchmark_min_recall_at_k=1.0,
            retrieval_benchmark_max_mean_latency_ms=10_000.0,
            retrieval_benchmark_max_peak_memory_bytes=100_000_000,
        )
    )

    accepted = assess_fts5_benchmark(result, thresholds)
    assert accepted.dense_retrieval_justified is False
    assert accepted.failed_thresholds == ()

    rejected = assess_fts5_benchmark(
        result,
        replace(thresholds, min_precision_at_k=1.1),
    )
    assert rejected.dense_retrieval_justified is True
    assert rejected.failed_thresholds == ("precision_at_k",)
