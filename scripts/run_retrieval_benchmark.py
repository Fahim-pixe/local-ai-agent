"""Run the versioned local FTS5 retrieval benchmark and print a safe JSON report."""

from __future__ import annotations

import json
from pathlib import Path

from local_ai_agent.config import ensure_workspace, load_settings
from local_ai_agent.evaluation.retrieval_benchmark import (
    RetrievalBenchmarkThresholds,
    assess_fts5_benchmark,
    run_fts5_benchmark,
)


def main() -> None:
    settings = load_settings()
    ensure_workspace(settings)
    database_path = settings.workspace_internal_path / "retrieval_benchmark.db"
    thresholds = RetrievalBenchmarkThresholds.from_settings(settings)
    result = run_fts5_benchmark(database_path, top_k=thresholds.top_k)
    assessment = assess_fts5_benchmark(result, thresholds)
    print(
        json.dumps(
            {
                "case_count": result.case_count,
                "top_k": result.top_k,
                "precision_at_k": result.precision_at_k,
                "recall_at_k": result.recall_at_k,
                "mean_latency_ms": result.mean_latency_ms,
                "peak_memory_bytes": result.peak_memory_bytes,
                "dense_retrieval_justified": assessment.dense_retrieval_justified,
                "failed_thresholds": assessment.failed_thresholds,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
