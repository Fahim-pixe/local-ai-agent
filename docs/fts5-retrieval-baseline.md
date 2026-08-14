# Local FTS5 Retrieval Baseline

## Purpose

This report records the first reproducible result from the local FTS5 retrieval benchmark. The benchmark exercises the runtime’s real `MemoryRepository` and SQLite FTS5 path using a fixed sanitized corpus. It measures retrieval quality, local query latency, and transient Python allocation before any dense-retrieval infrastructure is considered.

> **Decision:** The measured baseline satisfies every configured acceptance threshold. A dense-retrieval pilot is **not justified** at this time.

## Measured result

The benchmark was run with:

```bash
PYTHONPATH=src python3 scripts/run_retrieval_benchmark.py
```

| Metric | Configured threshold | Observed baseline | Status |
| --- | ---: | ---: | --- |
| Cases | N/A | 4 | Recorded |
| Retrieval depth | 3 | 3 | Pass |
| Precision@k | ≥ 0.90 | 1.00 | Pass |
| Recall@k | ≥ 0.90 | 1.00 | Pass |
| Mean search latency | ≤ 250.0 ms | 0.899 ms | Pass |
| Transient peak memory | ≤ 10,000,000 bytes | 14,292 bytes | Pass |

The output was:

```json
{"case_count": 4, "dense_retrieval_justified": false, "failed_thresholds": [], "mean_latency_ms": 0.8985945, "peak_memory_bytes": 14292, "precision_at_k": 1.0, "recall_at_k": 1.0, "top_k": 3}
```

## Scope and limitations

The corpus deliberately contains only sanitized, non-user operational facts. The result establishes a deterministic **functional baseline** for the current FTS5 implementation; it does not claim production retrieval quality across a large or domain-specific corpus. Future retrieval changes must add representative sanitized benchmark cases and compare the same quality, latency, and memory metrics before altering the runtime’s local-first storage or authorization boundaries.

## Dense-retrieval entry criterion

A local dense-retrieval pilot may be proposed only when the benchmark run fails at least one versioned `RETRIEVAL_BENCHMARK_*` threshold. The current result has no failures, so the correct next action is to broaden the safe benchmark corpus and establish target-hardware baselines—not to add embeddings, vector storage, fusion, or reranking prematurely.
