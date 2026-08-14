"""Reproducible local FTS5 retrieval benchmark used before any dense-retrieval pilot."""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path

from local_ai_agent.config import Settings, load_settings
from local_ai_agent.memory.repository import MemoryRepository
from local_ai_agent.schemas.contracts import ConfidenceLevel, MemoryCategory, MemoryRecord


@dataclass(frozen=True, slots=True)
class BenchmarkMemory:
    """A sanitized, public-style fact used to exercise the real local FTS5 path."""

    key: str
    value: str


@dataclass(frozen=True, slots=True)
class RetrievalBenchmarkCase:
    """One deterministic query and the memory keys that must be retrieved."""

    query: str
    relevant_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RetrievalBenchmarkThresholds:
    """Versioned acceptance limits that gate any dense-retrieval pilot."""

    top_k: int
    min_precision_at_k: float
    min_recall_at_k: float
    max_mean_latency_ms: float
    max_peak_memory_bytes: int

    @classmethod
    def from_settings(cls, settings: Settings) -> RetrievalBenchmarkThresholds:
        return cls(
            top_k=settings.retrieval_benchmark_top_k,
            min_precision_at_k=settings.retrieval_benchmark_min_precision_at_k,
            min_recall_at_k=settings.retrieval_benchmark_min_recall_at_k,
            max_mean_latency_ms=settings.retrieval_benchmark_max_mean_latency_ms,
            max_peak_memory_bytes=settings.retrieval_benchmark_max_peak_memory_bytes,
        )


@dataclass(frozen=True, slots=True)
class RetrievalCaseResult:
    query: str
    retrieved_keys: tuple[str, ...]
    precision_at_k: float
    recall_at_k: float
    latency_ms: float


@dataclass(frozen=True, slots=True)
class RetrievalBenchmarkResult:
    case_count: int
    top_k: int
    precision_at_k: float
    recall_at_k: float
    mean_latency_ms: float
    peak_memory_bytes: int
    cases: tuple[RetrievalCaseResult, ...]


@dataclass(frozen=True, slots=True)
class RetrievalBenchmarkAssessment:
    dense_retrieval_justified: bool
    failed_thresholds: tuple[str, ...]


FTS5_BENCHMARK_MEMORIES: tuple[BenchmarkMemory, ...] = (
    BenchmarkMemory(
        key="delegation-policy",
        value="delegationpolicy fixes specialist tool, model-turn, unit-count, and retry caps.",
    ),
    BenchmarkMemory(
        key="lease-recovery",
        value="leaserecovery fails expired worker claims without re-executing an action.",
    ),
    BenchmarkMemory(
        key="fts-baseline",
        value="ftsbalance measures lexical retrieval quality before a dense retrieval pilot.",
    ),
    BenchmarkMemory(
        key="sandbox-isolation",
        value="sandboxisolation disables network access and uses a read-only root filesystem.",
    ),
)

FTS5_BENCHMARK_CASES: tuple[RetrievalBenchmarkCase, ...] = (
    RetrievalBenchmarkCase(query="delegationpolicy", relevant_keys=("delegation-policy",)),
    RetrievalBenchmarkCase(query="leaserecovery", relevant_keys=("lease-recovery",)),
    RetrievalBenchmarkCase(query="ftsbalance", relevant_keys=("fts-baseline",)),
    RetrievalBenchmarkCase(query="sandboxisolation", relevant_keys=("sandbox-isolation",)),
)


def run_fts5_benchmark(
    database_path: Path,
    *,
    top_k: int | None = None,
) -> RetrievalBenchmarkResult:
    """Measure deterministic FTS5 retrieval quality and local resource use on sanitized records."""
    effective_top_k = top_k if top_k is not None else load_settings().retrieval_benchmark_top_k
    if effective_top_k < 1:
        raise ValueError("Benchmark top_k must be at least one.")
    repository = MemoryRepository(database_path)
    repository.initialize()
    for memory in FTS5_BENCHMARK_MEMORIES:
        repository.upsert(
            MemoryRecord(
                category=MemoryCategory.SEMANTIC,
                key=memory.key,
                value=memory.value,
                confidence=ConfidenceLevel.CONFIRMED,
            )
        )

    if tracemalloc.is_tracing():
        raise RuntimeError("Retrieval benchmark requires exclusive tracemalloc ownership.")
    tracemalloc.start()
    try:
        case_results: list[RetrievalCaseResult] = []
        for case in FTS5_BENCHMARK_CASES:
            started = time.perf_counter_ns()
            records = repository.search(case.query, limit=effective_top_k)
            latency_ms = (time.perf_counter_ns() - started) / 1_000_000
            retrieved_keys = tuple(record.key for record in records)
            relevant_keys = set(case.relevant_keys)
            relevant_results = sum(key in relevant_keys for key in retrieved_keys)
            precision = relevant_results / len(retrieved_keys) if retrieved_keys else 0.0
            recall = relevant_results / len(relevant_keys)
            case_results.append(
                RetrievalCaseResult(
                    query=case.query,
                    retrieved_keys=retrieved_keys,
                    precision_at_k=precision,
                    recall_at_k=recall,
                    latency_ms=latency_ms,
                )
            )
        _, peak_memory_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    return RetrievalBenchmarkResult(
        case_count=len(case_results),
        top_k=effective_top_k,
        precision_at_k=sum(case.precision_at_k for case in case_results) / len(case_results),
        recall_at_k=sum(case.recall_at_k for case in case_results) / len(case_results),
        mean_latency_ms=sum(case.latency_ms for case in case_results) / len(case_results),
        peak_memory_bytes=peak_memory_bytes,
        cases=tuple(case_results),
    )


def assess_fts5_benchmark(
    result: RetrievalBenchmarkResult,
    thresholds: RetrievalBenchmarkThresholds,
) -> RetrievalBenchmarkAssessment:
    """Require measured FTS5 failure before a local dense-retrieval pilot may be proposed."""
    failed_thresholds: list[str] = []
    if result.precision_at_k < thresholds.min_precision_at_k:
        failed_thresholds.append("precision_at_k")
    if result.recall_at_k < thresholds.min_recall_at_k:
        failed_thresholds.append("recall_at_k")
    if result.mean_latency_ms > thresholds.max_mean_latency_ms:
        failed_thresholds.append("mean_latency_ms")
    if result.peak_memory_bytes > thresholds.max_peak_memory_bytes:
        failed_thresholds.append("peak_memory_bytes")
    return RetrievalBenchmarkAssessment(
        dense_retrieval_justified=bool(failed_thresholds),
        failed_thresholds=tuple(failed_thresholds),
    )
