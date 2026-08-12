"""Deterministic configuration and identity primitives for SC-MV-DMER."""

from sc_mv_dmer.foundation.canonical import canonical_json, sha256_canonical
from sc_mv_dmer.foundation.config import (
    ConfigInvariantError,
    FormalOverrideError,
    ResolvedConfigSnapshot,
    RunMode,
    UnknownConfigKeyError,
    resolve_config,
)
from sc_mv_dmer.foundation.identity import stable_id
from sc_mv_dmer.foundation.lifecycle import (
    GateEvaluationAttempt,
    HistoricalVerdict,
    InvalidationRecord,
    RunManifest,
    SupersessionRecord,
    TerminalState,
    finalize_run,
    start_run,
)
from sc_mv_dmer.foundation.manifests import ArtifactRef, RunSpec
from sc_mv_dmer.foundation.validity import Registry, resolve_effective_validity

__all__ = [
    "ConfigInvariantError",
    "FormalOverrideError",
    "ResolvedConfigSnapshot",
    "RunMode",
    "UnknownConfigKeyError",
    "ArtifactRef",
    "GateEvaluationAttempt",
    "HistoricalVerdict",
    "InvalidationRecord",
    "Registry",
    "RunManifest",
    "RunSpec",
    "SupersessionRecord",
    "TerminalState",
    "canonical_json",
    "resolve_config",
    "resolve_effective_validity",
    "sha256_canonical",
    "start_run",
    "stable_id",
    "finalize_run",
]
