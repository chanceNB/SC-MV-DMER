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

__all__ = [
    "ConfigInvariantError",
    "FormalOverrideError",
    "ResolvedConfigSnapshot",
    "RunMode",
    "UnknownConfigKeyError",
    "canonical_json",
    "resolve_config",
    "sha256_canonical",
    "stable_id",
]
