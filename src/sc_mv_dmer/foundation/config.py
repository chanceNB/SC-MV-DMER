"""Layered research configuration resolution with dual deterministic hashes."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from sc_mv_dmer.foundation.canonical import canonical_json, sha256_canonical


class FormalOverrideError(ValueError):
    """Raised when a formal run attempts a non-runtime CLI override."""


class UnknownConfigKeyError(ValueError):
    """Raised when a configuration layer or override introduces an unknown key."""


class ConfigInvariantError(ValueError):
    """Raised when required configuration metadata is absent or inconsistent."""


class RunMode(str, Enum):
    FORMAL = "formal"
    DEBUG = "debug"


@dataclass(frozen=True)
class ResolvedConfigSnapshot:
    resolved_config: dict[str, Any]
    canonical_json: str
    semantic_config_hash: str
    resolved_config_hash: str
    config_schema_version: str
    research_spec_version: str
    source_provenance: tuple[dict[str, object], ...]
    cli_overrides: dict[str, object]
    semantic_diff: dict[str, object]
    run_mode: RunMode
    paper_eligible: bool


FORMAL_RUNTIME_OVERRIDE_ALLOWLIST = frozenset(
    {
        "runtime.device",
        "runtime.batch_size",
        "runtime.gradient_accumulation",
        "runtime.num_workers",
        "runtime.precision",
        "runtime.checkpointing",
    }
)
_SEMANTIC_EXCLUDED_KEYS = frozenset(
    {
        "runtime",
        "data_root",
        "local_path",
        "machine",
        "gpu",
        "gpu_type",
        "output",
        "output_dir",
        "timestamp",
        "paths",
    }
)


def _repository_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    raise ConfigInvariantError("configuration source is not inside a repository")


def _logical_path(path: Path, repository_root: Path) -> str:
    try:
        return path.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError as exc:
        raise ConfigInvariantError(
            f"configuration source must be repository-relative: {path}"
        ) from exc


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigInvariantError(f"unable to read configuration source: {path}") from exc
    if not isinstance(loaded, dict):
        raise ConfigInvariantError(f"configuration source must contain an object: {path}")
    if not all(isinstance(key, str) for key in loaded):
        raise ConfigInvariantError(f"configuration source keys must be strings: {path}")
    return loaded


def _validate_known_keys(value: Mapping[str, Any], shape: Mapping[str, Any], prefix: str = "") -> None:
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else key
        if key not in shape:
            raise UnknownConfigKeyError(f"unknown configuration key: {path}")
        shape_value = shape[key]
        if isinstance(item, Mapping) and isinstance(shape_value, Mapping):
            _validate_known_keys(item, shape_value, path)


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _apply_override(config: dict[str, Any], dotted_path: str, value: object) -> None:
    parts = dotted_path.split(".")
    if not dotted_path or any(not part for part in parts):
        raise UnknownConfigKeyError(f"unknown configuration key: {dotted_path}")
    cursor: dict[str, Any] = config
    for part in parts[:-1]:
        next_value = cursor.get(part)
        if not isinstance(next_value, dict):
            raise UnknownConfigKeyError(f"unknown configuration key: {dotted_path}")
        cursor = next_value
    if parts[-1] not in cursor:
        raise UnknownConfigKeyError(f"unknown configuration key: {dotted_path}")
    cursor[parts[-1]] = copy.deepcopy(value)


def _semantic_projection(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _semantic_projection(item)
            for key, item in value.items()
            if key not in _SEMANTIC_EXCLUDED_KEYS
        }
    if isinstance(value, list):
        return [_semantic_projection(item) for item in value]
    return value


def _semantic_diff(baseline: Any, resolved: Any, prefix: str = "") -> dict[str, object]:
    if isinstance(baseline, Mapping) and isinstance(resolved, Mapping):
        differences: dict[str, object] = {}
        for key in sorted(set(baseline) | set(resolved)):
            child_prefix = f"{prefix}.{key}" if prefix else key
            if key not in baseline or key not in resolved:
                differences[child_prefix] = {
                    "baseline": baseline.get(key),
                    "resolved": resolved.get(key),
                }
            else:
                differences.update(_semantic_diff(baseline[key], resolved[key], child_prefix))
        return differences
    if baseline != resolved:
        return {prefix: {"baseline": baseline, "resolved": resolved}}
    return {}


def _validate_invariants(config: Mapping[str, Any]) -> tuple[str, str]:
    schema_version = config.get("config_schema_version")
    research_version = config.get("research_spec_version")
    if not isinstance(schema_version, str) or not schema_version:
        raise ConfigInvariantError("config_schema_version must be a non-empty string")
    if not isinstance(research_version, str) or not research_version:
        raise ConfigInvariantError("research_spec_version must be a non-empty string")
    return schema_version, research_version


def resolve_config(
    paths: Sequence[Path], overrides: Mapping[str, object], run_mode: RunMode
) -> ResolvedConfigSnapshot:
    """Resolve ordered YAML layers and return their reproducible configuration snapshot."""

    if not paths:
        raise ConfigInvariantError("at least one configuration layer is required")
    if not isinstance(run_mode, RunMode):
        raise ConfigInvariantError("run_mode must be a RunMode")

    source_paths = [Path(path).resolve() for path in paths]
    repository_root = _repository_root(source_paths[0].parent)
    layers = [_read_yaml(path) for path in source_paths]
    shape = layers[0]
    resolved: dict[str, Any] = {}
    provenance: list[dict[str, object]] = []
    for merge_order, (path, layer) in enumerate(zip(source_paths, layers)):
        _validate_known_keys(layer, shape)
        resolved = _deep_merge(resolved, layer)
        provenance.append(
            {
                "logical_path": _logical_path(path, repository_root),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "merge_order": merge_order,
            }
        )

    for path, value in overrides.items():
        if run_mode is RunMode.FORMAL and path not in FORMAL_RUNTIME_OVERRIDE_ALLOWLIST:
            raise FormalOverrideError(f"formal CLI override is not allowlisted: {path}")
        _apply_override(resolved, path, value)

    schema_version, research_version = _validate_invariants(resolved)
    canonical = canonical_json(resolved)
    semantic = _semantic_projection(resolved)
    baseline_semantic = _semantic_projection(layers[0])
    return ResolvedConfigSnapshot(
        resolved_config=resolved,
        canonical_json=canonical,
        semantic_config_hash=sha256_canonical(semantic),
        resolved_config_hash=sha256_canonical(resolved),
        config_schema_version=schema_version,
        research_spec_version=research_version,
        source_provenance=tuple(provenance),
        cli_overrides=copy.deepcopy(dict(overrides)),
        semantic_diff=_semantic_diff(baseline_semantic, semantic),
        run_mode=run_mode,
        paper_eligible=run_mode is RunMode.FORMAL,
    )
