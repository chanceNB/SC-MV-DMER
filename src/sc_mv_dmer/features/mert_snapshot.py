"""Static binding for the exact MERT-v1-95M Transformers snapshot.

This module only reads files and metadata.  It deliberately has no model or
audio loading entry point.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import platform
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator

from sc_mv_dmer.foundation.canonical import canonical_json, sha256_canonical
from sc_mv_dmer.foundation.manifests import FrozenDict, ImmutableRecord, freeze_value


PINNED_MERT_REPOSITORY = "m-a-p/MERT-v1-95M"
PINNED_MERT_REVISION = "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5"
PINNED_MERT_LICENSE = "cc-by-nc-4.0"

_FILE_ROLES = {
    "config.json": "MODEL_CONFIG",
    "preprocessor_config.json": "PREPROCESSOR_CONFIG",
    "configuration_MERT.py": "CUSTOM_CONFIGURATION_CODE",
    "modeling_MERT.py": "CUSTOM_MODELING_CODE",
    "pytorch_model.bin": "CANONICAL_TRANSFORMERS_WEIGHT",
}
_CUSTOM_CODE_FILES = ("configuration_MERT.py", "modeling_MERT.py")
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class SnapshotBindingError(ValueError):
    """Raised when local bytes cannot form the pinned formal snapshot."""


class PinnedSnapshotFile(ImmutableRecord):
    role: str = Field(min_length=1)
    relative_path: str = Field(min_length=1)
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=_SHA256_PATTERN)


class PinnedMertUpstreamManifest(ImmutableRecord):
    schema_version: Literal["1.0"] = "1.0"
    upstream_model_id: Literal["mert-v1-95m-primary-v1"] = "mert-v1-95m-primary-v1"
    repository_id: Literal["m-a-p/MERT-v1-95M"] = PINNED_MERT_REPOSITORY
    revision: Literal[PINNED_MERT_REVISION] = PINNED_MERT_REVISION
    license: Literal["cc-by-nc-4.0"] = PINNED_MERT_LICENSE
    loader_role: Literal["TRANSFORMERS_AUTO_MODEL_LOCAL_PINNED"] = (
        "TRANSFORMERS_AUTO_MODEL_LOCAL_PINNED"
    )
    trust_remote_code_required: Literal[True] = True
    local_files_only_required: Literal[True] = True
    canonical_weight_file: Literal["pytorch_model.bin"] = "pytorch_model.bin"
    other_weight_files: Mapping[str, str]
    files: tuple[PinnedSnapshotFile, ...] = Field(min_length=5, max_length=5)
    config_sha256: str = Field(pattern=_SHA256_PATTERN)
    preprocessor_sha256: str = Field(pattern=_SHA256_PATTERN)
    custom_code_aggregate_sha256: str = Field(pattern=_SHA256_PATTERN)
    weight_sha256: str = Field(pattern=_SHA256_PATTERN)
    snapshot_aggregate_sha256: str = Field(pattern=_SHA256_PATTERN)
    semantic_identity_sha256: str = Field(pattern=_SHA256_PATTERN)
    config_class: Literal["MERTConfig"] = "MERTConfig"
    model_class: Literal["MERTModel"] = "MERTModel"
    processor_class: Literal["Wav2Vec2FeatureExtractor"] = "Wav2Vec2FeatureExtractor"
    model_type: Literal["mert_model"] = "mert_model"
    architectures: tuple[str, ...]
    declared_sample_rate_hz: Literal[24000] = 24000
    snapshot_declared_transformers_version: str
    custom_code_dependencies: Mapping[str, tuple[str, ...]]
    environment_binding: Mapping[str, Any]

    @field_validator("other_weight_files", "custom_code_dependencies", "environment_binding", mode="after")
    @classmethod
    def _freeze_mappings(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        return freeze_value(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SnapshotBindingError(f"invalid JSON file: {path.name}") from exc
    if not isinstance(value, dict):
        raise SnapshotBindingError(f"JSON file must contain an object: {path.name}")
    return value


def _static_imports(path: Path) -> tuple[str, ...]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise SnapshotBindingError(f"invalid custom-code file: {path.name}") from exc
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add("." * node.level + (node.module or ""))
    return tuple(sorted(names))


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def _validate_static_contract(config: Mapping[str, Any], preprocessor: Mapping[str, Any]) -> None:
    auto_map = config.get("auto_map")
    if not isinstance(auto_map, Mapping) or auto_map.get("AutoConfig") != "configuration_MERT.MERTConfig":
        raise SnapshotBindingError("config AutoConfig does not bind MERTConfig")
    if auto_map.get("AutoModel") != "modeling_MERT.MERTModel":
        raise SnapshotBindingError("config AutoModel does not bind MERTModel")
    if config.get("architectures") != ["MERTModel"] or config.get("model_type") != "mert_model":
        raise SnapshotBindingError("config architecture/model_type is not canonical MERT")
    if config.get("sample_rate") != 24000:
        raise SnapshotBindingError("config sample_rate is not 24000")
    if preprocessor.get("feature_extractor_type") != "Wav2Vec2FeatureExtractor":
        raise SnapshotBindingError("preprocessor class is not Wav2Vec2FeatureExtractor")
    if preprocessor.get("sampling_rate") != 24000:
        raise SnapshotBindingError("preprocessor sampling_rate is not 24000")


def _semantic_payload(manifest_payload: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {"semantic_identity_sha256", "environment_binding"}
    return {key: value for key, value in manifest_payload.items() if key not in excluded}


def build_pinned_mert_manifest(
    snapshot_root: Path,
    *,
    revision: str,
    environment_binding: Mapping[str, Any] | None = None,
) -> PinnedMertUpstreamManifest:
    """Bind the five exact local files without importing model code or weights."""

    if revision != PINNED_MERT_REVISION:
        raise SnapshotBindingError("formal binding requires the exact pinned revision")
    root = Path(snapshot_root)
    missing = [name for name in _FILE_ROLES if not (root / name).is_file()]
    if missing:
        raise SnapshotBindingError("missing required snapshot file(s): " + ", ".join(missing))

    config = _read_json_object(root / "config.json")
    preprocessor = _read_json_object(root / "preprocessor_config.json")
    _validate_static_contract(config, preprocessor)

    files = tuple(
        PinnedSnapshotFile(
            role=_FILE_ROLES[name],
            relative_path=name,
            size_bytes=(root / name).stat().st_size,
            sha256=_sha256_file(root / name),
        )
        for name in sorted(_FILE_ROLES)
    )
    by_name = {item.relative_path: item for item in files}
    custom_files = [by_name[name].model_dump(mode="json") for name in _CUSTOM_CODE_FILES]
    snapshot_files = [item.model_dump(mode="json") for item in files]
    dependencies = FrozenDict(
        {name: _static_imports(root / name) for name in _CUSTOM_CODE_FILES}
    )
    environment = dict(
        environment_binding
        or {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "huggingface_hub_version": _package_version("huggingface-hub"),
            "transformers_version": _package_version("transformers"),
            "pytorch_version": _package_version("torch"),
        }
    )
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "upstream_model_id": "mert-v1-95m-primary-v1",
        "repository_id": PINNED_MERT_REPOSITORY,
        "revision": PINNED_MERT_REVISION,
        "license": PINNED_MERT_LICENSE,
        "loader_role": "TRANSFORMERS_AUTO_MODEL_LOCAL_PINNED",
        "trust_remote_code_required": True,
        "local_files_only_required": True,
        "canonical_weight_file": "pytorch_model.bin",
        "other_weight_files": {
            "MERT-v1-95M_fairseq.pt": "NOT_CONSUMED_BY_CANONICAL_TRANSFORMERS_LOADER"
        },
        "files": snapshot_files,
        "config_sha256": by_name["config.json"].sha256,
        "preprocessor_sha256": by_name["preprocessor_config.json"].sha256,
        "custom_code_aggregate_sha256": sha256_canonical(custom_files),
        "weight_sha256": by_name["pytorch_model.bin"].sha256,
        "snapshot_aggregate_sha256": sha256_canonical(
            {
                "repository_id": PINNED_MERT_REPOSITORY,
                "revision": PINNED_MERT_REVISION,
                "files": snapshot_files,
            }
        ),
        "config_class": "MERTConfig",
        "model_class": "MERTModel",
        "processor_class": "Wav2Vec2FeatureExtractor",
        "model_type": "mert_model",
        "architectures": tuple(config["architectures"]),
        "declared_sample_rate_hz": 24000,
        "snapshot_declared_transformers_version": str(
            config.get("transformers_version", "UNDECLARED")
        ),
        "custom_code_dependencies": dependencies,
        "environment_binding": environment,
    }
    payload["semantic_identity_sha256"] = sha256_canonical(_semantic_payload(payload))
    return PinnedMertUpstreamManifest.model_validate(payload)


def verify_pinned_mert_manifest(
    manifest: PinnedMertUpstreamManifest, snapshot_root: Path
) -> None:
    """Re-hash local bytes and reject any departure from the terminal manifest."""

    root = Path(snapshot_root)
    for item in manifest.files:
        path = root / item.relative_path
        if not path.is_file() or path.stat().st_size != item.size_bytes or _sha256_file(path) != item.sha256:
            raise SnapshotBindingError(f"checksum mismatch: {item.relative_path}")
    rebuilt = build_pinned_mert_manifest(
        root,
        revision=manifest.revision,
        environment_binding=manifest.environment_binding,
    )
    if (
        rebuilt.custom_code_aggregate_sha256 != manifest.custom_code_aggregate_sha256
        or rebuilt.snapshot_aggregate_sha256 != manifest.snapshot_aggregate_sha256
        or rebuilt.semantic_identity_sha256 != manifest.semantic_identity_sha256
    ):
        raise SnapshotBindingError("checksum mismatch: aggregate identity")


def write_immutable_manifest(
    manifest: PinnedMertUpstreamManifest, destination: Path
) -> None:
    """Create a canonical terminal manifest using exclusive file creation."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as output:
        output.write(canonical_json(manifest.model_dump(mode="json")) + "\n")
