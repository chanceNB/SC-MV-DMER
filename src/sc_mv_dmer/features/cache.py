"""Content-addressed, immutable feature cache contracts."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from sc_mv_dmer.features.models import BinnedFeatureRecord, ndarray_sha256
from sc_mv_dmer.foundation.canonical import canonical_json, sha256_canonical


class ImmutableCacheError(RuntimeError):
    """Raised when a finalized cache version would be mutated."""


class CacheSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    cache_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    logical_root: str = Field(min_length=1)
    dataset_manifest_logical_path: str = Field(min_length=1)
    dataset_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_manifest_logical_path: str = Field(min_length=1)
    split_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dimensions_logical_path: str = Field(min_length=1)
    dimensions_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    timegrid_logical_path: str = Field(min_length=1)
    timegrid_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    upstream_manifest_logical_path: str = Field(min_length=1)
    upstream_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rg01_evidence_logical_path: str = Field(min_length=1)
    rg01_evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CacheRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str
    song_id: str
    sample_id: str
    source_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    payload_logical_path: str
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    views: dict[str, dict[str, Any]]


class FourViewFeatureCacheManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "1.0"
    cache_id: str
    cache_version: str
    dataset_manifest_logical_path: str
    dataset_manifest_sha256: str
    split_manifest_logical_path: str
    split_manifest_sha256: str
    dimensions_logical_path: str
    dimensions_sha256: str
    timegrid_logical_path: str
    timegrid_sha256: str
    upstream_manifest_logical_path: str
    upstream_manifest_sha256: str
    rg01_evidence_logical_path: str
    rg01_evidence_sha256: str
    record_count: int
    records: tuple[CacheRecord, ...]
    manifest_sha256: str

    def unsigned_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("manifest_sha256", None)
        return payload


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **{k: np.ascontiguousarray(v) for k, v in arrays.items()})
    return _sha256_file(path)


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_cache(
    records: Iterable[Mapping[str, Any]],
    spec: CacheSpec,
    destination: Path,
) -> FourViewFeatureCacheManifest:
    """Write a cache atomically and reject mutation of an existing version.

    Each record mapping must contain identity fields and ``views`` where each
    view is a :class:`BinnedFeatureRecord` or a compatible mapping with values.
    """

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        candidate = dict(existing)
        candidate.pop("manifest_sha256", None)
        raise ImmutableCacheError(
            "finalized cache manifest already exists; content-addressed versions cannot be overwritten"
        )
    staging = Path(tempfile.mkdtemp(prefix=f".{spec.cache_id}-staging-", dir=str(destination.parent)))
    try:
        manifest_records: list[CacheRecord] = []
        for item in records:
            identity = {key: item[key] for key in ("dataset_id", "song_id", "sample_id", "source_audio_sha256")}
            views = item["views"]
            payload_arrays: dict[str, np.ndarray] = {}
            view_meta: dict[str, dict[str, Any]] = {}
            for view_name in ("deep", "mel", "mfcc", "chroma"):
                value = views[view_name]
                if isinstance(value, BinnedFeatureRecord):
                    array = value.values
                    meta = value.model_dump(mode="json", exclude={"values"})
                else:
                    array = np.asarray(value["values"])
                    meta = {k: v for k, v in value.items() if k != "values"}
                if array.ndim != 2 or array.shape[0] != 90 or not np.isfinite(array).all():
                    raise ValueError(f"invalid finalized feature view: {view_name}")
                payload_arrays[view_name] = array
                view_meta[view_name] = {
                    **meta,
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "finite": True,
                    "content_sha256": ndarray_sha256(array),
                }
            filename = f"{identity['sample_id']}.npz"
            payload_path = staging / filename
            payload_hash = _write_npz(payload_path, payload_arrays)
            manifest_records.append(
                CacheRecord(
                    **identity,
                    payload_logical_path=f"{spec.logical_root}/{filename}",
                    payload_sha256=payload_hash,
                    views=view_meta,
                )
            )
        unsigned = {
            "schema_version": "1.0",
            "cache_id": spec.cache_id,
            "cache_version": spec.version,
            "dataset_manifest_logical_path": spec.dataset_manifest_logical_path,
            "dataset_manifest_sha256": spec.dataset_manifest_sha256,
            "split_manifest_logical_path": spec.split_manifest_logical_path,
            "split_manifest_sha256": spec.split_manifest_sha256,
            "dimensions_logical_path": spec.dimensions_logical_path,
            "dimensions_sha256": spec.dimensions_sha256,
            "timegrid_logical_path": spec.timegrid_logical_path,
            "timegrid_sha256": spec.timegrid_sha256,
            "upstream_manifest_logical_path": spec.upstream_manifest_logical_path,
            "upstream_manifest_sha256": spec.upstream_manifest_sha256,
            "rg01_evidence_logical_path": spec.rg01_evidence_logical_path,
            "rg01_evidence_sha256": spec.rg01_evidence_sha256,
            "record_count": len(manifest_records),
            "records": [record.model_dump(mode="json") for record in manifest_records],
        }
        manifest = FourViewFeatureCacheManifest(
            **unsigned, manifest_sha256=sha256_canonical(unsigned)
        )
        temp_manifest = staging / "manifest.json"
        temp_manifest.write_text(canonical_json(manifest.model_dump(mode="json")) + "\n", encoding="utf-8")
        # Move payloads into the ignored cache root, then publish the tracked manifest last.
        # A manifest conventionally lives at ``manifests/features/<name>.json``;
        # resolve the repository/workspace root without ever serializing it.
        root = destination.parents[2] if len(destination.parents) >= 3 else destination.parent
        final_root = root / spec.logical_root
        final_root.mkdir(parents=True, exist_ok=True)
        for payload in staging.glob("*.npz"):
            target = final_root / payload.name
            if target.exists():
                if _sha256_file(target) != _sha256_file(payload):
                    raise ImmutableCacheError(f"immutable payload mutation: {target}")
            else:
                os.replace(payload, target)
        os.replace(temp_manifest, destination)
        return manifest
    finally:
        for child in staging.glob("*"):
            child.unlink(missing_ok=True)
        staging.rmdir()


def verify_cache_manifest(path: Path) -> FourViewFeatureCacheManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    manifest = FourViewFeatureCacheManifest.model_validate(payload)
    if sha256_canonical(manifest.unsigned_payload()) != manifest.manifest_sha256:
        raise ImmutableCacheError("cache manifest checksum mismatch")
    return manifest
