"""Exact, specified-but-unexecuted DEAM probe contracts."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from sc_mv_dmer.data.discovery import DeamSource
from sc_mv_dmer.foundation.canonical import canonical_json, sha256_canonical
from sc_mv_dmer.foundation.identity import stable_id
from sc_mv_dmer.foundation.manifests import ImmutableRecord


PROBE_SECONDS = 45
TARGET_SAMPLE_RATE = 24_000
TARGET_SAMPLE_COUNT = PROBE_SECONDS * TARGET_SAMPLE_RATE
PROBE_RULE_VERSION = "deam-smallest-numeric-eligible-45s-v1"


class DeamProbeManifest(ImmutableRecord):
    dataset_id: str
    song_id: str
    sample_id: str
    source_relative_path: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_size_bytes: int
    source_duration_seconds: float
    probe_seconds: int = PROBE_SECONDS
    target_sample_rate_hz: int = TARGET_SAMPLE_RATE
    target_sample_count: int = TARGET_SAMPLE_COUNT
    crop_resample_contract: str
    preparation_status: str
    semantic_identity_json: str
    semantic_identity_sha256: str


def build_probe(source: DeamSource) -> DeamProbeManifest:
    if source.duration_seconds < PROBE_SECONDS:
        raise ValueError("source duration is below exact 45 second probe window")
    semantic = {
        "dataset_id": source.dataset_id,
        "song_id": source.song_id,
        "source_relative_path": source.source_relative_path,
        "source_sha256": source.source_sha256,
        "probe_seconds": PROBE_SECONDS,
        "target_sample_rate_hz": TARGET_SAMPLE_RATE,
        "target_sample_count": TARGET_SAMPLE_COUNT,
        "rule_version": PROBE_RULE_VERSION,
    }
    return DeamProbeManifest(
        dataset_id=source.dataset_id,
        song_id=source.song_id,
        sample_id=stable_id("sample", (source.song_id, PROBE_RULE_VERSION, "0", "45")),
        source_relative_path=source.source_relative_path,
        source_sha256=source.source_sha256,
        source_size_bytes=source.source_size_bytes,
        source_duration_seconds=source.duration_seconds,
        crop_resample_contract="crop [0,45)s; resample only if needed to 24000 Hz; not executed",
        preparation_status="SPECIFIED_NOT_EXECUTED",
        semantic_identity_json=canonical_json(semantic),
        semantic_identity_sha256=sha256_canonical(semantic),
    )


def write_probe_manifest(probe: DeamProbeManifest, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as artifact:
        artifact.write(probe.model_dump_json(indent=2) + "\n")
