"""Read-only full DEAM inventory and primary population binding."""

from __future__ import annotations

import csv
import hashlib
import re
from collections import defaultdict
from pathlib import Path

from sc_mv_dmer.data.discovery import DEAM_DATASET_VERSION
from sc_mv_dmer.data.identity import make_dataset_id, make_sample_id, make_song_id
from sc_mv_dmer.data.manifests import (
    DatasetManifest,
    DatasetRecord,
    SourceArtifact,
    SourceInventory,
)
from sc_mv_dmer.foundation.canonical import canonical_json, sha256_canonical


class PopulationMismatchError(ValueError):
    """Raised when DEAM source/annotation membership is not the frozen population."""


class SourceInventoryError(ValueError):
    """Raised when source inventory structure is ambiguous or unsafe."""


_AUDIO_RELATIVE = Path("DEAM_audio/MEMD_audio")
_ANNOTATION_RELATIVE = Path("DEAM_Annotations/annotations")
_STATIC_ANNOTATION_PARTS = ("annotations averaged per song", "song_level")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _song_key_from_filename(path: Path) -> str | None:
    return path.stem if re.fullmatch(r"[0-9]+", path.stem) else None


def _annotation_song_keys(root: Path) -> tuple[str, ...]:
    static_dir = root / _ANNOTATION_RELATIVE / "annotations averaged per song" / "song_level"
    files = sorted(static_dir.glob("*.csv"))
    if len(files) != 2:
        raise SourceInventoryError("expected exactly two averaged song-level annotation files")
    keys: set[str] = set()
    for path in files:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source, skipinitialspace=True)
            if not reader.fieldnames or "song_id" not in {name.strip() for name in reader.fieldnames}:
                raise SourceInventoryError(f"annotation file lacks song_id column: {path.name}")
            for row in reader:
                raw = row.get("song_id") or row.get(" song_id")
                if raw is None or not re.fullmatch(r"[0-9]+", raw.strip()):
                    raise SourceInventoryError(f"invalid song_id in annotation file: {path.name}")
                keys.add(raw.strip())
    return tuple(sorted(keys, key=int))


def discover_deam(data_root: Path) -> SourceInventory:
    """Hash all audio and annotation files without decoding or mutating them."""

    root = Path(data_root).resolve()
    audio_dir = root / _AUDIO_RELATIVE
    annotation_dir = root / _ANNOTATION_RELATIVE
    if not audio_dir.is_dir() or not annotation_dir.is_dir():
        raise SourceInventoryError("DEAM audio/annotation directories are missing")
    audio_files = sorted(audio_dir.glob("*.mp3"), key=lambda path: int(path.stem) if path.stem.isdigit() else 10**9)
    if any(_song_key_from_filename(path) is None for path in audio_files):
        raise SourceInventoryError("all DEAM audio filenames must be numeric song keys")
    if len(audio_files) != 1802:
        raise PopulationMismatchError(f"expected 1802 DEAM MP3 files, found {len(audio_files)}")
    audio_keys = tuple(path.stem for path in audio_files)
    if len(set(audio_keys)) != len(audio_keys):
        raise SourceInventoryError("duplicate DEAM audio logical song key")
    annotation_keys = _annotation_song_keys(root)
    if set(audio_keys) != set(annotation_keys):
        raise PopulationMismatchError("audio and averaged annotation song IDs do not match")

    artifacts: list[SourceArtifact] = []
    by_hash: dict[str, list[str]] = defaultdict(list)
    for path, role, base in [
        *[(item, "AUDIO", root) for item in audio_files],
        *[(item, "ANNOTATION", root) for item in sorted(annotation_dir.rglob("*")) if item.is_file()],
    ]:
        relative = path.relative_to(base).as_posix()
        digest = _sha256(path)
        by_hash[digest].append(relative)
        logical_key = _song_key_from_filename(path) if role == "AUDIO" else None
        artifacts.append(
            SourceArtifact(
                relative_path=relative,
                role=role,
                sha256=digest,
                size_bytes=path.stat().st_size,
                logical_song_key=logical_key,
            )
        )
    groups = {digest: f"duplicate-content-{index:04d}" for index, (digest, paths) in enumerate(sorted(by_hash.items())) if len(paths) > 1}
    artifacts = [item.model_copy(update={"duplicate_content_group": groups.get(item.sha256)}) for item in artifacts]
    primary = tuple(key for key in audio_keys if int(key) <= 2000)
    long = tuple(key for key in audio_keys if int(key) >= 2001)
    if len(primary) != 1744 or len(long) != 58:
        raise PopulationMismatchError("DEAM primary/long-song partition is not 1744/58")
    payload = {
        "schema_version": "1.0",
        "inventory_id": "DEAM-SOURCE-INVENTORY-v1",
        "dataset_name": "DEAM",
        "dataset_version": DEAM_DATASET_VERSION,
        "audio_root_relative": "DEAM_audio/MEMD_audio",
        "annotation_root_relative": "DEAM_Annotations/annotations",
        "source_artifacts": [item.model_dump(mode="json") for item in artifacts],
        "audio_song_keys": list(audio_keys),
        "annotation_song_keys": list(annotation_keys),
        "primary_song_keys": list(primary),
        "long_song_keys": list(long),
        "primary_population_count": 1744,
        "long_song_count": 58,
    }
    return SourceInventory(**payload, inventory_sha256=sha256_canonical(payload))


def bind_deam_primary(inventory: SourceInventory, *, inventory_logical_path: str) -> DatasetManifest:
    """Bind 1,744 primary records; long-song records remain auxiliary only."""

    if len(inventory.primary_song_keys) != 1744 or len(inventory.long_song_keys) != 58:
        raise PopulationMismatchError("inventory does not contain the exact 1744/58 population")
    dataset_id = make_dataset_id()
    audio = {item.logical_song_key: item for item in inventory.source_artifacts if item.role == "AUDIO"}
    records = tuple(
        DatasetRecord(
            dataset_id=dataset_id,
            song_id=make_song_id(dataset_id, key),
            sample_id=make_sample_id(make_song_id(dataset_id, key)),
            logical_song_key=key,
            source_relative_path=audio[key].relative_path,
            source_sha256=audio[key].sha256,
            source_size_bytes=audio[key].size_bytes,
            population_role="PRIMARY",
        )
        for key in inventory.primary_song_keys
    )
    payload = {
        "schema_version": "1.0",
        "manifest_id": "DEAM-PRIMARY-v1",
        "dataset_name": "DEAM",
        "dataset_version": inventory.dataset_version,
        "dataset_id": dataset_id,
        "inventory_logical_path": inventory_logical_path,
        "inventory_sha256": inventory.inventory_sha256,
        "primary_records": [item.model_dump(mode="json") for item in records],
        "primary_song_keys": list(inventory.primary_song_keys),
        "long_song_keys": list(inventory.long_song_keys),
        "primary_population_count": 1744,
        "long_song_count": 58,
    }
    return DatasetManifest(**payload, manifest_sha256=sha256_canonical(payload))


def write_json_once(value: object, destination: Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as output:
        output.write(canonical_json(value) + "\n")
