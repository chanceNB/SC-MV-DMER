"""Step 2 full-population immutable cache materialization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from sc_mv_dmer.data.manifests import DatasetManifest, verify_dataset_manifest_checksum
from sc_mv_dmer.data.splits import load_frozen_split, verify_frozen_split
from sc_mv_dmer.features.binning import bin_feature
from sc_mv_dmer.features.cache import CacheSpec, FourViewFeatureCacheManifest, write_cache
from sc_mv_dmer.features.deep import extract_deep
from sc_mv_dmer.features.handcrafted import extract_handcrafted
from sc_mv_dmer.features.mert_upstream import PINNED_MERT_REVISION
from sc_mv_dmer.features.pseudo_labels import derive_pseudo_labels
from sc_mv_dmer.features.resample import load_waveform
from sc_mv_dmer.foundation.canonical import canonical_json, sha256_canonical


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_pinned_mert(model_root: Path):
    import torch
    from transformers import AutoModel, Wav2Vec2FeatureExtractor

    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_root, local_files_only=True)
    model = AutoModel.from_pretrained(model_root, trust_remote_code=True, local_files_only=True)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    return processor, model


def _deep_forward(waveform: np.ndarray, processor: Any, model: Any) -> dict[str, list[np.ndarray]]:
    import torch

    processed = processor(waveform, sampling_rate=24000, return_tensors="pt", padding=False)
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in processed.items() if key in {"input_values", "attention_mask"}}
    captured: dict[int, Any] = {}
    hooks = []
    if hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
        for block_number, layer_index in ((5, 4), (6, 5)):
            hooks.append(model.encoder.layers[layer_index].register_forward_hook(lambda _m, _a, out, n=block_number: captured.__setitem__(n, out[0] if isinstance(out, tuple) else out)))
    try:
        with torch.inference_mode():
            output = model(**inputs, output_hidden_states=True, output_attentions=False, return_dict=True)
    finally:
        for hook in hooks:
            hook.remove()
    states = output.hidden_states
    layer5 = captured.get(5, states[5]).detach().cpu().numpy()
    layer6 = captured.get(6, states[6]).detach().cpu().numpy()
    return {"hidden_states": [np.asarray(layer5), np.asarray(layer6)]}


def materialize_population(*, repo_root: Path, data_root: Path, dataset_manifest_path: Path, split_manifest_path: Path, upstream_manifest_path: Path, dimensions_path: Path, timegrid_path: Path, rg01_path: Path, mert_model_root: Path, cache_manifest_path: Path, pseudo_manifest_path: Path) -> tuple[FourViewFeatureCacheManifest, dict[str, Any]]:
    dataset_payload = _json(dataset_manifest_path)
    dataset = DatasetManifest.model_validate(dataset_payload)
    verify_dataset_manifest_checksum(dataset)
    split = load_frozen_split(split_manifest_path)
    verify_frozen_split(dataset, split, dataset_manifest_file_sha256=_sha256_file(dataset_manifest_path))
    upstream = _json(upstream_manifest_path)
    if upstream.get("revision") != PINNED_MERT_REVISION:
        raise ValueError("upstream MERT manifest revision is not pinned")
    dimensions = _json(dimensions_path)
    if dimensions.get("observed_t_raw") != 3374 or dimensions.get("canonical_downstream_t") != 90:
        raise ValueError("dimensions manifest is not E1.2 v2 3374->90")
    timegrid = _json(timegrid_path)
    if timegrid.get("bin_count") != 90 or timegrid.get("raw_frame_count") != 3374:
        raise ValueError("timegrid manifest is not E1.3 3374->90")
    rg01 = _json(rg01_path)
    if rg01.get("verdict") != "PASS":
        raise ValueError("RG-01 is not PASS/current-effective")
    timegrid_hash = _sha256_file(timegrid_path)
    dimension_hash = _sha256_file(dimensions_path)
    processor, model = _load_pinned_mert(mert_model_root)
    records: list[dict[str, Any]] = []
    pseudo_records: list[dict[str, Any]] = []
    for index, source in enumerate(dataset.primary_records, start=1):
        audio_path = data_root / Path(source.source_relative_path)
        wave22 = load_waveform(audio_path, song_id=source.song_id, sample_id=source.sample_id, source_relative_path=source.source_relative_path, target_sample_rate_hz=22050, source_audio_sha256=source.source_sha256)
        wave24 = load_waveform(audio_path, song_id=source.song_id, sample_id=source.sample_id, source_relative_path=source.source_relative_path, target_sample_rate_hz=24000, source_audio_sha256=source.source_sha256)
        hand = extract_handcrafted(wave22)
        binned = {view: bin_feature(record.model_copy(update={"dataset_id": dataset.dataset_id}), timegrid_binding_sha256=timegrid_hash).model_copy(update={"dimension_binding_sha256": dimension_hash}) for view, record in hand.items()}
        deep = extract_deep(wave24, _deep_forward(wave24.waveform, processor, model), timegrid_hash).model_copy(update={"dataset_id": dataset.dataset_id, "dimension_binding_sha256": dimension_hash})
        records.append({"dataset_id": dataset.dataset_id, "song_id": source.song_id, "sample_id": source.sample_id, "source_audio_sha256": source.source_sha256, "views": {**binned, "deep": deep}})
        labels = derive_pseudo_labels(wave22, timegrid_binding_sha256=timegrid_hash).model_copy(update={"dataset_id": dataset.dataset_id})
        pseudo_records.append(labels.model_dump(mode="json", exclude={"targets", "valid_target_mask"}) | {"targets": {name: {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()} for name, value in labels.targets.items()}, "valid_target_mask_sha256": hashlib.sha256(labels.valid_target_mask.tobytes()).hexdigest()})
        if index % 25 == 0:
            print(f"materialized {index}/1744", flush=True)
    spec = CacheSpec(cache_id="deam-four-view", version="v1", logical_root="artifacts/features/deam-four-view-v1", dataset_manifest_logical_path="manifests/datasets/deam-primary-v1.json", dataset_manifest_sha256=_sha256_file(dataset_manifest_path), split_manifest_logical_path="manifests/splits/deam-primary-song-80-10-10-v1.json", split_manifest_sha256=_sha256_file(split_manifest_path), dimensions_logical_path="manifests/dimensions/mert-downstream-dimensions-v2.json", dimensions_sha256=dimension_hash, timegrid_logical_path="manifests/timegrid/mert-2hz-timegrid-v2.json", timegrid_sha256=timegrid_hash, upstream_manifest_logical_path="manifests/upstream/mert-v1-95m-primary-v1.json", upstream_manifest_sha256=_sha256_file(upstream_manifest_path), rg01_evidence_logical_path="evidence/gates/rg-01/rg-01-attempt-v1.json", rg01_evidence_sha256=_sha256_file(rg01_path))
    cache = write_cache(records, spec, cache_manifest_path)
    pseudo_unsigned = {"schema_version": "1.0", "manifest_id": "DEAM-PSEUDO-LABEL-BUNDLE-v1", "rule_version": "deam-pseudo-label-v1", "dataset_manifest_logical_path": "manifests/datasets/deam-primary-v1.json", "dataset_manifest_sha256": _sha256_file(dataset_manifest_path), "split_manifest_logical_path": "manifests/splits/deam-primary-song-80-10-10-v1.json", "split_manifest_sha256": _sha256_file(split_manifest_path), "timegrid_logical_path": "manifests/timegrid/mert-2hz-timegrid-v2.json", "timegrid_sha256": timegrid_hash, "record_count": len(pseudo_records), "targets": ["rms", "brightness", "mode", "key"], "records": pseudo_records}
    pseudo_manifest = pseudo_unsigned | {"manifest_sha256": sha256_canonical(pseudo_unsigned)}
    pseudo_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pseudo_manifest_path.write_text(canonical_json(pseudo_manifest) + "\n", encoding="utf-8")
    return cache, pseudo_manifest
