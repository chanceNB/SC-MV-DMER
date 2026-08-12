"""Execute the strictly scoped E1.1 MERT frame-rate observation once."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from sc_mv_dmer.data.probes import DeamProbeManifest
from sc_mv_dmer.features.mert_e11_execution import (
    map_block_outputs_to_returned_hidden_states,
)
from sc_mv_dmer.features.mert_frame_rate import (
    E11_INPUT_SAMPLE_COUNT,
    E11_INPUT_SAMPLE_RATE_HZ,
    E11VerificationError,
    MertProbeObservation,
    build_probe_observation,
    require_structural_consistency,
    resolve_research_layer_mapping,
    validate_pinned_manifest_file,
    validate_preprocessed_contract,
    validate_registered_source,
)
from sc_mv_dmer.features.mert_snapshot import (
    PinnedMertUpstreamManifest,
    verify_pinned_mert_manifest,
)
from sc_mv_dmer.foundation.canonical import canonical_json, sha256_canonical


E11_MANIFEST_SHA256 = "0c9ebf2cdb148dfe392bac987f41c483f3fd0aa9a23e41b6810ed0daf5d20844"
_REPOSITORY_ID = "m-a-p/MERT-v1-95M"
_REVISION = "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5"


class _Tee:
    def __init__(self, stream: Any, file: Any) -> None:
        self._stream = stream
        self._file = file

    def write(self, text: str) -> int:
        self._file.write(text)
        self._file.flush()
        return self._stream.write(text)

    def flush(self) -> None:
        self._file.flush()
        self._stream.flush()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _runtime_git_clean(repo_root: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    if dirty:
        raise E11VerificationError("formal E1.1 execution requires a clean Git workspace")
    return commit, dirty


def prepare_e11_artifact_paths(repo_root: Path) -> dict[str, Path]:
    """Reserve terminal E1.1 names without creating files before preflight."""

    paths = {
        "report": repo_root / "reports/experiments/e1-1-mert-real-frame-rate-report.txt",
        "json": repo_root / "reports/experiments/e1-1-mert-real-frame-rate.json",
        "evidence": repo_root / "evidence/qualifications/e1-1-mert-frame-rate-verification.json",
    }
    if any(path.exists() for path in paths.values()):
        raise E11VerificationError("E1.1 terminal artifact path already exists; create a new run")
    return paths


def _prepare_real_deam_waveform(source: Path) -> tuple[np.ndarray, dict[str, Any]]:
    info = sf.info(source)
    requested_source_frames = 45 * info.samplerate
    with sf.SoundFile(source) as decoder:
        decoded = decoder.read(
            frames=requested_source_frames,
            start=0,
            dtype="float32",
            always_2d=True,
        )
    if decoded.shape[0] != requested_source_frames:
        raise E11VerificationError("decoder cannot provide the registered [0,45) source segment")
    mono = decoded.mean(axis=1, dtype=np.float32)
    divisor = math.gcd(info.samplerate, E11_INPUT_SAMPLE_RATE_HZ)
    waveform = resample_poly(
        mono,
        E11_INPUT_SAMPLE_RATE_HZ // divisor,
        info.samplerate // divisor,
    ).astype(np.float32, copy=False)
    validate_preprocessed_contract(
        sample_rate_hz=E11_INPUT_SAMPLE_RATE_HZ,
        sample_count=int(waveform.shape[0]),
        duration_seconds=waveform.shape[0] / E11_INPUT_SAMPLE_RATE_HZ,
    )
    if not bool(np.isfinite(waveform).all()):
        raise E11VerificationError("production preprocessed waveform contains non-finite values")
    return waveform, {
        "probe_source": "registered DEAM source",
        "decoder_library": "soundfile",
        "decoder_format": info.format,
        "decoder_subtype": info.subtype,
        "source_sample_rate_hz": info.samplerate,
        "source_total_frames": info.frames,
        "source_channels": info.channels,
        "segment_start_frame": 0,
        "segment_stop_frame_exclusive": requested_source_frames,
        "source_tail_excluded_frames": info.frames - requested_source_frames,
        "channel_policy": "mean stereo channels to mono",
        "resampler": "scipy.signal.resample_poly",
        "output_finite": True,
    }


def _prepare_canonical_waveform() -> tuple[np.ndarray, dict[str, Any]]:
    waveform = np.zeros(E11_INPUT_SAMPLE_COUNT, dtype=np.float32)
    return waveform, {
        "probe_source": "canonical deterministic zero waveform",
        "construction": "numpy.zeros(target_sample_count, dtype=float32)",
        "output_finite": True,
    }


def _run_probe(
    *,
    probe_id: str,
    probe_type: str,
    waveform: np.ndarray,
    processor: Any,
    model: Any,
    torch: Any,
) -> MertProbeObservation:
    processed = processor(
        waveform,
        sampling_rate=E11_INPUT_SAMPLE_RATE_HZ,
        return_tensors="pt",
        padding=False,
    )
    inputs = {
        key: value.to("cuda")
        for key, value in processed.items()
        if key in {"input_values", "attention_mask"}
    }
    if tuple(inputs["input_values"].shape) != (1, E11_INPUT_SAMPLE_COUNT):
        raise E11VerificationError("pinned processor changed the exact input tensor shape")

    captured: dict[int, Any] = {}

    def capture(block_number: int):
        def callback(_module: Any, _arguments: Any, output: Any) -> None:
            captured[block_number] = output[0] if isinstance(output, tuple) else output

        return callback

    hooks = [
        model.encoder.layers[4].register_forward_hook(capture(5)),
        model.encoder.layers[5].register_forward_hook(capture(6)),
    ]
    try:
        with torch.inference_mode():
            output = model(
                **inputs,
                output_hidden_states=True,
                output_attentions=False,
                return_dict=True,
            )
    finally:
        for hook in hooks:
            hook.remove()

    returned = tuple(output.hidden_states)
    candidates = map_block_outputs_to_returned_hidden_states(
        returned_hidden_states=returned,
        captured_block_outputs=captured,
    )
    mapping = resolve_research_layer_mapping(
        hidden_states_count=len(returned),
        block_to_hidden_state_candidates=candidates,
    )
    layer5 = returned[mapping.research_layer_5_actual_index]
    layer6 = returned[mapping.research_layer_6_actual_index]
    return build_probe_observation(
        probe_id=probe_id,
        probe_type=probe_type,  # type: ignore[arg-type]
        input_sample_rate=E11_INPUT_SAMPLE_RATE_HZ,
        input_sample_count=E11_INPUT_SAMPLE_COUNT,
        input_tensor_shape=tuple(int(value) for value in inputs["input_values"].shape),
        hidden_states_count=len(returned),
        layer_mapping=mapping,
        layer5_shape=tuple(int(value) for value in layer5.shape),
        layer6_shape=tuple(int(value) for value in layer6.shape),
        layer5_finite=bool(torch.isfinite(layer5).all().item()),
        layer6_finite=bool(torch.isfinite(layer6).all().item()),
    )


def _print_probe(observation: MertProbeObservation) -> None:
    print(f"probe_id={observation.probe_id}")
    print(f"probe_type={observation.probe_type}")
    print(f"input_sample_rate={observation.input_sample_rate}")
    print(f"input_sample_count={observation.input_sample_count}")
    print(f"input_duration_seconds={observation.input_duration_seconds}")
    print(f"input_tensor_shape={observation.input_tensor_shape}")
    print(f"hidden_states_count={observation.hidden_states_count}")
    print(f"research_layer_5_actual_index={observation.research_layer_5_actual_index}")
    print(f"research_layer_5_block_identity={observation.research_layer_5_block_identity}")
    print(f"research_layer_6_actual_index={observation.research_layer_6_actual_index}")
    print(f"research_layer_6_block_identity={observation.research_layer_6_block_identity}")
    print(f"layer5_shape={observation.layer5_shape}")
    print(f"layer6_shape={observation.layer6_shape}")
    print(f"observed_frame_count = T_raw = {observation.observed_frame_count}")
    print(f"hidden_dimension = D = {observation.hidden_dimension}")
    print(
        "observed_fps = T_raw / 45.0 = "
        f"{observation.observed_frame_count} / 45.0 = {observation.observed_fps} Hz"
    )
    print(f"layer5_finite={observation.layer5_finite}")
    print(f"layer6_finite={observation.layer6_finite}")


def _write_json_once(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        output.write(canonical_json(payload) + "\n")
    return _sha256_file(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    repo_root = arguments.repo_root.resolve()
    git_commit, git_dirty = _runtime_git_clean(repo_root)
    artifact_paths = prepare_e11_artifact_paths(repo_root)
    report_path = artifact_paths["report"]
    json_path = artifact_paths["json"]
    evidence_path = artifact_paths["evidence"]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8") as report:
        tee_stdout = _Tee(sys.stdout, report)
        tee_stderr = _Tee(sys.stderr, report)
        with contextlib.redirect_stdout(tee_stdout), contextlib.redirect_stderr(tee_stderr):
            manifest_path = repo_root / "manifests/upstream/mert-v1-95m-primary-v1.json"
            probe_path = repo_root / "manifests/probes/deam-e1-real-45s-v1.json"
            validate_pinned_manifest_file(manifest_path, expected_sha256=E11_MANIFEST_SHA256)
            manifest = PinnedMertUpstreamManifest.model_validate(_load_json(manifest_path))
            if manifest.repository_id != _REPOSITORY_ID or manifest.revision != _REVISION:
                raise E11VerificationError("pinned upstream identity mismatch")
            verify_pinned_mert_manifest(manifest, arguments.model_root)
            probe = DeamProbeManifest.model_validate(_load_json(probe_path))
            source = validate_registered_source(
                data_root=arguments.data_root,
                source_relative_path=probe.source_relative_path,
                expected_sha256=probe.source_sha256,
            )
            real_waveform, real_preprocessing = _prepare_real_deam_waveform(source)
            canonical_waveform, canonical_preprocessing = _prepare_canonical_waveform()

            import torch
            import transformers
            from transformers import AutoModel, Wav2Vec2FeatureExtractor

            processor = Wav2Vec2FeatureExtractor.from_pretrained(
                arguments.model_root,
                local_files_only=True,
            )
            model = AutoModel.from_pretrained(
                arguments.model_root,
                trust_remote_code=True,
                local_files_only=True,
            ).to("cuda")
            model.eval()
            if model.config.hidden_size != 768:
                raise E11VerificationError("loaded hidden dimension differs from frozen MERT config")

            print("=== MERT E1.1 REAL FRAME-RATE VERIFICATION ===")
            print(f"repository_id={manifest.repository_id}")
            print(f"revision={manifest.revision}")
            print(f"snapshot_aggregate_sha256={manifest.snapshot_aggregate_sha256}")
            print(f"git_commit={git_commit}")
            print(f"transformers_version={transformers.__version__}")
            print(f"torch_version={torch.__version__}")
            print(f"device={torch.cuda.get_device_name(0)}")

            p1 = _run_probe(
                probe_id="P1_CANONICAL_EXACT_LENGTH",
                probe_type="P1_CANONICAL",
                waveform=canonical_waveform,
                processor=processor,
                model=model,
                torch=torch,
            )
            print("--- P1 ---")
            print(f"preprocessing={canonical_json(canonical_preprocessing)}")
            _print_probe(p1)
            p2 = _run_probe(
                probe_id=probe.sample_id,
                probe_type="P2_REAL_DEAM",
                waveform=real_waveform,
                processor=processor,
                model=model,
                torch=torch,
            )
            print("--- P2 ---")
            print(f"preprocessing={canonical_json(real_preprocessing)}")
            _print_probe(p2)
            require_structural_consistency(p1, p2)
            print("p1_p2_structural_consistency=PASS")
            report.flush()

    report_sha256 = _sha256_file(report_path)
    report_payload = {
        "schema_version": "1.0",
        "qualification_id": "E1-1-MERT-REAL-FRAME-RATE-v1",
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "repository_id": manifest.repository_id,
        "revision": manifest.revision,
        "manifest_logical_path": "manifests/upstream/mert-v1-95m-primary-v1.json",
        "manifest_file_sha256": E11_MANIFEST_SHA256,
        "snapshot_aggregate_sha256": manifest.snapshot_aggregate_sha256,
        "probe_manifest_logical_path": "manifests/probes/deam-e1-real-45s-v1.json",
        "probe_manifest_sha256": _sha256_file(probe_path),
        "source_relative_path": probe.source_relative_path,
        "source_sha256": probe.source_sha256,
        "p1_preprocessing": canonical_preprocessing,
        "p2_preprocessing": real_preprocessing,
        "p1": p1.model_dump(mode="json"),
        "p2": p2.model_dump(mode="json"),
        "structural_consistency": "PASS",
        "raw_stdout_stderr_logical_path": "reports/experiments/e1-1-mert-real-frame-rate-report.txt",
        "raw_stdout_stderr_sha256": report_sha256,
        "environment": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": torch.cuda.get_device_name(0),
            "cuda": torch.version.cuda,
            "model_mode": "eval",
            "inference_mode": True,
            "optimizer_created": False,
            "backward_executed": False,
            "weight_mutation": False,
            "timesnet_period_derived": False,
            "mapping_2hz_or_90bin_executed": False,
            "downstream_temporal_dimension_hardcoded": False,
        },
    }
    report_payload["report_sha256"] = sha256_canonical(report_payload)
    report_file_sha256 = _write_json_once(json_path, report_payload)
    evidence_payload = {
        "schema_version": "1.0",
        "qualification_id": "E1-1-MERT-REAL-FRAME-RATE-v1",
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "predecessor_qualification": "evidence/qualifications/e1-0-mert-upstream-binding.json",
        "predecessor_qualification_sha256": "16302b5582e5954e7f47623b2d6c54682229861240f569c89bf16b520d6dbe82",
        "report_logical_path": "reports/experiments/e1-1-mert-real-frame-rate.json",
        "report_file_sha256": report_file_sha256,
        "raw_stdout_stderr_logical_path": report_payload["raw_stdout_stderr_logical_path"],
        "raw_stdout_stderr_sha256": report_sha256,
        "snapshot_aggregate_sha256": manifest.snapshot_aggregate_sha256,
        "p1_p2_structural_consistency": "PASS",
        "e1_1_verdict": "PASS",
        "e1_2_readiness": "READY",
        "forbidden_downstream_actions": report_payload["environment"],
    }
    evidence_payload["qualification_sha256"] = sha256_canonical(evidence_payload)
    _write_json_once(evidence_path, evidence_payload)


if __name__ == "__main__":
    main()
