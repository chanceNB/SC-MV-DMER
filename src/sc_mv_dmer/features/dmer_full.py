"""Full DEAM feature windows; original audio and frozen MERT layers stay intact.

Frame membership uses local frame centres and half-open 500 ms intervals.
The final partially observed interval is retained. Signal padding is confined
to memory; audio coverage records real samples, independently of labels.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from sc_mv_dmer.data.deam_dmer_full import file_sha256
from sc_mv_dmer.foundation.canonical import canonical_json, sha256_canonical

FEATURE_VERSION = 'deam-dmer-features-v2'
MERT_REVISION = '12af15fef9d0ac838c3f475bfbbf26d2060dd4f5'
DEFAULT_MODEL_ROOT = Path('D:/SC-MV-DMER-data/upstream_models/mert-v1-95m') / MERT_REVISION
DECODED_SUPPORT_POLICY = 'DECODED_PCM_CANONICAL_WINDOWS/v1'
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def pool_to_2hz(values: np.ndarray, times: np.ndarray, *, valid_seconds: float,
                 frame_valid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    values, times = np.asarray(values), np.asarray(times)
    if values.shape[0] != len(times) or values.ndim not in (1, 2):
        raise ValueError('frame values and timestamps do not align')
    if not np.isfinite(values).all() or not np.isfinite(times).all():
        raise ValueError('nonfinite feature frames')
    valid = (times >= 0) & (times < min(valid_seconds, 45.))
    if frame_valid is not None:
        valid &= np.asarray(frame_valid, dtype=bool)
    bin_index = np.floor(times * 2).astype(np.int64)
    result = np.zeros((90,) + values.shape[1:], dtype=np.float32)
    observed = np.zeros(90, dtype=bool)
    for idx in range(90):
        take = valid & (bin_index == idx)
        if take.any():
            result[idx] = values[take].mean(axis=0, dtype=np.float64)
            observed[idx] = True
    return result, observed


def ks_mode_key(chroma: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """KS Pearson profiles; signed best-major/minor margin / 2, tonic 0=C.

    This is a KS diagnostic target, not Essentia EDMA key extraction.
    Constant and silent chroma have undefined correlations and are masked.
    """
    values = np.asarray(chroma, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 12 or not np.isfinite(values).all():
        raise ValueError('chroma must be finite [frames, 12]')
    templates = np.stack([np.roll(profile, key) for profile in (MAJOR, MINOR) for key in range(12)])
    templates -= templates.mean(axis=1, keepdims=True)
    templates /= np.linalg.norm(templates, axis=1, keepdims=True)
    centered = values - values.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(centered, axis=1, keepdims=True)
    valid = (norm[:, 0] > 1e-10) & (values.sum(axis=1) > 1e-10)
    normalized = np.divide(centered, norm, out=np.zeros_like(centered), where=norm > 1e-10)
    corr = np.clip(normalized @ templates.T, -1., 1.)
    mode = (corr[:, :12].max(axis=1) - corr[:, 12:].max(axis=1)) / 2.
    mode[~valid] = 0.
    return mode.astype(np.float32), (corr.argmax(axis=1) % 12).astype(np.int64), valid


def extract_handcrafted_window(wave22: np.ndarray, *, valid_seconds: float) -> dict[str, np.ndarray]:
    import librosa

    if not 0 < valid_seconds <= 45:
        raise ValueError('valid_seconds must be in (0, 45]')
    wave = np.asarray(wave22, dtype=np.float32)
    if wave.ndim != 1 or not np.isfinite(wave).all():
        raise ValueError('waveform must be finite mono')
    padded = np.zeros(45 * 22050, dtype=np.float32)
    # Ignore any samples beyond the declared source coverage.
    n = min(len(wave), int(np.ceil(valid_seconds * 22050)), len(padded))
    padded[:n] = wave[:n]
    spectrum = librosa.stft(padded, n_fft=2048, hop_length=512, center=True, pad_mode='constant')
    power = np.abs(spectrum) ** 2
    mel = librosa.feature.melspectrogram(S=power, sr=22050, n_fft=2048, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=1., amin=1e-10, top_db=None)
    mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=40, dct_type=2, norm='ortho')
    # norm=1 performs the required raw-frame L1 normalization before pooling.
    chroma = librosa.feature.chroma_cqt(y=padded, sr=22050, hop_length=512,
                                      bins_per_octave=36, n_chroma=12, norm=1,
                                      tuning=0., n_octaves=7, threshold=0.)
    time = np.arange(power.shape[1], dtype=np.float64) * 512 / 22050
    chroma_time = np.arange(chroma.shape[1], dtype=np.float64) * 512 / 22050
    coverage = np.clip((valid_seconds - np.arange(90) * .5) / .5, 0., 1.).astype(np.float32)
    arrays = {'audio_coverage': coverage, 'audio_mask': coverage > 0}
    for name, values, times in [('mel', mel_db.T, time), ('mfcc', mfcc.T, time), ('chroma', chroma.T, chroma_time)]:
        arrays[name], _ = pool_to_2hz(values, times, valid_seconds=valid_seconds)
    rms = librosa.feature.rms(y=padded, frame_length=2048, hop_length=512, center=True, pad_mode='constant')[0]
    brightness = librosa.feature.spectral_centroid(S=np.abs(spectrum), sr=22050)[0] / 11025.
    mode, _, tonal = ks_mode_key(chroma.T)
    arrays['sensor_rms'], rms_valid = pool_to_2hz(rms, time, valid_seconds=valid_seconds)
    arrays['sensor_brightness'], brightness_valid = pool_to_2hz(brightness, time, valid_seconds=valid_seconds, frame_valid=rms > 1e-10)
    arrays['sensor_mode'], mode_valid = pool_to_2hz(mode, chroma_time, valid_seconds=valid_seconds, frame_valid=tonal)
    arrays['sensor_valid_mask'] = np.stack([rms_valid, brightness_valid, mode_valid], axis=-1)
    arrays['sensor_key'] = np.zeros(9, dtype=np.int64)
    arrays['sensor_key_mask'] = np.zeros(9, dtype=bool)
    for segment in range(9):
        take = (chroma_time >= segment * 5.) & (chroma_time < min((segment + 1) * 5., valid_seconds)) & tonal
        if take.any():
            _, key, valid = ks_mode_key(chroma[:, take].mean(axis=1)[None])
            arrays['sensor_key'][segment] = key[0]
            arrays['sensor_key_mask'][segment] = valid[0]
    return arrays


def attach_mert_layers(arrays: dict[str, np.ndarray], layers: np.ndarray, *, valid_seconds: float) -> None:
    layers = np.asarray(layers)
    if layers.shape != (2, 3374, 768) or not np.isfinite(layers).all():
        raise ValueError('pinned MERT layers must be finite [2, 3374, 768]')
    times = (399 + 640 * np.arange(3374, dtype=np.float64)) / 48000.
    arrays['mert_layers'] = layers.astype(np.float16)
    arrays['mert_time_seconds'] = times
    arrays['mert_mask'] = (400 + 320 * np.arange(3374)) <= int(np.ceil(valid_seconds * 24000))
    # Convenience baseline only; the full model consumes separate raw layers.
    arrays['deep'], _ = pool_to_2hz(layers.astype(np.float32).mean(axis=0), times, valid_seconds=valid_seconds,
                                  frame_valid=arrays['mert_mask'])


def _atomic_json(path: Path, payload: dict) -> None:
    staging = path.with_suffix(path.suffix + '.tmp')
    staging.write_text(canonical_json(payload) + '\n', encoding='utf-8')
    os.replace(staging, path)


def save_window(path: Path, arrays: dict[str, np.ndarray], provenance: dict) -> str:
    path = Path(path)
    if path.exists():
        return verify_window(path, provenance)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.npz.tmp')
    with temp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    digest = file_sha256(temp)
    # Sidecar first makes a crash recoverable without trusting an unbound NPZ.
    _atomic_json(path.with_suffix('.provenance.json'), {'provenance': provenance, 'sha256': digest})
    os.replace(temp, path)
    return digest


def verify_window(path: Path, provenance: dict) -> str:
    path = Path(path)
    sidecar = path.with_suffix('.provenance.json')
    if not sidecar.exists():
        raise ValueError(f'cache provenance is missing: {path}')
    stored = json.loads(sidecar.read_text(encoding='utf-8'))
    if stored.get('provenance') != provenance:
        raise ValueError(f'cache provenance mismatch: {path}')
    digest = file_sha256(path)
    if digest != stored.get('sha256'):
        raise ValueError(f'cache checksum mismatch: {path}')
    return digest


def load_decoded_audio_audit(path: Path, dataset_internal_sha256: str,
                             records: list[dict]) -> tuple[dict, dict[str, dict]]:
    """Validate the immutable decoded-PCM inventory used for input support."""
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if payload.get('audit_sha256') != sha256_canonical({k: v for k, v in payload.items() if k != 'audit_sha256'}):
        raise ValueError('decoded audio audit checksum mismatch')
    if payload.get('dataset_manifest_sha256') != dataset_internal_sha256:
        raise ValueError('decoded audio audit dataset identity mismatch')
    entries = payload.get('records')
    if payload.get('record_count') != len(entries or []) or len(entries or []) != len(records):
        raise ValueError('decoded audio audit record count mismatch')
    if payload.get('uncovered_supervision_point_count') != 0:
        raise ValueError('decoded audio audit reports uncovered supervision points')
    expected = {r['song_id']: r for r in records}
    index: dict[str, dict] = {}
    for item in entries or []:
        song_id = item.get('song_id')
        if song_id in index or song_id not in expected:
            raise ValueError('decoded audio audit identity mismatch')
        record = expected[song_id]
        if item.get('logical_song_key') != record.get('logical_song_key'):
            raise ValueError('decoded audio audit identity mismatch')
        if item.get('source_sha256') != record.get('source_audio', {}).get('source_sha256'):
            raise ValueError('decoded audio audit identity mismatch')
        count, rate, duration = item.get('decoded_sample_count'), item.get('sample_rate'), item.get('decoded_duration_seconds')
        if not isinstance(count, int) or count <= 0 or not isinstance(rate, int) or rate <= 0:
            raise ValueError('decoded audio audit has invalid decoded sample metadata')
        if not np.isclose(float(duration), count / rate, atol=1.0 / rate):
            raise ValueError('decoded audio audit duration/sample mismatch')
        outside = item.get('supervised_timestamps_outside_decoded_audio', {})
        if any(outside.get(dim, []) for dim in ('arousal', 'valence')):
            raise ValueError('decoded audio audit reports supervision outside decoded audio')
        index[song_id] = item
    if set(index) != set(expected):
        raise ValueError('decoded audio audit identity mismatch')
    return payload, index


def load_mert(model_root: Path, device: str):
    """Verify every executable/weight artifact before loading local custom code."""
    import torch
    import yaml
    from transformers import AutoModel, Wav2Vec2FeatureExtractor

    pin_path = Path(__file__).resolve().parents[3] / 'configs/upstream/mert-primary.yaml'
    pins = yaml.safe_load(pin_path.read_text(encoding='utf-8'))
    if pins['revision'] != MERT_REVISION:
        raise ValueError('MERT revision changed')
    for role, relative in pins['required_roles'].items():
        if file_sha256(Path(model_root) / relative) != pins['expected_sha256'][role]:
            raise ValueError(f'MERT artifact checksum mismatch: {relative}')
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_root, local_files_only=True)
    model = AutoModel.from_pretrained(model_root, local_files_only=True, trust_remote_code=True)
    model = model.to(device).eval()
    model.requires_grad_(False)

    def extract(wave24: np.ndarray, valid_seconds: float) -> np.ndarray:
        observed = min(len(wave24), int(np.ceil(valid_seconds * 24000)), 45 * 24000)
        if observed < 400:
            raise ValueError('MERT requires at least 400 observed samples')
        processed = processor(wave24[:observed], sampling_rate=24000, return_tensors='pt')
        values = torch.nn.functional.pad(processed['input_values'], (0, 45 * 24000 - observed)).to(device)
        mask = torch.zeros_like(values, dtype=torch.long)
        mask[:, :observed] = 1
        with torch.inference_mode():
            result = model(input_values=values, attention_mask=mask, output_hidden_states=True, return_dict=True)
        return np.stack([result.hidden_states[i][0].float().cpu().numpy() for i in (5, 6)])

    return extract, {'revision': MERT_REVISION, 'artifact_sha256': pins['expected_sha256'],
                     'normalization': 'OBSERVED_SAMPLES_BEFORE_PADDING', 'dtype': 'float32'}


def materialize_features(manifest_path: Path, data_root: Path, output_root: Path, *,
                         device: str = 'cuda', limit: int | None = None,
                         handcrafted_only: bool = False, mert_model_root: Path = DEFAULT_MODEL_ROOT,
                         decoded_audit_path: Path | None = None,
                         progress=None) -> dict:
    """Resume verified windows; a limit never changes the expected population.

    ``limit`` is the number of source songs selected from the beginning of the
    immutable inventory. Repeating a limited run validates and reuses those
    windows; omit it to resume through all songs. No failures exclude songs.
    """
    import importlib.metadata
    import librosa
    import soundfile as sf

    from sc_mv_dmer.data import dmer_protocol
    from sc_mv_dmer.foundation.canonical import sha256_canonical

    manifest_path, data_root, output_root = Path(manifest_path), Path(data_root), Path(output_root)
    dataset = json.loads(manifest_path.read_text(encoding='utf-8'))
    dmer_protocol._verify_manifest(dataset)
    if limit is not None and limit <= 0:
        raise ValueError('limit must be positive')
    records = sorted(dataset['records'], key=lambda r: int(r['logical_song_key']))
    if any(r.get('status') != 'RETAINED' for r in records):
        raise ValueError('full DMER features cannot exclude songs')
    if decoded_audit_path is not None:
        decoded_audit, decoded_index = load_decoded_audio_audit(
            decoded_audit_path, dataset['manifest_sha256'], records)
        decoded_audit_sha256 = file_sha256(decoded_audit_path)
    else:
        # Legacy/unit callers may omit the audit; the production CLI requires it.
        decoded_audit, decoded_index, decoded_audit_sha256 = None, {}, None
    upstream, extractor = None, None
    if not handcrafted_only:
        extractor, upstream = load_mert(Path(mert_model_root), device)
    packages = ['numpy', 'librosa', 'scipy', 'soundfile', 'soxr']
    if not handcrafted_only:
        packages += ['torch', 'transformers']
    provenance = {
        'feature_version': FEATURE_VERSION,
        'dataset_manifest_sha256': file_sha256(manifest_path),
        'dataset_internal_sha256': dataset['manifest_sha256'],
        'processing_code_sha256': file_sha256(Path(__file__)),
        'protocol_code_sha256': file_sha256(Path(dmer_protocol.__file__)),
        'library_versions': {name: importlib.metadata.version(name) for name in packages},
        'handcrafted_only': handcrafted_only, 'mert': upstream,
        'handcrafted': {'sample_rate': 22050, 'n_fft': 2048, 'hop_length': 512, 'center': True,
                        'pad_mode': 'constant', 'mel': 'librosa-slaney-power-db-ref1-no-topdb',
                        'mfcc': 'ortho-dct2-40-including-c0', 'chroma': 'librosa-chroma-cqt-12',
                        'bins_per_octave': 36, 'n_octaves': 7, 'tuning': 0., 'chroma_norm': 'RAW_FRAME_L1'},
        'sensor_mode': 'HALF_BEST_MAJOR_MINUS_BEST_MINOR_KS_PEARSON_CORRELATION',
        'sensor_key_source': 'KRUMHANSL_SCHMUCKLER_DIAGNOSTIC_NOT_ESSENTIA',
        'frame_policy': 'CENTRES_HALF_OPEN_2HZ;MERT_REQUIRES_FULL_400_SAMPLE_SUPPORT',
        'decoded_audio_support_policy': DECODED_SUPPORT_POLICY,
        'decoded_audio_audit_sha256': decoded_audit_sha256,
    }
    destination = output_root / 'manifest.json'
    if destination.exists():
        previous = json.loads(destination.read_text(encoding='utf-8'))
        unsigned = {k: v for k, v in previous.items() if k != 'manifest_sha256'}
        if previous.get('manifest_sha256') != sha256_canonical(unsigned):
            raise ValueError('feature manifest checksum mismatch')
        if previous.get('provenance') != provenance:
            raise ValueError('feature cache provenance changed; use a new versioned output root')
    output_root.mkdir(parents=True, exist_ok=True)
    window_records, completed_songs = [], []
    total_windows = sum(len(dmer_protocol.make_windows(record)) for record in records)

    def publish() -> dict:
        all_songs = len(completed_songs) == 1802
        result = {
            'schema_version': '2.0', 'feature_version': FEATURE_VERSION,
            'dataset_manifest_sha256': provenance['dataset_manifest_sha256'],
            'dataset_internal_sha256': dataset['manifest_sha256'], 'provenance': provenance,
            'total_song_count': 1802, 'short_clip_count': 1744, 'full_song_count': 58,
            'completed_song_count': len(completed_songs), 'completed_song_ids': completed_songs.copy(),
            'total_window_count': total_windows, 'completed_window_count': len(window_records),
            'full_population_ready': all_songs, 'four_view_ready': all_songs and not handcrafted_only,
            'formal_training_ready': False, 'sensor_key_source': provenance['sensor_key_source'],
            'sensor_key_essentia_conformant': False,
            'decoded_audio_support_policy': DECODED_SUPPORT_POLICY,
            'decoded_audio_audit_sha256': decoded_audit_sha256,
            'windows': window_records.copy(),
        }
        result['manifest_sha256'] = sha256_canonical(result)
        _atomic_json(destination, result)
        return result

    result = None
    for index, record in enumerate(records[:limit], 1):
        source = record['source_audio']
        relative = Path(source['source_relative_path'])
        if relative.is_absolute() or relative.drive or '..' in relative.parts:
            raise ValueError('source path escapes data root')
        path = data_root / relative
        if file_sha256(path) != source['source_sha256']:
            raise ValueError(f'source checksum mismatch: {record["logical_song_key"]}')
        wave22 = wave24 = None
        decoded_seconds = None
        decoded_samples = None
        decoded_rate = None
        for window in dmer_protocol.make_windows(record):
            decoded = decoded_index.get(record['song_id'])
            if decoded is not None:
                decoded_seconds = float(decoded['decoded_duration_seconds'])
                decoded_samples = int(decoded['decoded_sample_count'])
                decoded_rate = int(decoded['sample_rate'])
            else:
                decoded_seconds = float(record['audio_duration_seconds'])
            valid_seconds = min(45., max(0., decoded_seconds - window['start_ms'] / 1000.))
            if valid_seconds <= 0:
                raise ValueError(f'window has no decoded audio support: {record["logical_song_key"]}/{window["window_id"]}')
            binding = provenance | {'source_audio_sha256': source['source_sha256'],
                                    'window_id': window['window_id'], 'start_ms': window['start_ms'],
                                    'registered_audio_duration_seconds': record['audio_duration_seconds'],
                                    'decoded_audio_duration_seconds': decoded_seconds,
                                    'decoded_sample_count': decoded_samples,
                                    'decoded_sample_rate': decoded_rate}
            relative_cache = Path('windows') / (window['window_id'] + '.npz')
            cache_path = output_root / relative_cache
            if cache_path.exists():
                digest = verify_window(cache_path, binding)
            else:
                if wave22 is None:
                    mono, sr = sf.read(path, dtype='float32', always_2d=True)
                    mono = mono.mean(axis=1, dtype=np.float32)
                    if decoded is not None and (len(mono) != decoded_samples or int(sr) != decoded_rate):
                        raise ValueError(f'decoded audio audit mismatch: {record["logical_song_key"]}')
                    decoded_seconds = len(mono) / float(sr)
                    wave22 = librosa.resample(mono, orig_sr=sr, target_sr=22050, res_type='soxr_hq')
                    if not handcrafted_only:
                        wave24 = librosa.resample(mono, orig_sr=sr, target_sr=24000, res_type='soxr_hq')
                start22 = round(window['start_ms'] * 22050 / 1000)
                arrays = extract_handcrafted_window(wave22[start22:start22 + 45 * 22050], valid_seconds=valid_seconds)
                expected_decoded_coverage = np.clip(
                    (valid_seconds - np.arange(90) * .5) / .5, 0., 1.).astype(np.float32)
                np.testing.assert_allclose(arrays['audio_coverage'], expected_decoded_coverage, atol=1e-6)
                arrays['input_time_ms'] = np.asarray(window['input_time_ms'], dtype=np.int64)
                arrays['output_time_ms'] = np.asarray(window['output_time_ms'], dtype=np.int64)
                if extractor is not None:
                    start24 = round(window['start_ms'] * 24000 / 1000)
                    layers = extractor(wave24[start24:start24 + 45 * 24000], valid_seconds)
                    attach_mert_layers(arrays, layers, valid_seconds=valid_seconds)
                digest = save_window(cache_path, arrays, binding)
            window_records.append({k: window[k] for k in ('window_id', 'song_id', 'logical_song_key', 'start_ms')}
                                  | {'path': relative_cache.as_posix(), 'sha256': digest})
        completed_songs.append(record['song_id'])
        result = publish()
        if progress:
            progress(f'features verified {index}/1802 songs; {len(window_records)}/{total_windows} windows')
    return result or publish()
