"""Full-population feature boundaries and resume integrity."""
import importlib
import importlib.util

import numpy as np
import pytest
import json
from pathlib import Path


@pytest.fixture
def api():
    name = 'sc_mv_dmer.features.dmer_full'
    assert importlib.util.find_spec(name), 'full DMER feature extraction is not implemented'
    return importlib.import_module(name)


def test_partial_terminal_bin_keeps_observed_frames(api):
    values = np.array([[2.0], [4.0], [20.0], [999.0]])
    result, valid = api.pool_to_2hz(values, np.array([0.0, .49, .62, .66]), valid_seconds=.643)
    np.testing.assert_allclose(result[:3, 0], [3., 20., 0.])
    np.testing.assert_array_equal(valid[:3], [True, True, False])


def test_decoded_audio_audit_is_the_feature_support_authority(api, tmp_path):
    from sc_mv_dmer.foundation.canonical import sha256_canonical
    record = {'song_id': 'song-a', 'logical_song_key': '2',
              'source_audio': {'source_sha256': 'a' * 64}}
    payload = {'schema_version': '1.0', 'dataset_manifest_sha256': 'dataset-hash',
               'decoder': 'soundfile 0.14.0', 'record_count': 1,
               'metadata_diff_count': 1, 'uncovered_supervision_point_count': 0,
               'elapsed_seconds': 0.1, 'records': [{
                   'song_id': 'song-a', 'logical_song_key': '2', 'source_sha256': 'a' * 64,
                   'registered_duration_seconds': 45.33, 'decoded_duration_seconds': 45.087,
                   'decoded_sample_count': 1988337, 'sample_rate': 44100,
                   'difference_seconds': -.243,
                   'supervised_timestamps_outside_decoded_audio': {'valence': [], 'arousal': []}}]}
    payload['audit_sha256'] = sha256_canonical(payload)
    path = tmp_path / 'decoded.json'
    path.write_text(json.dumps(payload), encoding='utf-8')
    loaded, index = api.load_decoded_audio_audit(path, 'dataset-hash', [record])
    assert loaded['uncovered_supervision_point_count'] == 0
    assert index['song-a']['decoded_duration_seconds'] == 45.087


def test_decoded_audio_audit_rejects_uncovered_labels_and_identity_mismatch(api, tmp_path):
    from sc_mv_dmer.foundation.canonical import sha256_canonical
    record = {'song_id': 'song-a', 'logical_song_key': '2',
              'source_audio': {'source_sha256': 'a' * 64}}
    base = {'schema_version': '1.0', 'dataset_manifest_sha256': 'dataset-hash',
            'decoder': 'soundfile 0.14.0', 'record_count': 1, 'metadata_diff_count': 1,
            'uncovered_supervision_point_count': 1, 'elapsed_seconds': .1,
            'records': [{'song_id': 'song-a', 'logical_song_key': '2',
                         'source_sha256': 'a' * 64, 'registered_duration_seconds': 45.33,
                         'decoded_duration_seconds': 44.0, 'decoded_sample_count': 1940400,
                         'sample_rate': 44100, 'difference_seconds': -1.33,
                         'supervised_timestamps_outside_decoded_audio': {'valence': [44500], 'arousal': []}}]}
    base['audit_sha256'] = sha256_canonical(base)
    path = tmp_path / 'bad.json'; path.write_text(json.dumps(base), encoding='utf-8')
    with pytest.raises(ValueError, match='supervision'):
        api.load_decoded_audio_audit(path, 'dataset-hash', [record])
    base['uncovered_supervision_point_count'] = 0
    base['records'][0]['supervised_timestamps_outside_decoded_audio'] = {'valence': [], 'arousal': []}
    base['records'][0]['source_sha256'] = 'b' * 64
    base['audit_sha256'] = sha256_canonical({k: v for k, v in base.items() if k != 'audit_sha256'})
    path.write_text(json.dumps(base), encoding='utf-8')
    with pytest.raises(ValueError, match='identity'):
        api.load_decoded_audio_audit(path, 'dataset-hash', [record])


def test_actual_decoded_duration_controls_coverage_without_changing_grid(api):
    arrays = api.extract_handcrafted_window(np.zeros(round(44.1 * 22050), np.float32),
                                            valid_seconds=44.1)
    assert arrays['audio_mask'].shape == (90,)
    assert arrays['audio_mask'][:89].all() and not arrays['audio_mask'][89]
    assert arrays['audio_coverage'][88] == pytest.approx(.2, abs=1e-5)


def test_ks_profiles_identify_transposition_and_silence(api):
    major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    mode, key, valid = api.ks_mode_key(np.stack([major, np.roll(major, 7), minor, np.zeros(12)]))
    assert mode[0] > 0 and mode[1] > 0 and mode[2] < 0
    assert np.max(np.abs(mode)) <= 1
    np.testing.assert_array_equal(key[:3], [0, 7, 0])
    np.testing.assert_array_equal(valid, [True, True, True, False])


def test_mert_cache_preserves_layers_and_masks_padding(api):
    layers = np.zeros((2, 3374, 768), dtype=np.float32)
    layers[0] = 2.
    layers[1] = 6.
    arrays = {'audio_mask': np.r_[np.ones(2, dtype=bool), np.zeros(88, dtype=bool)],
              'audio_coverage': np.r_[1., .286, np.zeros(88)].astype(np.float32)}
    api.attach_mert_layers(arrays, layers, valid_seconds=.643)
    assert arrays['mert_layers'].dtype == np.float16
    np.testing.assert_array_equal(arrays['mert_layers'][:, 0, 0], [2., 6.])
    np.testing.assert_allclose(arrays['mert_time_seconds'][:2], [399/48000, 1039/48000])
    np.testing.assert_allclose(arrays['deep'][:3, 0], [4., 4., 0.])
    # Centres alone would include a 48th frame whose 400-sample support
    # extends beyond 0.643 seconds; the upstream attention mask excludes it.
    assert arrays['mert_mask'].sum() == 47


def test_handcrafted_real_cqt_retains_partial_tail_and_invalid_silence(api):
    sr = 22050
    t = np.arange(sr * 2) / sr
    wave = np.sin(2 * np.pi * 261.625565 * t).astype(np.float32) * .1
    out = api.extract_handcrafted_window(wave, valid_seconds=1.643)
    assert out['mel'].shape == (90, 128)
    assert out['mfcc'].shape == (90, 40)
    assert out['chroma'].shape == (90, 12)
    assert out['audio_mask'].sum() == 4
    assert out['audio_coverage'][3] == pytest.approx(.286)
    assert out['chroma'][3].sum() == pytest.approx(1.)
    assert out['sensor_valid_mask'][3, 0]
    assert not out['sensor_valid_mask'][4:].any()
    assert not out['chroma'][4:].any()
    silence = api.extract_handcrafted_window(np.zeros(sr * 2, dtype=np.float32), valid_seconds=2.)
    assert not silence['sensor_valid_mask'][:, 2].any()
    assert not silence['sensor_key_mask'].any()


def test_window_resume_rejects_tampered_payload_and_provenance(api, tmp_path):
    path = tmp_path / 'window.npz'
    provenance = {'dataset_manifest_sha256': 'abc', 'source_audio_sha256': 'def'}
    expected = api.save_window(path, {'mel': np.zeros((90, 128), dtype=np.float32)}, provenance)
    assert api.verify_window(path, provenance) == expected
    with pytest.raises(ValueError, match='provenance'):
        api.verify_window(path, provenance | {'dataset_manifest_sha256': 'other'})
    with path.open('ab') as f:
        f.write(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        api.verify_window(path, provenance)


def test_materializer_keeps_full_inventory_when_limited_and_verifies_raw_sources(api, tmp_path):
    import json
    import soundfile as sf
    from sc_mv_dmer.foundation.canonical import canonical_json, sha256_canonical

    audio = tmp_path / 'tone.wav'
    sf.write(audio, np.sin(np.arange(22050 * 2) * 2 * np.pi * 440 / 22050) * .1, 22050)
    source = {'source_relative_path': 'tone.wav', 'source_sha256': api.file_sha256(audio)}
    records = [{'song_id': f'song{i}', 'sample_id': f'sample{i}', 'logical_song_key': str(i),
                'audio_duration_seconds': 2., 'status': 'RETAINED', 'source_audio': source,
                'population_kind': 'SHORT_CLIP' if i <= 1744 else 'FULL_SONG'}
               for i in range(1, 1803)]
    manifest = {'manifest_id': 'DEAM-DMER-FULL-v1', 'dataset_id': 'full', 'records': records,
                'artifacts': {'dmer_targets': {'record_count': 1802, 'file_sha256': '0' * 64}}}
    manifest['manifest_sha256'] = sha256_canonical(manifest)
    path = tmp_path / 'dataset.json'
    path.write_text(canonical_json(manifest), encoding='utf-8')
    out = api.materialize_features(path, tmp_path, tmp_path / 'cache', limit=1, handcrafted_only=True)
    assert out['total_song_count'] == 1802
    assert out['completed_song_count'] == 1
    assert not out['full_population_ready']
    assert not out['four_view_ready']
    assert out['sensor_key_source'] == 'KRUMHANSL_SCHMUCKLER_DIAGNOSTIC_NOT_ESSENTIA'
    assert len(out['windows']) == 1
    stored = json.loads((tmp_path / 'cache' / 'manifest.json').read_text())
    assert stored == out
    rerun = api.materialize_features(path, tmp_path, tmp_path / 'cache', limit=1, handcrafted_only=True)
    assert rerun == out
    audio.write_bytes(b'tampered')
    with pytest.raises(ValueError, match='source checksum'):
        api.materialize_features(path, tmp_path, tmp_path / 'cache', limit=1, handcrafted_only=True)
