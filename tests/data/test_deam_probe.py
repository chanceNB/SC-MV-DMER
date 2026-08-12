from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from sc_mv_dmer.data.discovery import discover_deam_source, read_windows_audio_duration
from sc_mv_dmer.data.probes import PROBE_SECONDS, TARGET_SAMPLE_RATE, build_probe, write_probe_manifest
from sc_mv_dmer.cli import main


def duration(path):
    return {"2.mp3": 45.087, "10.mp3": 46.0, "1.mp3": 44.99}[path.name]


def deam_root(tmp_path):
    audio = tmp_path / "DEAM_audio" / "MEMD_audio"
    audio.mkdir(parents=True)
    for name, payload in (("10.mp3", b"ten"), ("2.mp3", b"two"), ("1.mp3", b"short")):
        (audio / name).write_bytes(payload)
    return tmp_path


def test_discovery_selects_smallest_numeric_eligible_logical_song_and_is_relocation_invariant(tmp_path):
    root = deam_root(tmp_path / "one")
    relocated = deam_root(tmp_path / "two")

    first = discover_deam_source(root, duration_reader=duration)
    second = discover_deam_source(relocated, duration_reader=duration)

    assert first.source_relative_path == "DEAM_audio/MEMD_audio/2.mp3"
    assert first.dataset_id == second.dataset_id
    assert first.song_id == second.song_id
    assert first.source_sha256 == second.source_sha256
    assert first.source_size_bytes == 3


def test_probe_rejects_short_audio_and_binds_exact_unexecuted_45_second_contract(tmp_path):
    root = deam_root(tmp_path)
    with pytest.raises(ValueError, match="45"):
        build_probe(discover_deam_source(root, duration_reader=lambda _: 44.0))

    probe = build_probe(discover_deam_source(root, duration_reader=duration))
    assert probe.probe_seconds == PROBE_SECONDS == 45
    assert probe.target_sample_rate_hz == TARGET_SAMPLE_RATE == 24_000
    assert probe.target_sample_count == 1_080_000
    assert probe.preparation_status == "SPECIFIED_NOT_EXECUTED"
    assert "MERT" not in probe.model_dump_json()
    assert str(root) not in probe.semantic_identity_json


def test_source_mutation_changes_registered_hash_and_deterministic_write_refuses_overwrite(tmp_path):
    root = deam_root(tmp_path / "data")
    before = discover_deam_source(root, duration_reader=duration)
    (root / "DEAM_audio" / "MEMD_audio" / "2.mp3").write_bytes(b"changed")
    after = discover_deam_source(root, duration_reader=duration)
    assert before.source_sha256 != after.source_sha256

    output = tmp_path / "probe.json"
    write_probe_manifest(build_probe(after), output)
    assert json.loads(output.read_text())["source_relative_path"] == "DEAM_audio/MEMD_audio/2.mp3"
    with pytest.raises(FileExistsError):
        write_probe_manifest(build_probe(after), output)


def test_cli_registers_probe_from_explicit_runtime_data_root_without_overwrite(tmp_path, monkeypatch):
    root = deam_root(tmp_path / "data")
    output = tmp_path / "probe.json"
    import sc_mv_dmer.cli as cli

    monkeypatch.setattr(cli, "discover_deam_source", lambda data_root: discover_deam_source(root, duration_reader=duration))
    assert main(["register-deam-probe", "--data-root", str(root), "--output", str(output)]) == 0
    with pytest.raises(FileExistsError):
        main(["register-deam-probe", "--data-root", str(root), "--output", str(output)])


def test_duration_reader_passes_untrusted_path_only_through_task_specific_environment(monkeypatch, tmp_path):
    path = tmp_path / "O'Brien; Write-Error injected" / "2.mp3"
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(stdout="00:00:45")

    import sc_mv_dmer.data.discovery as discovery

    monkeypatch.setattr(discovery.subprocess, "run", fake_run)
    assert read_windows_audio_duration(path) == 45.0
    assert str(path) not in captured["command"][-1]
    assert "Write-Error injected" not in captured["command"][-1]
    assert captured["kwargs"]["env"]["SC_MV_DMER_DURATION_SOURCE"] == str(path)
    assert captured["kwargs"]["env"]["PATH"] == os.environ["PATH"]
