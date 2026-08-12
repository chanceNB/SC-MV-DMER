from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from sc_mv_dmer.features.mert_snapshot import (
    PINNED_MERT_REVISION,
    SnapshotBindingError,
    build_pinned_mert_manifest,
    verify_pinned_mert_manifest,
    write_immutable_manifest,
)


REQUIRED_FILES = (
    "config.json",
    "preprocessor_config.json",
    "configuration_MERT.py",
    "modeling_MERT.py",
    "pytorch_model.bin",
)


def snapshot_fixture(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["MERTModel"],
                "auto_map": {
                    "AutoConfig": "configuration_MERT.MERTConfig",
                    "AutoModel": "modeling_MERT.MERTModel",
                },
                "model_type": "mert_model",
                "sample_rate": 24000,
                "transformers_version": "4.24.0",
            }
        ),
        encoding="utf-8",
    )
    (root / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "feature_extractor_type": "Wav2Vec2FeatureExtractor",
                "sampling_rate": 24000,
            }
        ),
        encoding="utf-8",
    )
    (root / "configuration_MERT.py").write_text(
        "from transformers.configuration_utils import PretrainedConfig\n"
        "class MERTConfig(PretrainedConfig):\n    model_type = 'mert_model'\n",
        encoding="utf-8",
    )
    (root / "modeling_MERT.py").write_text(
        "import torch\nfrom .configuration_MERT import MERTConfig\n"
        "class MERTModel: pass\n",
        encoding="utf-8",
    )
    (root / "pytorch_model.bin").write_bytes(b"transformers-weight-bytes")
    return root


@pytest.mark.parametrize("revision", ["main", "latest", "12af15f"])
def test_floating_or_non_exact_revision_is_rejected(tmp_path: Path, revision: str) -> None:
    """A mutable or abbreviated revision must never identify formal bytes."""

    with pytest.raises(SnapshotBindingError, match="exact pinned revision"):
        build_pinned_mert_manifest(snapshot_fixture(tmp_path / revision), revision=revision)


@pytest.mark.parametrize("missing_name", REQUIRED_FILES)
def test_every_canonical_transformers_file_is_required(
    tmp_path: Path, missing_name: str
) -> None:
    """Dropping any actually consumed file must make snapshot binding fail closed."""

    root = snapshot_fixture(tmp_path / "snapshot")
    (root / missing_name).unlink()

    with pytest.raises(SnapshotBindingError, match=missing_name):
        build_pinned_mert_manifest(root, revision=PINNED_MERT_REVISION)


def test_static_loader_contract_and_canonical_weight_are_bound(tmp_path: Path) -> None:
    """The manifest must bind custom AutoModel code and only the Transformers weight."""

    manifest = build_pinned_mert_manifest(
        snapshot_fixture(tmp_path / "snapshot"), revision=PINNED_MERT_REVISION
    )

    assert manifest.repository_id == "m-a-p/MERT-v1-95M"
    assert manifest.revision == PINNED_MERT_REVISION
    assert manifest.canonical_weight_file == "pytorch_model.bin"
    assert manifest.loader_role == "TRANSFORMERS_AUTO_MODEL_LOCAL_PINNED"
    assert manifest.trust_remote_code_required is True
    assert manifest.local_files_only_required is True
    assert manifest.model_class == "MERTModel"
    assert manifest.config_class == "MERTConfig"
    assert manifest.processor_class == "Wav2Vec2FeatureExtractor"
    assert manifest.model_type == "mert_model"
    assert manifest.declared_sample_rate_hz == 24000
    assert {item.relative_path for item in manifest.files} == set(REQUIRED_FILES)
    assert {item.role for item in manifest.files} == {
        "MODEL_CONFIG",
        "PREPROCESSOR_CONFIG",
        "CUSTOM_CONFIGURATION_CODE",
        "CUSTOM_MODELING_CODE",
        "CANONICAL_TRANSFORMERS_WEIGHT",
    }
    assert manifest.other_weight_files == {
        "MERT-v1-95M_fairseq.pt": "NOT_CONSUMED_BY_CANONICAL_TRANSFORMERS_LOADER"
    }


def test_relocated_identical_snapshot_has_same_semantic_identity(tmp_path: Path) -> None:
    """Machine-local placement must not change the formal upstream identity."""

    left = snapshot_fixture(tmp_path / "left")
    right = tmp_path / "right"
    shutil.copytree(left, right)

    a = build_pinned_mert_manifest(left, revision=PINNED_MERT_REVISION)
    b = build_pinned_mert_manifest(right, revision=PINNED_MERT_REVISION)

    assert a.snapshot_aggregate_sha256 == b.snapshot_aggregate_sha256
    assert a.semantic_identity_sha256 == b.semantic_identity_sha256
    assert str(left.resolve()) not in a.model_dump_json()
    assert str(right.resolve()) not in b.model_dump_json()


@pytest.mark.parametrize(
    "mutated_name",
    ["config.json", "preprocessor_config.json", "configuration_MERT.py", "modeling_MERT.py", "pytorch_model.bin"],
)
def test_checksum_or_custom_code_mutation_is_rejected(
    tmp_path: Path, mutated_name: str
) -> None:
    """A manifest cannot authorize bytes changed after the static bind."""

    root = snapshot_fixture(tmp_path / "snapshot")
    manifest = build_pinned_mert_manifest(root, revision=PINNED_MERT_REVISION)
    with (root / mutated_name).open("ab") as changed:
        changed.write(b"tampered")

    with pytest.raises(SnapshotBindingError, match="checksum mismatch"):
        verify_pinned_mert_manifest(manifest, root)


def test_manifest_write_is_exclusive_and_never_overwrites(tmp_path: Path) -> None:
    """A terminal upstream manifest must remain byte-for-byte immutable."""

    manifest = build_pinned_mert_manifest(
        snapshot_fixture(tmp_path / "snapshot"), revision=PINNED_MERT_REVISION
    )
    destination = tmp_path / "mert.json"
    write_immutable_manifest(manifest, destination)
    before = destination.read_bytes()

    with pytest.raises(FileExistsError):
        write_immutable_manifest(manifest, destination)

    assert destination.read_bytes() == before


def test_malformed_static_loader_metadata_is_rejected(tmp_path: Path) -> None:
    """Wrong custom class mapping must not be silently accepted as canonical MERT."""

    root = snapshot_fixture(tmp_path / "snapshot")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    config["auto_map"]["AutoModel"] = "other.Model"
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(SnapshotBindingError, match="AutoModel"):
        build_pinned_mert_manifest(root, revision=PINNED_MERT_REVISION)
