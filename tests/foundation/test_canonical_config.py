from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from sc_mv_dmer.foundation.canonical import canonical_json, sha256_canonical
from sc_mv_dmer.foundation.config import (
    FormalOverrideError,
    RunMode,
    UnknownConfigKeyError,
    resolve_config,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = REPOSITORY_ROOT / "configs" / "research" / "defaults.yaml"


def test_canonical_hash_ignores_mapping_key_order() -> None:
    first = {"z": [True, None, {"b": 2, "a": "\u4e2d\u6587"}], "a": 1}
    second = {"a": 1, "z": [True, None, {"a": "\u4e2d\u6587", "b": 2}]}

    assert canonical_json(first) == '{"a":1,"z":[true,null,{"a":"\u4e2d\u6587","b":2}]}'
    assert sha256_canonical(first) == sha256_canonical(second)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_canonical_json_rejects_non_finite_numbers(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        canonical_json({"value": value})


def test_runtime_and_data_root_changes_preserve_semantic_hash() -> None:
    linux = resolve_config(
        [DEFAULTS],
        {
            "runtime.device": "cuda:0",
            "dataset.data_root": "/mnt/research/sc-mv-dmer",
        },
        RunMode.DEBUG,
    )
    windows = resolve_config(
        [DEFAULTS],
        {
            "runtime.device": "cuda:1",
            "dataset.data_root": "D:\\SC-MV-DMER-data",
        },
        RunMode.DEBUG,
    )

    assert linux.semantic_config_hash == windows.semantic_config_hash
    assert linux.resolved_config_hash != windows.resolved_config_hash


def test_formal_rejects_semantic_override() -> None:
    with pytest.raises(FormalOverrideError, match="markov.shared_A"):
        resolve_config([DEFAULTS], {"markov.shared_A": False}, RunMode.FORMAL)


def test_formal_accepts_allowlisted_runtime_override() -> None:
    snapshot = resolve_config(
        [DEFAULTS], {"runtime.batch_size": 8}, RunMode.FORMAL
    )

    assert snapshot.resolved_config["runtime"]["batch_size"] == 8
    assert snapshot.paper_eligible is True


def test_debug_allows_semantic_override_but_is_not_paper_eligible() -> None:
    snapshot = resolve_config(
        [DEFAULTS], {"markov.shared_A": False}, RunMode.DEBUG
    )

    assert snapshot.resolved_config["markov"]["shared_A"] is False
    assert snapshot.paper_eligible is False


def test_unknown_yaml_key_is_rejected(tmp_path: Path) -> None:
    unknown_layer = tmp_path / "unknown.yaml"
    unknown_layer.write_text("unknown_section:\n  value: true\n", encoding="utf-8")

    with pytest.raises(UnknownConfigKeyError, match="unknown_section"):
        resolve_config([DEFAULTS, unknown_layer], {}, RunMode.DEBUG)


def test_unknown_override_key_is_rejected() -> None:
    with pytest.raises(UnknownConfigKeyError, match="runtime.unknown"):
        resolve_config([DEFAULTS], {"runtime.unknown": True}, RunMode.DEBUG)


def test_snapshot_records_canonical_json_provenance_and_versions() -> None:
    snapshot = resolve_config([DEFAULTS], {}, RunMode.FORMAL)

    assert json.loads(snapshot.canonical_json) == snapshot.resolved_config
    assert snapshot.config_schema_version == "1.0"
    assert snapshot.research_spec_version == "1.0"
    assert snapshot.source_provenance == (
        {
            "logical_path": "configs/research/defaults.yaml",
            "sha256": hashlib.sha256(DEFAULTS.read_bytes()).hexdigest(),
            "merge_order": 0,
        },
    )
    assert snapshot.cli_overrides == {}
    assert snapshot.semantic_diff == {}


def test_snapshot_schema_rejects_unknown_top_level_fields() -> None:
    schema_path = REPOSITORY_ROOT / "schemas" / "resolved_config.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert schema["additionalProperties"] is False
    assert {
        "semantic_config_hash",
        "resolved_config_hash",
        "source_provenance",
    }.issubset(schema["required"])
