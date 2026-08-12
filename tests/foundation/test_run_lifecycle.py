from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from sc_mv_dmer.foundation.lifecycle import (
    DuplicateRecordError,
    LifecycleStore,
    TerminalRecordMutationError,
    TerminalState,
    finalize_run,
    start_run,
)
from sc_mv_dmer.foundation.manifests import ArtifactRef, RunSpec


def artifact(name: str) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=f"artifact-{name}",
        kind="fixture",
        sha256="a" * 64,
        logical_path=f"fixtures/{name}",
        version="v1",
    )


def run_spec(run_id: str = "run-001", **changes: object) -> RunSpec:
    values: dict[str, object] = {
        "run_id": run_id,
        "run_mode": "formal",
        "semantic_config_hash": "b" * 64,
        "resolved_config_hash": "c" * 64,
        "config_snapshot_ref": artifact("config"),
        "config_provenance_ref": artifact("provenance"),
        "git_commit": "1234567890abcdef",
        "git_dirty": False,
        "data_ref": artifact("data"),
        "split_ref": artifact("split"),
        "feature_cache_ref": artifact("cache"),
        "seed": 42,
        "environment": {"python": "3.10", "platform": "test"},
        "launch_command": ("python", "-m", "sc_mv_dmer"),
        "schema_versions": {"run_spec": "1"},
    }
    values.update(changes)
    return RunSpec(**values)


def test_terminal_run_cannot_reopen_or_finalize_twice() -> None:
    active = start_run(run_spec(), lifecycle_store=LifecycleStore())
    manifest = finalize_run(
        active,
        TerminalState.SUCCEEDED,
        terminal_reason="completed expected fixture work",
        completed_at=datetime(2026, 8, 12, tzinfo=timezone.utc),
    )

    assert manifest.terminal_state is TerminalState.SUCCEEDED
    with pytest.raises(TerminalRecordMutationError, match="finalized"):
        finalize_run(active, TerminalState.FAILED, terminal_reason="rewrite history")


def test_lifecycle_store_rejects_duplicate_run_identity_before_two_handles_can_finalize() -> None:
    store = LifecycleStore()
    first = start_run(run_spec(), lifecycle_store=store)

    with pytest.raises(DuplicateRecordError, match="run ID"):
        start_run(run_spec(), lifecycle_store=store)

    finalize_run(first, TerminalState.SUCCEEDED, terminal_reason="completed fixture")
    with pytest.raises(DuplicateRecordError, match="run ID"):
        start_run(run_spec(), lifecycle_store=store)


def test_start_run_single_argument_facade_uses_default_store_and_rejects_duplicates() -> None:
    spec = run_spec("facade-run-001")
    active = start_run(spec)

    with pytest.raises(DuplicateRecordError, match="run ID"):
        start_run(spec)

    finalize_run(active, TerminalState.SUCCEEDED, terminal_reason="completed fixture")


def test_terminal_state_enum_excludes_running() -> None:
    assert {state.value for state in TerminalState} == {
        "SUCCEEDED",
        "FAILED",
        "INTERRUPTED",
    }


def test_run_spec_accepts_only_existing_formal_or_debug_run_modes() -> None:
    assert run_spec(run_mode="formal").run_mode.value == "formal"
    assert run_spec(run_mode="debug").run_mode.value == "debug"

    with pytest.raises(ValidationError, match="run_mode"):
        run_spec(run_mode="reproducibility")


def test_continuation_has_new_identity_and_preserves_provenance() -> None:
    original = run_spec("run-original")
    continuation = run_spec(
        "run-continuation",
        parent_run_id=original.run_id,
        continuation_of_run_id=original.run_id,
        resume_checkpoint_ref=artifact("checkpoint"),
    )

    assert continuation.run_id != original.run_id
    assert continuation.parent_run_id == "run-original"
    assert continuation.continuation_of_run_id == "run-original"
    assert continuation.resume_checkpoint_ref == artifact("checkpoint")


def test_finalized_nested_values_are_not_mutable_through_aliases() -> None:
    metrics = {"scores": {"accuracy": 0.75}}
    checkpoints = [artifact("checkpoint")]
    artifacts = [artifact("output")]
    manifest = finalize_run(
        start_run(run_spec(), lifecycle_store=LifecycleStore()),
        TerminalState.SUCCEEDED,
        terminal_reason="completed expected fixture work",
        metrics=metrics,
        checkpoints=checkpoints,
        artifacts=artifacts,
    )

    metrics["scores"]["accuracy"] = 0.0
    checkpoints.append(artifact("later-checkpoint"))
    artifacts.append(artifact("later-output"))

    assert manifest.metrics["scores"]["accuracy"] == 0.75
    assert len(manifest.checkpoints) == 1
    assert len(manifest.artifacts) == 1
    with pytest.raises(TypeError):
        manifest.metrics["scores"]["accuracy"] = 1.0


def test_immutable_records_remain_json_serializable() -> None:
    manifest = finalize_run(
        start_run(run_spec(), lifecycle_store=LifecycleStore()),
        TerminalState.SUCCEEDED,
        terminal_reason="completed expected fixture work",
        metrics={"scores": {"accuracy": 0.75}},
    )

    dumped = manifest.model_dump_json()

    assert '"accuracy":0.75' in dumped


def test_lifecycle_schemas_are_valid_json() -> None:
    schema_directory = Path(__file__).parents[2] / "schemas"

    for filename in (
        "run_spec.schema.json",
        "run_manifest.schema.json",
        "invalidation_record.schema.json",
        "gate_evaluation.schema.json",
    ):
        schema = json.loads((schema_directory / filename).read_text(encoding="utf-8"))
        assert schema["additionalProperties"] is False
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
