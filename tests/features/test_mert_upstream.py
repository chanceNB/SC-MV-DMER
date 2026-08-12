from __future__ import annotations

import pytest
import hashlib

from sc_mv_dmer.features.mert_upstream import (
    MertUpstreamStatus,
    PINNED_MERT_REPOSITORY,
    PINNED_MERT_REVISION,
    UpstreamModelBlocker,
    bind_mert_formal,
    discover_mert_local,
)


def test_mert_identity_is_pinned_and_missing_local_bytes_fail_closed(tmp_path):
    status = discover_mert_local(tmp_path)
    assert status.repository == PINNED_MERT_REPOSITORY
    assert status.revision == PINNED_MERT_REVISION
    assert status.local_verification_status == "EXPECTED_CHECKSUMS_UNREGISTERED"
    with pytest.raises(UpstreamModelBlocker, match="expected"):
        bind_mert_formal(status)


def test_complete_fixture_inventory_remains_formally_blocked_when_authoritative_registry_is_incomplete(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    for root in (first, second):
        root.mkdir()
        for filename, content in (("config.json", b"config"), ("preprocessor_config.json", b"processor"), ("pytorch_model.bin", b"weights")):
            (root / filename).write_bytes(content)

    first_status = discover_mert_local(first)
    second_status = discover_mert_local(second)
    assert first_status.upstream_identity == second_status.upstream_identity
    assert first_status.local_verification_status == "EXPECTED_CHECKSUMS_UNREGISTERED"
    with pytest.raises(UpstreamModelBlocker, match="expected"):
        bind_mert_formal(first_status)


def test_floating_mert_revision_is_rejected():
    with pytest.raises(ValueError, match="pinned"):
        discover_mert_local(None, revision="main")


def test_production_discovery_does_not_accept_caller_supplied_expected_checksums(tmp_path):
    with pytest.raises(TypeError):
        discover_mert_local(tmp_path, expected_checksums={"config": "a" * 64})


def test_same_named_local_files_fail_closed_without_registered_expected_checksums(tmp_path):
    for filename in ("config.json", "preprocessor_config.json", "pytorch_model.bin"):
        (tmp_path / filename).write_bytes(b"arbitrary")

    unregistered = discover_mert_local(tmp_path)
    assert unregistered.local_verification_status == "EXPECTED_CHECKSUMS_UNREGISTERED"
    with pytest.raises(UpstreamModelBlocker, match="expected"):
        bind_mert_formal(unregistered)



def test_formal_bind_rejects_caller_forged_verified_status():
    forged = MertUpstreamStatus(
        repository=PINNED_MERT_REPOSITORY,
        revision=PINNED_MERT_REVISION,
        license="cc-by-nc-4.0",
        required_roles=("config", "processor", "weights"),
        local_files={},
        expected_checksums={},
        missing_roles=(),
        checksum_mismatches=(),
        local_verification_status="VERIFIED",
        upstream_identity="0" * 64,
    )
    with pytest.raises(UpstreamModelBlocker, match="formal verification"):
        bind_mert_formal(forged)
