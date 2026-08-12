from __future__ import annotations

import pytest

from sc_mv_dmer.features.mert_upstream import (
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
    assert status.local_verification_status == "BLOCKED_MISSING_LOCAL_BYTES"
    with pytest.raises(UpstreamModelBlocker, match="missing"):
        bind_mert_formal(status)


def test_complete_fixture_inventory_binds_and_is_relocation_invariant(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    for root in (first, second):
        root.mkdir()
        for filename, content in (("config.json", b"config"), ("preprocessor_config.json", b"processor"), ("pytorch_model.bin", b"weights")):
            (root / filename).write_bytes(content)

    bound_first = bind_mert_formal(discover_mert_local(first))
    bound_second = bind_mert_formal(discover_mert_local(second))
    assert bound_first.local_verification_status == "VERIFIED"
    assert bound_first.upstream_identity == bound_second.upstream_identity


def test_floating_mert_revision_is_rejected():
    with pytest.raises(ValueError, match="pinned"):
        discover_mert_local(None, revision="main")
