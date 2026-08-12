"""Pinned MERT identity and fail-closed local-byte binding."""

from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import Field

from sc_mv_dmer.foundation.canonical import sha256_canonical
from sc_mv_dmer.foundation.manifests import ImmutableRecord


PINNED_MERT_REPOSITORY = "m-a-p/MERT-v1-95M"
PINNED_MERT_REVISION = "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5"
PINNED_MERT_LICENSE = "cc-by-nc-4.0"
_ROLES = {"config": "config.json", "processor": "preprocessor_config.json", "weights": "pytorch_model.bin"}


class UpstreamModelBlocker(RuntimeError):
    """Raised when a formal model bind lacks locally verified required bytes."""


class MertUpstreamStatus(ImmutableRecord):
    repository: str
    revision: str
    license: str
    required_roles: tuple[str, ...]
    local_files: dict[str, dict[str, str]]
    expected_checksums: dict[str, str]
    missing_roles: tuple[str, ...]
    checksum_mismatches: tuple[str, ...]
    local_verification_status: str
    upstream_identity: str


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def discover_mert_local(local_root: Path | None, *, revision: str = PINNED_MERT_REVISION, expected_checksums: dict[str, str] | None = None) -> MertUpstreamStatus:
    if revision != PINNED_MERT_REVISION:
        raise ValueError("MERT revision must be pinned to the frozen revision")
    files: dict[str, dict[str, str]] = {}
    missing: list[str] = []
    expected = dict(expected_checksums or {})
    mismatches: list[str] = []
    for role, filename in _ROLES.items():
        path = Path(local_root) / filename if local_root is not None else None
        if path is None or not path.is_file():
            missing.append(role)
        else:
            actual = _hash(path)
            files[role] = {"source_relative_path": filename, "sha256": actual}
            if role in expected and expected[role] != actual:
                mismatches.append(role)
    identity = sha256_canonical({"repository": PINNED_MERT_REPOSITORY, "revision": revision, "license": PINNED_MERT_LICENSE, "expected_checksums": expected})
    unregistered = [role for role in _ROLES if role not in expected]
    status = "EXPECTED_CHECKSUMS_UNREGISTERED" if unregistered else "BLOCKED_MISSING_LOCAL_BYTES" if missing else "CHECKSUM_MISMATCH" if mismatches else "VERIFIED"
    return MertUpstreamStatus(
        repository=PINNED_MERT_REPOSITORY, revision=revision, license=PINNED_MERT_LICENSE,
        required_roles=tuple(_ROLES), local_files=files, expected_checksums=expected, missing_roles=tuple(missing), checksum_mismatches=tuple(mismatches),
        local_verification_status=status,
        upstream_identity=identity,
    )


def bind_mert_formal(status: MertUpstreamStatus) -> MertUpstreamStatus:
    if status.local_verification_status == "EXPECTED_CHECKSUMS_UNREGISTERED":
        raise UpstreamModelBlocker("expected checksums are unregistered")
    if status.missing_roles:
        raise UpstreamModelBlocker("missing required local MERT roles: " + ", ".join(status.missing_roles))
    if status.checksum_mismatches:
        raise UpstreamModelBlocker("checksum mismatch for roles: " + ", ".join(status.checksum_mismatches))
    return status
