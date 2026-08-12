"""Stable, location-independent identifiers."""

from __future__ import annotations

from collections.abc import Sequence

from sc_mv_dmer.foundation.canonical import sha256_canonical


def stable_id(namespace: str, parts: Sequence[str]) -> str:
    """Build a namespace-separated ID from stable logical identity components."""

    if not isinstance(namespace, str) or not namespace.strip():
        raise ValueError("identity namespace must not be empty")
    if not parts:
        raise ValueError("identity parts must not be empty")
    if any(not isinstance(part, str) or not part.strip() for part in parts):
        raise ValueError("identity parts must not contain empty values")

    digest = sha256_canonical({"namespace": namespace, "parts": list(parts)})
    return f"{namespace}_{digest}"
