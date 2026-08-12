from __future__ import annotations

import pytest

from sc_mv_dmer.foundation.identity import stable_id


def test_stable_id_is_deterministic_and_independent_of_data_location() -> None:
    components = ("dataset:deam-v1", "song:stable-recording-42", "0.0", "45.0")

    first = stable_id("sample", components)
    relocated = stable_id("sample", components)

    assert first == relocated
    assert first.startswith("sample_")


@pytest.mark.parametrize(
    ("namespace", "parts"),
    [
        ("", ("dataset:deam-v1",)),
        ("sample", ()),
        ("sample", ("",)),
    ],
)
def test_stable_id_rejects_empty_identity_components(
    namespace: str, parts: tuple[str, ...]
) -> None:
    with pytest.raises(ValueError, match="empty"):
        stable_id(namespace, parts)
