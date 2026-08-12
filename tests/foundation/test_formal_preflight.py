from __future__ import annotations

from pathlib import Path

import pytest

from sc_mv_dmer.foundation.config import RunMode, resolve_config
from sc_mv_dmer.foundation.preflight import FormalPreflightError, observe_preflight


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = REPOSITORY_ROOT / "configs" / "research" / "defaults.yaml"


def test_clean_formal_e0_preflight_is_allowed() -> None:
    """Marking a clean formal observation invalid would prevent E0 bootstrap evidence."""

    observation = observe_preflight(
        REPOSITORY_ROOT,
        resolve_config([DEFAULTS], {}, RunMode.FORMAL),
        git_commit="a" * 40,
        git_dirty=False,
    )

    assert observation.formal_allowed is True
    assert observation.paper_eligible is False


def test_dirty_formal_preflight_is_rejected() -> None:
    """Accepting a dirty formal observation would destroy qualification provenance."""

    with pytest.raises(FormalPreflightError, match="clean Git workspace"):
        observe_preflight(
            REPOSITORY_ROOT,
            resolve_config([DEFAULTS], {}, RunMode.FORMAL),
            git_commit="a" * 40,
            git_dirty=True,
        )


def test_dirty_debug_with_semantic_override_is_allowed_but_not_paper_eligible() -> None:
    """Treating debug semantic changes as paper eligible would bypass formal controls."""

    observation = observe_preflight(
        REPOSITORY_ROOT,
        resolve_config([DEFAULTS], {"markov.shared_A": False}, RunMode.DEBUG),
        git_commit="a" * 40,
        git_dirty=True,
    )

    assert observation.formal_allowed is False
    assert observation.paper_eligible is False
