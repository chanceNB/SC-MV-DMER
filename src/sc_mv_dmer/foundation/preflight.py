"""Read-only formal/debug E0 preflight observations."""

from __future__ import annotations

import subprocess
from pathlib import Path

from sc_mv_dmer.foundation.config import ResolvedConfigSnapshot, RunMode
from sc_mv_dmer.foundation.manifests import ImmutableRecord


class FormalPreflightError(ValueError):
    """Raised when a formal observation cannot support clean provenance."""


class PreflightObservation(ImmutableRecord):
    implementation_git_commit: str
    observed_git_dirty: bool
    run_mode: RunMode
    formal_allowed: bool
    paper_eligible: bool
    config_schema_version: str
    research_spec_version: str
    semantic_config_hash: str
    resolved_config_hash: str


def _git_value(workspace: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=workspace, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def observe_preflight(
    workspace: Path,
    config: ResolvedConfigSnapshot,
    *,
    git_commit: str | None = None,
    git_dirty: bool | None = None,
) -> PreflightObservation:
    """Capture pre-write Git/config facts without creating artifacts or executing models."""

    workspace = Path(workspace)
    commit = git_commit if git_commit is not None else _git_value(workspace, "rev-parse", "HEAD")
    dirty = (
        git_dirty
        if git_dirty is not None
        else bool(_git_value(workspace, "status", "--porcelain"))
    )
    if config.run_mode is RunMode.FORMAL and dirty:
        raise FormalPreflightError("formal preflight requires a clean Git workspace")
    return PreflightObservation(
        implementation_git_commit=commit,
        observed_git_dirty=dirty,
        run_mode=config.run_mode,
        formal_allowed=config.run_mode is RunMode.FORMAL and not dirty,
        paper_eligible=False,
        config_schema_version=config.config_schema_version,
        research_spec_version=config.research_spec_version,
        semantic_config_hash=config.semantic_config_hash,
        resolved_config_hash=config.resolved_config_hash,
    )
