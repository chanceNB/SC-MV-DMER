"""Read-only DEAM source discovery with location-independent identity."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from pathlib import Path
from typing import Callable

from pydantic import Field

from sc_mv_dmer.foundation.identity import stable_id
from sc_mv_dmer.foundation.manifests import ImmutableRecord


DEAM_DATASET_VERSION = "deam-e1-v1"
_AUDIO_DIRECTORY = Path("DEAM_audio/MEMD_audio")


class DeamSource(ImmutableRecord):
    dataset_id: str
    song_id: str
    source_relative_path: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_size_bytes: int = Field(ge=1)
    duration_seconds: float = Field(gt=0)


def _numeric_key(path: Path) -> int:
    match = re.fullmatch(r"(\d+)", path.stem)
    if not match:
        raise ValueError(f"DEAM source filename is not a numeric song key: {path.name}")
    return int(match.group(1))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_windows_audio_duration(path: Path) -> float:
    """Read Windows media metadata only; this never decodes audio or loads a model."""

    script = (
        "$shell=New-Object -ComObject Shell.Application;"
        "$source=Get-Item -LiteralPath $env:SC_MV_DMER_DURATION_SOURCE;"
        "$folder=$shell.Namespace($source.DirectoryName);"
        "$item=$folder.ParseName($source.Name);"
        "$duration=$folder.GetDetailsOf($item,27); Write-Output $duration"
    )
    environment = os.environ.copy()
    environment["SC_MV_DMER_DURATION_SOURCE"] = str(path)
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    values = [int(value) for value in re.findall(r"\d+", result.stdout)]
    if not values:
        raise ValueError(f"Windows media metadata did not provide duration for {path.name}")
    if len(values) == 1:
        return float(values[0])
    if len(values) == 2:
        return float(values[0] * 60 + values[1])
    return float(values[-3] * 3600 + values[-2] * 60 + values[-1])


def discover_deam_source(
    data_root: Path,
    *,
    duration_reader: Callable[[Path], float] = read_windows_audio_duration,
    minimum_seconds: float = 45.0,
) -> DeamSource:
    """Select the smallest numeric eligible source using only a versioned rule."""

    audio_directory = Path(data_root) / _AUDIO_DIRECTORY
    candidates = sorted(audio_directory.glob("*.mp3"), key=_numeric_key)
    for candidate in candidates:
        duration = duration_reader(candidate)
        if duration < minimum_seconds:
            continue
        song_key = str(_numeric_key(candidate))
        dataset_id = stable_id("dataset", ("DEAM", DEAM_DATASET_VERSION))
        return DeamSource(
            dataset_id=dataset_id,
            song_id=stable_id("song", (dataset_id, song_key)),
            source_relative_path=candidate.relative_to(Path(data_root)).as_posix(),
            source_sha256=_sha256(candidate),
            source_size_bytes=candidate.stat().st_size,
            duration_seconds=duration,
        )
    raise ValueError("no DEAM source is eligible for the exact 45 second probe")
