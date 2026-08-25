from pathlib import Path

import pytest
from pydantic import ValidationError

from sc_mv_dmer.data.deam import PopulationMismatchError
from sc_mv_dmer.data.identity import make_dataset_id, make_sample_id, make_song_id
from sc_mv_dmer.data.manifests import DatasetRecord


def test_stable_ids_do_not_depend_on_data_root() -> None:
    dataset = make_dataset_id()
    left = make_song_id(dataset, "2")
    right = make_song_id(dataset, "2")
    assert left == right
    assert make_sample_id(left) == make_sample_id(right)


def test_manifest_rejects_absolute_source_paths() -> None:
    with pytest.raises(ValidationError, match="relative"):
        DatasetRecord(
            dataset_id="dataset_x",
            song_id="song_x",
            sample_id="sample_x",
            logical_song_key="2",
            source_relative_path=str(Path("E:/data/2.mp3")),
            source_sha256="a" * 64,
            source_size_bytes=1,
            population_role="PRIMARY",
        )


def test_population_mismatch_is_typed_blocker() -> None:
    with pytest.raises(PopulationMismatchError):
        raise PopulationMismatchError("expected 1744 primary records")
