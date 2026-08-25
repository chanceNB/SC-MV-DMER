import pytest

from sc_mv_dmer.features.alignment_review import HumanAuthorityRequired, finalize_rg02, select_alignment_sample


def test_alignment_sample_is_three_unique_train_songs():
    sample = select_alignment_sample([f"song-{i}" for i in range(10)])
    assert len(sample) == len(set(sample)) == 3


def test_automatic_result_cannot_supply_human_approval():
    with pytest.raises(HumanAuthorityRequired):
        finalize_rg02({"verdict": "PASS"}, reviewer_records=[])
