import numpy as np

from sc_mv_dmer.data.coverage import compile_input_coverage


def test_input_and_target_masks_remain_distinct():
    input_mask = np.ones(90, dtype=np.bool_)
    target_mask = input_mask.copy()
    target_mask[0] = False
    coverage = compile_input_coverage(input_mask, target_mask, np.ones(90, dtype=np.bool_), np.ones(90, dtype=np.bool_))
    assert coverage.input_coverage_mask_hash != coverage.target_validity_mask_hash
