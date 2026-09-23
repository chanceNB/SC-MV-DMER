import pytest

from sc_mv_dmer.evaluation.experiments import aggregate_five_seeds, check_frozen_run
from sc_mv_dmer.training.dmer_acoustic import SEEDS


def test_seed_aggregation_rejects_mixed_splits_and_duplicate_seeds():
    def make(seed, v):
        return {"spec":{"seed":seed,"variant":"crnn","split_sha256":"same","mode":"formal"},
                "metrics":{"macro":{"valence":{"ccc":v,"pcc":v,"rmse":.2},
                                      "arousal":{"ccc":v,"pcc":v,"rmse":.2}}}}
    rows=[make(seed, i/10) for i,seed in enumerate(SEEDS)]
    summary=aggregate_five_seeds(rows)
    assert summary["macro"]["valence"]["ccc"]["mean"]==pytest.approx(.2)
    assert summary["macro"]["valence"]["ccc"]["std"]==pytest.approx(.158113883)
    rows[-1]["spec"]["split_sha256"]="different"
    with pytest.raises(ValueError,match="split"):
        aggregate_five_seeds(rows)
    rows[-1]["spec"]["split_sha256"]="same"
    rows[-1]["spec"]["seed"]=SEEDS[0]
    with pytest.raises(ValueError,match="seed"):
        aggregate_five_seeds(rows)


def test_heldout_evaluation_requires_successful_formal_frozen_source():
    with pytest.raises(ValueError,match="formal"):
        check_frozen_run({"mode":"development"},{"status":"SUCCEEDED"})
    with pytest.raises(ValueError,match="SUCCEEDED"):
        check_frozen_run({"mode":"formal"},{"status":"FAILED"})
    check_frozen_run({"mode":"formal"},{"status":"SUCCEEDED"})
