from sc_mv_dmer.data.training_contract_preflight import build_v3_training_contract


def test_v3_training_budget_is_recomputed_for_995_population():
    artifact = build_v3_training_contract(manifest_sha256="a" * 64)
    budget = artifact["resolved"]["budget"]
    assert budget["samples_per_epoch"] == 995
    assert budget["batches_per_epoch"] == 249
    assert budget["updates_per_epoch"] == 63
    assert budget["total_updates"] == 315
    assert budget["warmup_updates"] == 32
    assert budget["checkpoint_boundaries_updates"] == [63, 126, 189, 252, 315]
    assert artifact["training_authorized"] is False
    assert artifact["semantic"]["architecture_identity"] == "A1-v1"
