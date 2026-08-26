def _batch():
    import torch

    return {name: torch.randn(2, 90, dim) for name, dim in {"mel": 128, "mfcc": 40, "chroma": 12}.items()}


def test_sensor_branch_is_handcrafted_only_and_gradients_reach_encoders():
    import torch
    from sc_mv_dmer.training.sensor_executor import SensorLossContract, SensorOnlyModel

    model = SensorOnlyModel(hidden_dim=8, view_dropout_p=0.9).train()
    output = model(_batch())
    assert set(output["sensor"]) == {"rms", "brightness", "mode", "key"}
    assert set(output["encoded_views"]) == {"mel", "mfcc", "chroma"}
    targets = {name: torch.randn(2, 90, 1) for name in ("rms", "brightness", "mode")}
    targets["key"] = torch.nn.functional.one_hot(torch.randint(0, 12, (2, 90)), 12).float()
    masks = {name: torch.ones(2, 90, dtype=torch.bool) for name in targets}
    losses = SensorLossContract.compute(output["sensor"], targets, masks)
    assert losses.total is not None
    losses.total.backward()
    assert all(model.encoders[name].projection.weight.grad is not None for name in ("mel", "mfcc", "chroma"))


def test_view_dropout_does_not_change_sensor_supervision_path():
    import torch
    from sc_mv_dmer.training.sensor_executor import SensorOnlyModel

    model = SensorOnlyModel(hidden_dim=8, view_dropout_p=0.9)
    values = _batch()
    model.eval()
    evaluation = model(values)["sensor"]
    model.train()
    training = model(values)["sensor"]
    for name in evaluation:
        assert torch.equal(evaluation[name], training[name])


def test_key_contract_pools_nine_segments_and_excludes_short_segment():
    import torch
    from sc_mv_dmer.training.sensor_executor import SensorLossContract

    logits = torch.randn(1, 90, 12)
    targets = torch.nn.functional.one_hot(torch.zeros(1, 90, dtype=torch.long), 12).float()
    mask = torch.ones(1, 90, dtype=torch.bool)
    mask[:, :6] = False
    term = SensorLossContract.key_ce(logits, targets, mask)
    assert term.denominator == 8
    assert term.eligibility


def test_formal_executor_dry_run_cannot_start_training():
    from sc_mv_dmer.training.sensor_executor import SensorFormalExecutor, TrainingExecutionNotAuthorized

    executor = SensorFormalExecutor()
    assert executor.dry_run()["training_started"] is False
    try:
        executor.fit()
    except TrainingExecutionNotAuthorized:
        pass
    else:
        raise AssertionError("fit must be explicitly blocked by the dry-run executor facade")
