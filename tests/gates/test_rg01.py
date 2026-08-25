from pathlib import Path

from sc_mv_dmer.gates import rg01
from sc_mv_dmer.gates.rg01 import RG01Evidence, evaluate_rg01


ROOT = Path(__file__).parents[2]


def test_rg01_requires_all_current_effective_predecessors(monkeypatch) -> None:
    monkeypatch.setattr(rg01, "_git_state", lambda _root: ("f8e37024eddc741a62ff53134358ffee866b9c3a", False))
    attempt = evaluate_rg01(RG01Evidence(ROOT))
    assert attempt.verdict == "PASS"
    assert attempt.definition_id == "RG-01"
    assert attempt.authority_metadata.automation_identity == "sc_mv_dmer.rg01.formal_evaluator.v1"


def test_missing_dimension_edge_is_non_pass(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(rg01, "_git_state", lambda _root: ("f8e37024eddc741a62ff53134358ffee866b9c3a", False))
    for source in (
        "evidence/qualifications/e1-1-mert-frame-rate-verification.json",
        "evidence/qualifications/e1-2-mert-downstream-dimension-derivation-v2.json",
        "evidence/qualifications/e1-3-mert-timegrid-binding-v2.json",
        "manifests/dimensions/mert-downstream-dimensions-v2.json",
        "manifests/timegrid/mert-2hz-timegrid-v2.json",
        "manifests/upstream/mert-v1-95m-primary-v1.json",
        "reports/experiments/e1-1-mert-real-frame-rate.json",
    ):
        destination = tmp_path / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / source).read_bytes())
    binding = tmp_path / "manifests/dimensions/mert-downstream-dimensions-v2.json"
    payload = binding.read_text(encoding="utf-8").replace('"target_node_id":"fusion_concat"', '"target_node_id":"missing_node"', 1)
    binding.write_text(payload, encoding="utf-8")
    attempt = evaluate_rg01(RG01Evidence(tmp_path))
    assert attempt.verdict != "PASS"
