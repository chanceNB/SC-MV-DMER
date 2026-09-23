import json
from pathlib import Path

from sc_mv_dmer.gates.rg03 import dry_run, run_preflight


def test_rg03_dry_run_fails_closed_without_immutable_payload_root():
    repo = Path(__file__).parents[2]
    report = run_preflight(repo)
    assert report.verdict == "BLOCKED"
    assert "FORMAL_EXECUTION_PAUSED_DMER_FULL_POPULATION_REBUILD" in report.blockers
    assert not report.provenance["absolute_paths_serialized"]


def test_rg03_dry_run_declares_contract_without_training():
    repo = Path(__file__).parents[2]
    result = dry_run(repo)
    assert result["dry_run"] is True
    assert result["training_started"] is False
    assert result["checkpoint_published"] is False
    assert result["rg03_pass_evidence_published"] is False
    assert result["planned_population"] == {"optimization_train": 896, "validation": 99, "test": 58}
    assert result["selection"] == {"metric": "validation CE", "cadence": "every epoch", "patience": 10, "min_delta": 1e-4, "tie_break": ["earliest epoch", "smallest global step"]}
