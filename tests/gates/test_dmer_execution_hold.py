import json

from sc_mv_dmer.gates.rg03 import run_preflight


def test_hold_blocks_legacy_training_before_loading_any_data(tmp_path):
    path = tmp_path / 'configs/research/execution-hold.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'status': 'PAUSED', 'reason': 'DMER_FULL_POPULATION_REBUILD'}))
    result = run_preflight(tmp_path)
    assert result.verdict == 'BLOCKED'
    assert 'FORMAL_EXECUTION_PAUSED_DMER_FULL_POPULATION_REBUILD' in result.blockers
