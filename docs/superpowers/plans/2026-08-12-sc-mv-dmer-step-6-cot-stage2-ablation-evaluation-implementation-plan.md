# Step 6 — CoT、正式 Stage 2、消融与评估 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 基于 Stage-1 Sensor snapshot 先生成并批准 phase/variant-matched Event/CoT artifacts，再执行正式 Stage 2、A1 COMPLETE_S5、A1–A13 消融、long58/PMEmo、RG-07/08/09/10 和 paper aggregation。

**Architecture:** Event/CoT pipeline 先生成 `S2_EARLY` pseudo-label artifacts 与 `S2_LATE` frozen Stage-1 Sensor-snapshot artifacts；Stage-2 runner 在一个新 run 中按 PB-1+ 在 epoch 3 commit 后切换 artifact，但不重置训练状态。Variant orchestrator 从 canonical base + semantic diff + dependency closure 编译各自 artifact/stage/Gate graph；aggregation 只读取 immutable terminal runs 和 effective-validity resolver。

**Tech Stack:** Python、PyTorch、Transformers/PEFT/bitsandbytes、Pydantic、Jinja2 或等价 deterministic serializer、NumPy、pandas、NVML、pytest、static HTML/CSV review package。

## Global Constraints

- `COTC-SN21-DSDP-3R+`: `N_syn=floor(2N/3+0.5)`，其余为 `NATIVE_DEAM_VA_ONLY`；membership S5-independent、phase-invariant。
- Synthetic songs 缺少 exact approved phase artifact 是 integrity failure；native songs 合法 `L_CoT` zero-by-manifest，但可按能力参与 online `L_consist`。
- Early 使用 pseudo Events；late 使用 frozen Stage-1 Sensor-snapshot Events；live Stage-2 Sensor 在适用变体继续训练。
- Stage 2 有 5 logical epochs：1–3 early、4–5 late；只 designation epoch-5 terminal checkpoint。
- Final A1-like inference 使用 terminal Stage-2 own Sensor-derived Events；A5 shadow-only、A6 paired A1 replay、A10 local VA-only Event path、A2/A13 N/A。
- RG-07: 7B 每条 Event block maximum `<=180` tokens；禁止 silent truncation。
- RG-09: exactly 100 eligible synthetic records、两名独立真实 human reviewers、`qualified>=85`；Codex 不得代替人工。
- RG-10-v2: full primary validation generation population、exact 90 V/90 A、success rate exact `>=0.98`；parse failure 不 retry/repair/remove。
- Final evidence 使用统一 S5；不允许 best-seed、incomplete S5 或 stale dependency 进入 paper aggregation。
- 本机优先；合法 OOM 后保存 evidence/handoff，并在外部新 run_id 继续。14B 非退出条件。

---

### Task 1: 实现 EventSequence、symbolizer、A5 firewall 与 A6 replay

**Files:**
- Create: `src/sc_mv_dmer/events/models.py`
- Create: `src/sc_mv_dmer/events/symbolizer.py`
- Create: `src/sc_mv_dmer/events/serializer.py`
- Create: `src/sc_mv_dmer/events/replay.py`
- Create: `src/sc_mv_dmer/events/visibility.py`
- Create: `schemas/event_sequence.schema.json`
- Create: `schemas/event_replay_bundle.schema.json`
- Create: `tests/events/test_event_sequence.py`
- Create: `tests/events/test_variant_event_boundaries.py`

**Interfaces:**
- Consumes: pseudo-label bundle or Sensor bundle, symbolizer version, segment boundaries and variant capability.
- Produces: `symbolize_events(source: EventSourceBundle, rule: SymbolizerRule) -> EventSequence`; `resolve_event_visibility(...) -> EventDisposition`; `bind_a6_replay(...) -> EventReplayManifest`.

- [ ] **Step 1: Write provenance and visibility tests**

```python
def test_event_sequence_requires_explicit_source_provenance() -> None:
    with pytest.raises(ValidationError):
        EventSequence(event_source="sensor_prediction", segments=SEGMENTS)

def test_a5_is_shadow_only_and_a6_uses_same_seed_a1_replay() -> None:
    assert resolve_event_visibility(a5_profile()).model_visible is False
    with pytest.raises(CrossSeedReplayError):
        bind_a6_replay(a6_request(seed=S5[0]), a1_artifact(seed=S5[1]))
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/events/test_event_sequence.py tests/events/test_variant_event_boundaries.py -v`

Expected: FAIL because Event modules are absent.

- [ ] **Step 3: Implement deterministic event graph**

Require event source kind, Sensor bundle identity/hash where applicable, pseudo-label bundle identity, symbolizer rule/version, ordered segment boundaries, sample/TimeGrid/mask, source evidence and checksum. Serialize NFC/UTF-8/LF with fixed punctuation/spacing/trailing-newline rules. Enforce A2 none, A5 local shadow-only, A6 external replay model-visible with no local Sensor, A8 child-scoped and A10 local Event-conditioned VA.

- [ ] **Step 4: Run GREEN**

Run: `python -m pytest tests/events -v`

Expected: PASS for mutation, wrong-source, wrong-seed, visibility and canonical serialization tests.

- [ ] **Step 5: Future commit boundary**

```powershell
git add src/sc_mv_dmer/events schemas/event_sequence.schema.json schemas/event_replay_bundle.schema.json tests/events
git commit -m "feat: bind event provenance and variant boundaries"
```

### Task 2: 分区并物化 early/late CoT/ANSWER artifacts

**Files:**
- Create: `src/sc_mv_dmer/cot/partition.py`
- Create: `src/sc_mv_dmer/cot/materialize.py`
- Create: `src/sc_mv_dmer/cot/filters.py`
- Create: `schemas/cot_partition_manifest.schema.json`
- Create: `schemas/cot_artifact_manifest.schema.json`
- Create: `tests/cot/test_partition.py`
- Create: `tests/cot/test_materialization.py`
- Generate during execution: `manifests/cot/<variant>/<phase>/cot-artifact-v1.json`

**Interfaces:**
- Consumes: StageTrainingPopulationManifest, early/late EventSequence, target profile and approved teacher provenance.
- Produces: `partition_cot_coverage(...) -> CoTPartitionManifest`; `materialize_cot(...) -> CoTArtifactManifest`.

- [ ] **Step 1: Write partition and phase-integrity tests**

```python
def test_cot_partition_rounds_two_thirds_and_is_seed_independent() -> None:
    a = partition_cot_coverage(train_ids(10), coverage_rule(), seed=S5[0])
    b = partition_cot_coverage(train_ids(10), coverage_rule(), seed=S5[4])
    assert len(a.synthetic_ids) == 7
    assert a.membership_hash == b.membership_hash

def test_late_artifact_requires_stage1_sensor_snapshot() -> None:
    with pytest.raises(ArtifactIntegrityError):
        materialize_cot(partition_fixture(), source_bundle(phase="S2_LATE", sensor_snapshot=None))
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/cot/test_partition.py tests/cot/test_materialization.py -v`

Expected: FAIL because CoT modules are absent.

- [ ] **Step 3: Implement deterministic coverage and immutable materialization**

Rank songs by the frozen domain-separated SHA-256 inputs without reading targets/loss/difficulty/predictions. Generate phase/variant-matched records with Event provenance, raw CoT, raw ANSWER, parsed diagnostics, target profile, teacher identity, prompt/tokenizer/schema versions and checksums. Automatic filters retain failures; final artifact is atomic and cannot be overwritten.

- [ ] **Step 4: Run GREEN and materialize only after exact H3 inputs exist**

Run: `python -m pytest tests/cot -v`

Future command: `python -m sc_mv_dmer.cli materialize-cot --variant A1_FULL_7B --handoff manifests/handoffs --phases S2_EARLY,S2_LATE --output-root manifests/cot/A1_FULL_7B`

Expected: tests PASS; final manifest only when every synthetic song has both required phase records.

- [ ] **Step 5: Future commit boundary**

```powershell
git add src/sc_mv_dmer/cot schemas/cot_partition_manifest.schema.json schemas/cot_artifact_manifest.schema.json tests/cot
git commit -m "feat: materialize phase-bound cot artifacts"
```

### Task 3: 执行 RG-07 token Gate 与 RG-09 人工审核

**Files:**
- Create: `src/sc_mv_dmer/gates/rg07.py`
- Create: `src/sc_mv_dmer/gates/rg09.py`
- Create: `src/sc_mv_dmer/review/human_package.py`
- Create: `tests/gates/test_rg07.py`
- Create: `tests/gates/test_rg09.py`
- Create: `tests/review/test_human_review_package.py`
- Generate during execution: `evidence/gates/rg-07/<artifact-id>.json`
- Generate during execution: `evidence/gates/rg-09/review-package-v1/`
- Generate during execution: `evidence/gates/rg-09/rg-09-attempt-v1.json`

**Interfaces:**
- Consumes: immutable Event/CoT artifact and exact tokenizer.
- Produces: `evaluate_rg07(...)`; `select_rg09_records(...)`; `build_review_package(...)`; `finalize_rg09(...)`.

- [ ] **Step 1: Write max-token and human-authority tests**

```python
def test_rg07_fails_one_7b_record_at_181_tokens() -> None:
    assert evaluate_rg07(event_population(max_tokens=181), qwen7b_tokenizer()).historical_verdict == "FAIL"

def test_rg09_requires_exactly_100_unique_synthetic_records() -> None:
    sample = select_rg09_records(cot_population(150), sampling_rule())
    assert len(sample.record_ids) == len(set(sample.record_ids)) == 100

def test_rg09_cannot_be_finalized_by_model_reviews() -> None:
    with pytest.raises(HumanAuthorityRequired):
        finalize_rg09(sample_fixture(), model_generated_reviews())
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/gates/test_rg07.py tests/gates/test_rg09.py tests/review/test_human_review_package.py -v`

Expected: FAIL because Gate/review modules are absent.

- [ ] **Step 3: Implement all-record RG-07 and blinded RG-09 package**

RG-07 counts exact serialized blocks with no special tokens/truncation and stores max/violations plus source/version identities; every configured early/late source must PASS for stage entry. RG-09 is evaluated separately for every artifact/profile identity required by formal training; it freezes sampling frame/seed/IDs/order before review, excludes test/native records, imports two independent blinded real-human C1–C5 records, and requires real-human adjudication on critical disagreement. PASS is exact `qualified_count>=85` after all 100 adjudicated outcomes.

- [ ] **Step 4: Run tests, then build packages**

Run: `python -m pytest tests/gates/test_rg07.py tests/gates/test_rg09.py tests/review -v`

Run: `python -m sc_mv_dmer.cli build-rg09-review --cot-root manifests/cot --scope all-formal-required --output-root evidence/gates/rg-09`

Expected: tests PASS; Gate remains `PENDING_MANUAL_REVIEW` until genuine reviewer/adjudicator records are imported.

- [ ] **Step 5: Future commit boundary**

```powershell
git add src/sc_mv_dmer/gates/rg07.py src/sc_mv_dmer/gates/rg09.py src/sc_mv_dmer/review tests/gates/test_rg07.py tests/gates/test_rg09.py tests/review
git commit -m "test: qualify event length and cot quality"
```

### Task 4: 实现正式 Stage 2、PB-1+、CS-3+ 与 final inference

**Files:**
- Create: `src/sc_mv_dmer/training/stage2.py`
- Create: `src/sc_mv_dmer/training/event_curriculum.py`
- Create: `src/sc_mv_dmer/training/checkpoint_selection.py`
- Create: `src/sc_mv_dmer/inference/final_events.py`
- Create: `src/sc_mv_dmer/inference/predict.py`
- Create: `configs/training/stage2-a1-v1.yaml`
- Create: `tests/training/test_stage2_curriculum.py`
- Create: `tests/training/test_stage2_loss_profile.py`
- Create: `tests/inference/test_final_event_source.py`

**Interfaces:**
- Consumes: H3 package, approved early/late CoT artifacts and Stage-2 capability.
- Produces: `run_stage2(spec: Stage2RunSpec) -> Stage2TerminalBundle`; `run_final_inference(...) -> PredictionBundle`.

- [ ] **Step 1: Write phase, loss and terminal-selection tests**

```python
def test_phase_switch_occurs_only_after_epoch_three_commit() -> None:
    state = curriculum_fixture()
    assert state.event_source_at(epoch=3) == "pseudo_label"
    assert state.event_source_at(epoch=4) == "stage1_sensor_snapshot"

def test_a1_stage2_loss_active_set_is_constant_across_phases() -> None:
    early = compile_stage2_loss(a1_profile(), "S2_EARLY")
    late = compile_stage2_loss(a1_profile(), "S2_LATE")
    assert early.term_ids == late.term_ids
    assert early.kendall_groups == late.kendall_groups == {"VA","CoT","consist"}

def test_only_epoch_five_terminal_is_designated() -> None:
    assert designate_checkpoint(checkpoint_curve(epochs=5)).logical_epoch == 5
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/training/test_stage2_curriculum.py tests/training/test_stage2_loss_profile.py tests/inference/test_final_event_source.py -v`

Expected: FAIL because Stage-2/inference modules are absent.

- [ ] **Step 3: Implement immutable new-run Stage 2**

Validate artifact allowlist before run_spec creation. Rebuild optimizer/scheduler and create Stage-2 RNG namespace. Use exact A1-like objective `sum_{i in {VA,CoT,consist}}[0.5*exp(-s_i)*L_i+0.5*s_i]+0.05*L_smooth+0.01*L_state+0.10*L_sensor` plus applicable A11/A12 fixed terms. PB switch changes only Event/CoT identity; it does not reset state. Final inference materializes terminal own-Sensor events per variant disposition and never falls back to No-LLM.

- [ ] **Step 4: Run GREEN; then authorized local Stage 2**

Run: `python -m pytest tests/training/test_stage2_curriculum.py tests/training/test_stage2_loss_profile.py tests/inference/test_final_event_source.py -v`

Future command: `python -m sc_mv_dmer.cli train --variant A1_FULL_7B --stage stage2 --handoff manifests/handoffs --cot-root manifests/cot/A1_FULL_7B --seed-set SC-MV-DMER-FORMAL-S5/v1 --mode formal`

Expected: unit PASS; every Stage-2 run is a new immutable run and only epoch-5 terminal is designated.

- [ ] **Step 5: Future commit boundary**

```powershell
git add src/sc_mv_dmer/training/stage2.py src/sc_mv_dmer/training/event_curriculum.py src/sc_mv_dmer/training/checkpoint_selection.py src/sc_mv_dmer/inference configs/training/stage2-a1-v1.yaml tests/training tests/inference
git commit -m "feat: add formal stage2 curriculum and inference"
```

### Task 5: 完成 RG-08 artifact-dependent roles 与 RG-10-v2

**Files:**
- Modify: `src/sc_mv_dmer/gates/rg08.py`
- Create: `src/sc_mv_dmer/gates/rg10.py`
- Create: `tests/gates/test_rg08_stage2.py`
- Create: `tests/gates/test_rg10.py`
- Generate during execution: `evidence/gates/rg-08/<hardware-profile>/`
- Generate during execution: `evidence/gates/rg-10/rg-10-attempt-v2.json`

**Interfaces:**
- Consumes: exact approved Stage-2 artifacts/topology, terminal checkpoint and full validation generation population.
- Produces: remaining RG-08 role evidence and `evaluate_rg10(...) -> GateEvaluationAttempt`.

- [ ] **Step 1: Write P3/P4 and exact-rate tests**

```python
def test_rg08_p3_runs_two_complete_accumulation_cycles() -> None:
    evidence = probe_p3(stage2_probe_fixture())
    assert evidence.complete_accumulation_cycles == 2
    assert evidence.optimizer_lazy_init_observed

def test_rg10_exact_fraction_boundary() -> None:
    assert evaluate_rg10(parser_population(success=98, eligible=100)).historical_verdict == "PASS"
    assert evaluate_rg10(parser_population(success=97, eligible=100)).historical_verdict == "FAIL"
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/gates/test_rg08_stage2.py tests/gates/test_rg10.py -v`

Expected: FAIL because Stage-2 probes/RG-10 evaluator are incomplete.

- [ ] **Step 3: Implement exact role and population execution**

P3 runs two complete accumulation cycles on maximum formal input and actual artifact/topology; P4 runs exact prompt/Event/90-position generation where applicable. Measure fresh-process PyTorch/NVML peaks. RG-10 evaluates every completed primary-validation raw generation, includes empty/malformed/truncated outputs in denominator, forbids retry/repair, and uses exact integer comparison `100*N_success >= 98*N_eligible`.

- [ ] **Step 4: Run tests and local-first probes**

Run: `python -m pytest tests/gates/test_rg08_stage2.py tests/gates/test_rg10.py -v`

Future command: `python -m sc_mv_dmer.cli probe-rg08 --variant A1_FULL_7B --roles P3_STAGE2,P4_GENERATION --hardware-profile local-5060ti --output evidence/gates/rg-08/local-5060ti`

Expected: tests PASS; local OOM creates `BLOCKED_LOCAL_COMPUTE` and external handoff, not a fabricated canonical verdict.

- [ ] **Step 5: Future commit boundary**

```powershell
git add src/sc_mv_dmer/gates/rg08.py src/sc_mv_dmer/gates/rg10.py tests/gates/test_rg08_stage2.py tests/gates/test_rg10.py
git commit -m "test: qualify stage2 memory and parser stability"
```

### Task 6: 编译并执行 A1–A13 closed-world matrix

**Files:**
- Create: `src/sc_mv_dmer/experiments/catalog.py`
- Create: `src/sc_mv_dmer/experiments/compiler.py`
- Create: `src/sc_mv_dmer/experiments/orchestrator.py`
- Create: `configs/experiments/A1-v1.yaml`
- Create: `configs/experiments/A2-v1.yaml`
- Create: `configs/experiments/A3-v1.yaml`
- Create: `configs/experiments/A4-v1.yaml`
- Create: `configs/experiments/A5-v1.yaml`
- Create: `configs/experiments/A6-v1.yaml`
- Create: `configs/experiments/A7-v1.yaml`
- Create: `configs/experiments/A8-mel-v1.yaml`
- Create: `configs/experiments/A8-mfcc-v1.yaml`
- Create: `configs/experiments/A8-chroma-v1.yaml`
- Create: `configs/experiments/A9-v1.yaml`
- Create: `configs/experiments/A10-v1.yaml`
- Create: `configs/experiments/A11-v1.yaml`
- Create: `configs/experiments/A12-v1.yaml`
- Create: `configs/experiments/A13-v1.yaml`
- Create: `tests/experiments/test_variant_compiler.py`
- Create: `tests/experiments/test_variant_dependencies.py`

**Interfaces:**
- Consumes: canonical A1 base, versioned semantic overlay and dependency registry.
- Produces: `compile_experiment(variant_id, context) -> CompiledExperiment`; `run_experiment_matrix(...) -> ExperimentEvidenceMatrix`.

- [ ] **Step 1: Write closed-world and special-boundary tests**

```python
def test_catalog_has_exactly_fifteen_executable_variants() -> None:
    assert len(load_catalog().executable_variants) == 15

def test_special_boundaries_are_not_collapsed() -> None:
    assert compile_experiment("A5-v1").event_visibility == "SHADOW_ONLY"
    assert compile_experiment("A6-v1").event_source == "PAIRED_A1_REPLAY"
    assert compile_experiment("A10-v1").generation == "NOT_APPLICABLE"
    assert compile_experiment("A13-v1").base_experiment_variant_id is None
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/experiments/test_variant_compiler.py tests/experiments/test_variant_dependencies.py -v`

Expected: FAIL because catalog/compiler are absent.

- [ ] **Step 3: Implement semantic-diff and dependency closure compiler**

Reject undeclared changes and hidden flag bundles. Compile A8 three children independently, A12 only when preregistered trigger evidence authorizes it, and A13 as its own five-epoch No-LLM baseline. Require variant-specific Stage-1 snapshot/CoT/Gate attempts where identity differs; never borrow A1 evidence solely because topology is related.

- [ ] **Step 4: Run GREEN and execute in dependency/resource order**

Run: `python -m pytest tests/experiments -v`

Future command: `python -m sc_mv_dmer.cli run-matrix --catalog configs/experiments --scope primary --seed-set SC-MV-DMER-FORMAL-S5/v1 --mode formal`

Expected: compiler PASS; orchestrator blocks ineligible/stale variants and preserves one run_id per stage/seed/variant.

- [ ] **Step 5: Future commit boundary**

```powershell
git add src/sc_mv_dmer/experiments configs/experiments tests/experiments
git commit -m "feat: compile primary ablation matrix"
```

### Task 7: 实现 long58 与 PMEmo evaluation runs

**Files:**
- Create: `src/sc_mv_dmer/evaluation/long58.py`
- Create: `src/sc_mv_dmer/evaluation/pmemo.py`
- Create: `src/sc_mv_dmer/data/pmemo.py`
- Create: `configs/datasets/pmemo-2019.yaml`
- Create: `tests/evaluation/test_long58.py`
- Create: `tests/evaluation/test_pmemo.py`
- Generate during execution: `manifests/datasets/pmemo-2019-v1.json`
- Generate during execution: `evidence/data/pmemo-artifact-domain-audit-v1.json`

**Interfaces:**
- Consumes: A1 terminal Stage-2 checkpoints, evaluation data manifests and frozen window/mapping contracts.
- Produces: independent `EvaluationRunManifest` values for long58 and PMEmo A1-S5 zero-shot.

- [ ] **Step 1: Write no-training and timeline tests**

```python
def test_pmemo_is_zero_shot_a1_s5_only() -> None:
    run = compile_pmemo_evaluation(a1_s5_fixture())
    assert run.trainable_parameters == 0
    assert run.source_checkpoint_count == 5

def test_long58_windows_do_not_carry_hidden_state() -> None:
    result = evaluate_long58(long_song_fixture())
    assert result.hidden_state_carry_between_windows is False
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/evaluation/test_long58.py tests/evaluation/test_pmemo.py -v`

Expected: FAIL because evaluators are absent.

- [ ] **Step 3: Implement source audit and independent evaluation runs**

Discover/hash PMEmo 2019 artifact, audit label domain, chorus-clip-local `t=0`, first-15-second mapping, mask and sample identity before evaluation. long58 uses the frozen sliding-window contract and explicit aggregation without cross-window hidden state. Neither dataset trains/tunes/selects checkpoints; every evaluation run references `source_run_id` and is terminal/immutable.

- [ ] **Step 4: Run GREEN and source discovery**

Run: `python -m pytest tests/evaluation -v`

Run: `python -m sc_mv_dmer.cli discover-data --dataset pmemo-2019 --data-root $env:SC_MV_DMER_DATA_ROOT --output manifests/datasets/pmemo-2019-v1.json`

Expected: tests PASS; real audit either binds exact artifact/domain or closes with `PMEMO_ARTIFACT_HASH_DOMAIN_AUDIT` non-PASS evidence.

- [ ] **Step 5: Future commit boundary**

```powershell
git add src/sc_mv_dmer/evaluation src/sc_mv_dmer/data/pmemo.py configs/datasets/pmemo-2019.yaml tests/evaluation
git commit -m "feat: add locked external evaluations"
```

### Task 8: 聚合 COMPLETE_S5、effective validity 与 paper artifacts

**Files:**
- Create: `src/sc_mv_dmer/evaluation/aggregate.py`
- Create: `src/sc_mv_dmer/evaluation/reporting.py`
- Create: `schemas/experiment_evidence_matrix.schema.json`
- Create: `schemas/paper_aggregation_manifest.schema.json`
- Create: `tests/evaluation/test_s5_aggregation.py`
- Create: `tests/evaluation/test_effective_aggregation.py`
- Generate during execution: `evidence/experiments/experiment-evidence-matrix-v1.json`
- Generate during execution: `reports/paper/paper-aggregation-manifest-v1.json`
- Generate during execution: `evidence/qualifications/step-6-final.json`

**Interfaces:**
- Consumes: immutable run/Gate/evaluation/artifact registry and invalidation graph.
- Produces: `aggregate_experiments(...) -> ExperimentEvidenceMatrix`; `build_paper_manifest(...) -> PaperAggregationManifest`.

- [ ] **Step 1: Write S5 and effective-validity exclusion tests**

```python
def test_incomplete_s5_is_not_paper_eligible() -> None:
    result = aggregate_experiments(evidence_with_seeds(S5[:4]))
    assert result.paper_eligible is False
    assert result.reason_code == "INCOMPLETE_S5"

def test_historical_pass_with_invalidated_predecessor_is_excluded() -> None:
    result = aggregate_experiments(stale_dependency_fixture())
    assert result.excluded_rows[0].effective_status == "STALE_DEPENDENCY"
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/evaluation/test_s5_aggregation.py tests/evaluation/test_effective_aggregation.py -v`

Expected: FAIL because aggregator is absent.

- [ ] **Step 3: Implement exact aggregation and claim firewall**

Compute per-seed V/A CCC and MSE from pooled valid-frame FP64 moments/sums, S5 arithmetic mean and sample std `ddof=1`; paired deltas use identical seeds. Keep `Y_va` primary and report `Y_answer`, parser, consistency, Event, Sensor, Markov, beta, gradient and stress diagnostics separately. Exclude invalidated/stale/incomplete evidence and report propagation source/reason/replacement. Do not claim human-validated final explanations from RG-09 synthetic-artifact review.

- [ ] **Step 4: Run final verification and dry-run aggregation**

Run: `python -m pytest tests/evaluation tests/experiments tests/gates -v`

Run: `python -m sc_mv_dmer.cli aggregate --profile paper --require-complete-s5 --dry-run`

Run: `python -m sc_mv_dmer.cli qualify-step --step 6 --mode formal --output evidence/qualifications/step-6-final.json`

Expected: paper eligibility only when every required variant/seed/Gate/evaluation dependency is current-effective and exact S5 is complete. Missing canonical hardware or human review remains an explicit blocker.

- [ ] **Step 5: Future commit boundary**

```powershell
git add src/sc_mv_dmer/evaluation schemas tests/evaluation evidence/experiments reports/paper evidence/qualifications/step-6-final.json
git commit -m "feat: aggregate current-effective experiment evidence"
```

## 14B 非阻塞扩展边界

14B 只能在 7B primary evidence 完成且资源/独立 catalog/Gate applicability 已批准后，以新 semantic identity、hardware profile、CoT artifact 和 run graph 执行。不得复用 7B RG-07/08/09 PASS，也不得使 7B Step-6 qualification 等待 14B。
