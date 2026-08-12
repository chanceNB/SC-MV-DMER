# Step 5 — Qwen 7B 接入、Stage 1 与 H3 Handoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 No-LLM Gate PASS 后 pin Qwen2.5-7B/tokenizer，验证 Full LLM 双头与训练数值合同，执行正式 Stage 1，并生成 immutable H3 handoff 和 Stage-2 entry qualification；不在 CoT artifact 获批前启动 formal Stage 2。

**Architecture:** Upstream manifest 和 capability compiler 分别生成 Stage-1、Stage-2 QLoRA、A10 load specs；Full model 显式输出 hidden-state `Y_va` 与 LM-generated CoT/ANSWER/`Y_answer`。Training runtime 统一拥有 S5 RNG、exact-once sampler、CRR、durable state、loss reduction 和 optimizer；Stage 1 terminal state通过 H3 package交给新 Stage-2 run。

**Tech Stack:** Python、PyTorch、Transformers、PEFT、bitsandbytes、safetensors、NVML binding、Pydantic、pytest。

## Global Constraints

- Primary model 是 Qwen2.5-7B；14B/DPO/Qwen2.5-Audio deferred。
- 不自动下载 upstream；model/tokenizer/config/chat-template/special-token bytes 与 revision 全部 pin。
- Stage 1 不创建 LoRA；Stage 2 使用 rank 16、alpha 32、LoRA dropout 0、NF4 QLoRA。
- Full LLM 不 ensemble No-LLM head；双头必须保留原始输出、parsed result 和 consistency evidence。
- A10 是 Event-conditioned VA-only hidden-backbone model，无 LM generation/parser/CoT/ANSWER。
- Stage 1→2 新建 run_id；H3 只转移 typed model state，optimizer 重建、scheduler 重启、Stage-2 RNG 新 namespace。
- Qualification fixture 不能进入 formal Gate/paper evidence；RG-10-v2 formal attempt 只能由 Step 6 完整 generation population 形成。
- Step 5 只关闭无正式 CoT population 时合法的 RG-08 roles；P3/P4 等标记 `PENDING_STEP_6_ARTIFACT`。

---

### Task 1: Pin Qwen upstream、tokenizer 与 ANSWER budget

**Files:**
- Create: `src/sc_mv_dmer/models/qwen_upstream.py`
- Create: `src/sc_mv_dmer/generation/token_budget.py`
- Create: `configs/upstream/qwen-7b-primary.yaml`
- Create: `schemas/qwen_upstream_manifest.schema.json`
- Create: `tests/models/test_qwen_upstream.py`
- Create: `tests/generation/test_token_budget.py`
- Generate during execution: `manifests/upstream/qwen-7b-primary-v1.json`
- Generate during execution: `evidence/qualifications/tokenizer-answer-budget-v1.json`

**Interfaces:**
- Produces: `bind_qwen(inventory: UpstreamInventory) -> UpstreamModelManifest`; `qualify_answer_budget(binding: TokenizerBinding, schema: AnswerSchema) -> TokenBudgetEvidence`.

- [ ] **Step 1: Write identity and exact-count tests**

```python
def test_qwen_manifest_binds_tokenizer_and_chat_template() -> None:
    manifest = bind_qwen(qwen_inventory_fixture())
    assert manifest.tokenizer_sha256
    assert manifest.chat_template_sha256
    assert manifest.special_token_map_sha256

def test_answer_budget_encodes_exact_90_by_2_values() -> None:
    evidence = qualify_answer_budget(tokenizer_fixture(), answer_schema_v2())
    assert evidence.time_positions == 90
    assert evidence.values_per_position == 2
    assert evidence.max_new_tokens == evidence.b_think + evidence.b_answer + evidence.b_wrapper
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/models/test_qwen_upstream.py tests/generation/test_token_budget.py -v`

Expected: FAIL because upstream/budget modules are absent.

- [ ] **Step 3: Implement read-only binding and GEN-DTB-VA90-3R++ calculation**

Inventory weight shards, config, generation config, tokenizer files, chat template, special-token map, license and library compatibility. Compute `B_answer/B_wrapper` with the exact pinned tokenizer; `B_think=300` applies only to THINK payload. Save canonical 90-position signed-four-decimal ANSWER fixtures and reject total-budget assumptions based on 300 alone.

- [ ] **Step 4: Run GREEN and real inventory**

Run: `python -m pytest tests/models/test_qwen_upstream.py tests/generation/test_token_budget.py -v`

Run: `python -m sc_mv_dmer.cli discover-upstream --kind qwen-7b --root $env:SC_MV_DMER_MODEL_ROOT --output manifests/upstream/qwen-7b-primary-v1.json`

Expected: tests PASS; real inventory is checksum-complete or emits `PINNED_QWEN_UPSTREAM_MANIFEST` blocker.

- [ ] **Step 5: Future commit boundary**

```powershell
git add src/sc_mv_dmer/models/qwen_upstream.py src/sc_mv_dmer/generation configs/upstream/qwen-7b-primary.yaml schemas/qwen_upstream_manifest.schema.json tests/models/test_qwen_upstream.py tests/generation
git commit -m "feat: pin qwen and tokenizer budget"
```

### Task 2: 编译并验证 Stage-1、Stage-2、A10 loader topology

**Files:**
- Create: `src/sc_mv_dmer/models/qwen_loader.py`
- Create: `src/sc_mv_dmer/models/qlora.py`
- Create: `src/sc_mv_dmer/models/a10_loader.py`
- Create: `tests/models/test_qwen_loader.py`
- Create: `tests/models/test_a10_loader.py`

**Interfaces:**
- Consumes: `CompiledExecutionCapabilityManifest`, upstream manifest.
- Produces: `compile_qwen_load_spec(profile, stage) -> QwenLoadSpec`; `load_qwen(spec, upstream) -> LoadedModelBundle`; `qualify_a10_equivalence(...) -> ProjectionEquivalenceEvidence`.

- [ ] **Step 1: Write topology tests**

```python
def test_stage1_has_no_lora_parameters() -> None:
    bundle = load_qwen(stage1_spec(), upstream_fixture())
    assert bundle.lora_parameter_count == 0

def test_stage2_uses_exact_frozen_qlora_contract() -> None:
    spec = compile_qwen_load_spec(a1_profile(), Stage.STAGE2)
    assert (spec.rank, spec.alpha, spec.dropout, spec.quantization) == (16, 32, 0.0, "NF4")

def test_a10_has_no_lm_generation_path() -> None:
    assert load_a10(a10_spec(), upstream_fixture()).generation_capability is False
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/models/test_qwen_loader.py tests/models/test_a10_loader.py -v`

Expected: FAIL because loader modules are absent.

- [ ] **Step 3: Implement snapshot-pinned k-bit preparation**

Compile allowed target modules and dtype/device/use_cache/checkpointing from capability + stage. Assert actual loaded graph, trainable parameter set, weight tying and quantization metadata against spec. A10 loads the shared hidden backbone directly and must satisfy the frozen projection/equivalence criterion without constructing LM output roles.

- [ ] **Step 4: Run GREEN and executable loader probe**

Run: `python -m pytest tests/models/test_qwen_loader.py tests/models/test_a10_loader.py -v`

Run: `python -m sc_mv_dmer.cli qualify-qwen-loader --upstream manifests/upstream/qwen-7b-primary-v1.json --roles stage1,stage2,a10 --output evidence/qualifications/qwen-loader-v1.json`

Expected: unit PASS; actual loader records exact topology or immutable failure evidence.

- [ ] **Step 5: Future commit boundary**

```powershell
git add src/sc_mv_dmer/models/qwen_loader.py src/sc_mv_dmer/models/qlora.py src/sc_mv_dmer/models/a10_loader.py tests/models
git commit -m "feat: compile qwen stage loaders"
```

### Task 3: 实现 ChatTS adapter、Full LLM 双头、ANSWER parser 与 PredictionBundle

**Files:**
- Create: `src/sc_mv_dmer/models/chatts_adapter.py`
- Create: `src/sc_mv_dmer/models/full_llm.py`
- Create: `src/sc_mv_dmer/generation/answer_parser.py`
- Create: `src/sc_mv_dmer/predictions.py`
- Create: `schemas/prediction_bundle.schema.json`
- Create: `tests/models/test_full_llm_heads.py`
- Create: `tests/generation/test_answer_parser.py`

**Interfaces:**
- Consumes: `F_fused`, Event-conditioned prompt, loaded Qwen and masks.
- Produces: `FullLLMModel.forward(...) -> ForwardPrediction`; `parse_answer(raw: str, expected: TimeGridBinding) -> ParseResult`; `PredictionBundle`.

- [ ] **Step 1: Write dual-head and full-grid parser tests**

```python
def test_full_llm_exposes_two_distinct_outputs() -> None:
    out = full_llm_fixture()(batch_fixture())
    assert out.y_va.shape[-2:] == (90, 2)
    assert out.raw_lm_output is not None
    assert out.no_llm_output is None

def test_parser_requires_exact_full_timegrid() -> None:
    result = parse_answer(answer_with_pairs(89), timegrid_90())
    assert result.success is False
    assert "WRONG_PAIR_COUNT" in result.reason_codes
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/models/test_full_llm_heads.py tests/generation/test_answer_parser.py -v`

Expected: FAIL because adapter/model/parser are absent.

- [ ] **Step 3: Implement explicit two-path output contract**

Map `[B,90,256]` to registered time-series tokens. Hidden states feed the tanh VA regression head; LM head generates raw THINK/ANSWER text; deterministic RG-10-v2 parser checks one legal ANSWER, exactly 90 V and 90 A finite in-domain values, position/order/binding and no repair/clipping/retry. PredictionBundle stores raw/parsed outputs, failure reasons, masks and raw consistency evidence.

- [ ] **Step 4: Run GREEN**

Run: `python -m pytest tests/models/test_full_llm_heads.py tests/generation/test_answer_parser.py -v`

Expected: PASS for valid, empty, malformed, truncated, out-of-range, repeated-value-valid and duplicated-position-invalid fixtures.

- [ ] **Step 5: Future commit boundary**

```powershell
git add src/sc_mv_dmer/models/chatts_adapter.py src/sc_mv_dmer/models/full_llm.py src/sc_mv_dmer/generation/answer_parser.py src/sc_mv_dmer/predictions.py schemas/prediction_bundle.schema.json tests/models/test_full_llm_heads.py tests/generation/test_answer_parser.py
git commit -m "feat: add full llm dual-head prediction contract"
```

### Task 4: 实现 S5 RNG、CRR、TrainingState、loss reduction 与 numeric policy

**Files:**
- Create: `src/sc_mv_dmer/training/seeds.py`
- Create: `src/sc_mv_dmer/training/rng.py`
- Create: `src/sc_mv_dmer/training/sampler.py`
- Create: `src/sc_mv_dmer/training/replay.py`
- Create: `src/sc_mv_dmer/training/state.py`
- Create: `src/sc_mv_dmer/training/reduction.py`
- Create: `src/sc_mv_dmer/training/optimizer.py`
- Create: `src/sc_mv_dmer/training/numerics.py`
- Create: `tests/training/test_rng_replay.py`
- Create: `tests/training/test_training_state.py`
- Create: `tests/training/test_loss_reduction.py`
- Create: `tests/training/test_optimizer_numerics.py`

**Interfaces:**
- Produces: `derive_s5(...)`; `checkpoint_replay(...)`; `capture_training_state(...)`; `reduce_loss_terms(...)`; `build_optimizer(...)`.

- [ ] **Step 1: Write deterministic replay and reduction tests**

```python
def test_frozen_s5_identity() -> None:
    assert derive_s5("SC-MV-DMER-FORMAL-S5", "v1") == (52826381,128866372,1616929435,1871035633,1460830465)

def test_midstep_restart_replays_from_last_committed_logical_step() -> None:
    assert replay_fixture().sample_order_after_restart == replay_fixture().uninterrupted_sample_order

def test_microbatch_accumulation_matches_full_logical_batch() -> None:
    assert reduce_fixture(microbatches=4) == pytest.approx(reduce_fixture(microbatches=1))
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/training/test_rng_replay.py tests/training/test_training_state.py tests/training/test_loss_reduction.py tests/training/test_optimizer_numerics.py -v`

Expected: FAIL because runtime infrastructure is absent.

- [ ] **Step 3: Implement frozen execution contracts**

Use namespaced RNG domains, global-permutation exact-once epochs, invocation-local CRR capsules and durable logical-step TrainingState. Loss leaves emit numerator/denominator/eligibility; zero eligibility produces `grad=None` for conditional parameters. Compile parameter/mode/optimizer groups, AdamW schedule, clipping, nonfinite handling, BF16 execution and FP32 islands exactly from OSNP/OSNS/HLAC authorities.

- [ ] **Step 4: Run GREEN and qualification probes**

Run: `python -m pytest tests/training -v`

Expected: PASS for replay equivalence, sampler exact-once, denominator invariance, optimizer membership and nonfinite policy.

- [ ] **Step 5: Future commit boundary**

```powershell
git add src/sc_mv_dmer/training tests/training
git commit -m "feat: qualify deterministic training runtime"
```

### Task 5: 实现并执行正式 Stage 1

**Files:**
- Create: `src/sc_mv_dmer/training/stage1.py`
- Create: `src/sc_mv_dmer/training/parameter_roles.py`
- Create: `configs/training/stage1-a1-v1.yaml`
- Create: `tests/training/test_stage1_contract.py`
- Generate during execution: `runs/<run_id>/run_spec.json`
- Generate during execution: `runs/<run_id>/run_manifest.json`
- Generate during execution: `runs/<run_id>/artifacts/stage1-sensor-snapshot.safetensors`

**Interfaces:**
- Consumes: Full model, A1 capability, primary train population and S5 seed.
- Produces: `run_stage1(spec: Stage1RunSpec) -> Stage1TerminalBundle`.

- [ ] **Step 1: Write Stage-1 active-set and parameter-role tests**

```python
def test_stage1_has_no_lora_or_kendall_parameters() -> None:
    compiled = compile_stage1(a1_profile())
    assert compiled.lora_parameters == ()
    assert compiled.kendall_parameters == ()
    assert compiled.loss_weights == {"VA":1.0,"state":0.01,"sensor":0.10}
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/training/test_stage1_contract.py -v`

Expected: FAIL because Stage-1 runner is absent.

- [ ] **Step 3: Implement immutable Stage-1 TrainingStageRun**

Create one new run_id per S5 seed. Compile module modes/gradients/losses, rebuild nothing mid-run, save validation curves, beta/Markov/Sensor diagnostics, checkpoints, durable TrainingState and checksums. Terminal output includes the exact Stage-1 Sensor snapshot needed for late Event/CoT materialization.

- [ ] **Step 4: Run unit qualification, then formal local Stage 1 only when authorized**

Run: `python -m pytest tests/training/test_stage1_contract.py tests/training -v`

Future formal command: `python -m sc_mv_dmer.cli train --variant A1_FULL_7B --stage stage1 --seed-set SC-MV-DMER-FORMAL-S5/v1 --mode formal`

Expected: tests PASS; each authorized run terminates immutably with a Sensor snapshot. Test data is not loaded.

- [ ] **Step 5: Future commit boundary**

```powershell
git add src/sc_mv_dmer/training/stage1.py src/sc_mv_dmer/training/parameter_roles.py configs/training/stage1-a1-v1.yaml tests/training/test_stage1_contract.py
git commit -m "feat: add formal stage1 runner"
```

### Task 6: 实现 H3 handoff 与 Stage-2 entry compiler

**Files:**
- Create: `src/sc_mv_dmer/training/handoff.py`
- Create: `src/sc_mv_dmer/training/stage2_entry.py`
- Create: `schemas/h3_stage_handoff.schema.json`
- Create: `schemas/stage2_entry_qualification.schema.json`
- Create: `tests/training/test_h3_handoff.py`
- Create: `tests/training/test_stage2_entry.py`
- Generate during execution: `manifests/handoffs/<stage1_run_id>-to-stage2.json`

**Interfaces:**
- Consumes: `Stage1TerminalBundle`, Stage-2 capability and required artifact refs.
- Produces: `build_h3_handoff(...) -> H3StageHandoffPackage`; `qualify_stage2_entry(...) -> Stage2EntryQualificationBundle`.

- [ ] **Step 1: Write new-run and artifact-blocking tests**

```python
def test_handoff_does_not_transfer_optimizer_or_scheduler() -> None:
    handoff = build_h3_handoff(stage1_terminal_fixture(), stage2_expectation_fixture())
    assert handoff.optimizer_state is None
    assert handoff.scheduler_state is None

def test_stage2_entry_blocks_without_phase_matched_cot() -> None:
    result = qualify_stage2_entry(entry_fixture(cot_artifacts=[]))
    assert result.allowed is False
    assert result.reason_code == "STAGE2_BLOCKED_BY_COT_ARTIFACT"
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/training/test_h3_handoff.py tests/training/test_stage2_entry.py -v`

Expected: FAIL because handoff/entry modules are absent.

- [ ] **Step 3: Implement typed immutable boundary**

Bind source terminal run/checkpoint, semantic identity, transferable model tensors, Sensor snapshot, target compiled topology, target artifact requirements, same S5 lineage and checksums. Stage 2 must create a new run_id, optimizer, scheduler and RNG namespace. Entry accepts only current-effective exact phase/variant artifacts; qualification fixtures are explicitly forbidden for formal entry.

- [ ] **Step 4: Run GREEN**

Run: `python -m pytest tests/training/test_h3_handoff.py tests/training/test_stage2_entry.py -v`

Expected: PASS for legal transfer, cross-seed rejection, terminal immutability and missing-CoT block.

- [ ] **Step 5: Future commit boundary**

```powershell
git add src/sc_mv_dmer/training/handoff.py src/sc_mv_dmer/training/stage2_entry.py schemas tests/training/test_h3_handoff.py tests/training/test_stage2_entry.py
git commit -m "feat: bind immutable stage handoff"
```

### Task 7: 分阶段执行 RG-08 与 Step-5 qualification

**Files:**
- Create: `src/sc_mv_dmer/gates/rg08.py`
- Create: `src/sc_mv_dmer/runtime/memory_probe.py`
- Create: `tests/gates/test_rg08_applicability.py`
- Create: `tests/runtime/test_memory_probe.py`
- Generate during execution: `evidence/gates/rg-08/step5-role-evidence.json`
- Generate during execution: `evidence/qualifications/step-5-stage1.json`

**Interfaces:**
- Consumes: compiled exact topology, hardware profile and stage context.
- Produces: `compile_rg08_roles(...)`; `measure_rg08_role(...)`; `build_step5_qualification(...)`.

- [ ] **Step 1: Write applicability and measurement tests**

```python
def test_a10_role_set_excludes_p4() -> None:
    assert set(compile_rg08_roles(a10_profile())) == {"P1_STAGE1","P2_STAGE1","P1_STAGE2","P3_STAGE2","FVA_FINAL"}

def test_step5_does_not_pass_artifact_dependent_roles() -> None:
    result = build_step5_qualification(step5_role_fixture())
    assert result.role_status["P3_STAGE2"] == "PENDING_STEP_6_ARTIFACT"
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/gates/test_rg08_applicability.py tests/runtime/test_memory_probe.py -v`

Expected: FAIL because RG-08/probe modules are absent.

- [ ] **Step 3: Implement local-first role probes and external handoff**

In fresh processes record PyTorch allocated/reserved and NVML process/device peak; `measured_peak=max(torch_peak_reserved,observed_nvml_process_peak)`. Step 5 may complete P1/P2 Stage-1 and P1 Stage-2 load/resident roles. OOM/fallback saves immutable local evidence and a continuation handoff; local result is `LOCAL_FEASIBLE` or `BLOCKED_LOCAL_COMPUTE`, never canonical RTX4090 PASS.

- [ ] **Step 4: Run final Step-5 verification**

Run: `python -m pytest tests/models tests/generation tests/training tests/gates/test_rg08_applicability.py tests/runtime/test_memory_probe.py -v`

Run: `python -m sc_mv_dmer.cli qualify-step --step 5 --mode formal --output evidence/qualifications/step-5-stage1.json`

Expected: loader/training/handoff roles pass; artifact-dependent roles remain explicitly pending; no formal Stage 2 starts.

- [ ] **Step 5: Future commit boundary**

```powershell
git add src/sc_mv_dmer/gates/rg08.py src/sc_mv_dmer/runtime tests/gates/test_rg08_applicability.py tests/runtime evidence/qualifications/step-5-stage1.json
git commit -m "test: qualify qwen stage1 and stage2 entry"
```
