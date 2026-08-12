# Step 4 — No-LLM 四视图原型 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建 MERT/TimesNet + 三手工 Encoder + shared-A Markov + state-biased Fusion + direct VA head 的 No-LLM 原型，完成统一 S5 的 RG-04、RG-05、RG-06 Gate evidence。

**Architecture:** 四个 Encoder 输出 `[B,90,256]`；各视图具有独立 state prototypes，A1-like topology 共享 `A/pi`；Deep 作为 Fusion query anchor，手工视图提供 Markov-biased key/value。Sensor 从原始 handcrafted `E_v` 分叉，View Dropout 仅修改进入 Markov/Fusion 的手工路径；No-LLM VA head 是 Gate-only terminal，不进入 Full LLM ensemble。

**Tech Stack:** Python、PyTorch、NumPy、safetensors、Pydantic、pytest。

## Global Constraints

- Deep view 永不 dropout；Sensor branch 永不接收 `[VIEW-DROP]`，且 Sensor loss 回传对应 handcrafted Encoder。
- 四视图独立 prototypes；A1 canonical Markov 共享 `A` 和 `pi`，禁止退化为普通 early concatenation。
- Fusion 必须使用 Deep query anchor、handcrafted key/value、Markov consistency bias 和 effective `beta_v`。
- No-LLM 模型不得加载 Qwen hidden backbone、LM head、parser 或 CoT path。
- CRNN reference、primary validation population、label domain、mask、metric implementation、selection protocol 和 S5 在观察结果前冻结；test 不参与 Gate。
- RG-04 必须逐维 `Delta_V>=-0.02 AND Delta_A>=-0.02`；不得平均 V/A。
- RG-05 使用 selected checkpoint、eval mode、View Dropout off、四视图在线；32 个 occupancy cells 全部 `>0.05`，每视图 max `<=0.60`。
- RG-06 对每个 beta 取训练时间最后 3 个正常 validation checkpoints：全部 `>0` 且 population-CV `<=0.10`；`beta_chroma>beta_mel` 只作方向假设。

---

### Task 1: 实现 Deep TimesNet 与四视图 Encoder contract

**Files:**
- Create: `src/sc_mv_dmer/models/deep_encoder.py`
- Modify: `src/sc_mv_dmer/models/handcrafted_encoders.py`
- Create: `src/sc_mv_dmer/models/view_bundle.py`
- Create: `tests/models/test_four_view_encoders.py`

**Interfaces:**
- Consumes: cached `X_deep[90,768]`, `X_mel[90,128]`, `X_mfcc[90,40]`, `X_chroma[90,12]`.
- Produces: `encode_views(batch: FourViewBatch) -> EncodedViews` with four `[B,90,256]` tensors.

- [ ] **Step 1: Write shape and dependency tests**

```python
def test_four_encoders_emit_registered_shapes() -> None:
    out = encode_views(build_encoder_bundle(dimension_fixture()), batch_fixture())
    assert out.deep.shape[-2:] == (90, 256)
    assert out.mel.shape[-2:] == out.mfcc.shape[-2:] == out.chroma.shape[-2:] == (90, 256)
    assert out.dimension_binding_hash == DIM_HASH
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/models/test_four_view_encoders.py -v`

Expected: FAIL because deep/view bundle APIs are absent.

- [ ] **Step 3: Implement the registered encoder graphs**

Deep consumes the Step-2 MERT view and frozen TimesNet/post-bin contract. Handcrafted encoders reuse Step-3 modules with the exact input dimensions. Assert dimension checksum before forward and propagate `InputCoverageMask` without fabricating values.

- [ ] **Step 4: Run GREEN**

Run: `python -m pytest tests/models/test_four_view_encoders.py -v`

Expected: PASS.

- [ ] **Step 5: Future commit boundary**

```powershell
git add src/sc_mv_dmer/models tests/models/test_four_view_encoders.py
git commit -m "feat: add four-view encoder bundle"
```

### Task 2: 实现 shared-A Markov bridge 与完整 diagnostics

**Files:**
- Create: `src/sc_mv_dmer/models/markov.py`
- Create: `src/sc_mv_dmer/models/state_prototypes.py`
- Create: `tests/models/test_markov_bridge.py`
- Create: `tests/models/test_markov_gradients.py`

**Interfaces:**
- Consumes: `EncodedViews`, mask and Markov config.
- Produces: `MarkovBridge.forward(views, mask) -> MarkovOutputs` containing raw `gamma`, post-Markov `gamma_tilde`, `A`, `pi`, state bias and diagnostics.

- [ ] **Step 1: Write sharing, normalization and gradient tests**

```python
def test_canonical_markov_shares_a_and_pi_but_not_prototypes() -> None:
    model = MarkovBridge(canonical_markov_spec())
    assert model.transition_owner_count == 1
    assert model.initial_owner_count == 1
    assert model.prototype_owner_count == 4

def test_probabilities_are_finite_and_row_normalized() -> None:
    out = MarkovBridge(canonical_markov_spec())(encoded_fixture(), mask_fixture())
    assert torch.isfinite(out.gamma_tilde).all()
    assert torch.allclose(out.A.sum(-1), torch.ones(8), atol=1e-6)
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/models/test_markov_bridge.py tests/models/test_markov_gradients.py -v`

Expected: FAIL because Markov modules are absent.

- [ ] **Step 3: Implement the frozen recursion and evidence hooks**

Use exactly 8 states per view, independent prototypes, shared canonical transition/initial parameters and explicit masks. Export raw/post probabilities, state hard/soft summaries, full transition matrix, row entropy and `L_state` inputs without post-hoc state relabeling.

- [ ] **Step 4: Run GREEN**

Run: `python -m pytest tests/models/test_markov_bridge.py tests/models/test_markov_gradients.py -v`

Expected: PASS for probability, sharing, mask and gradient tests.

- [ ] **Step 5: Future commit boundary**

```powershell
git add src/sc_mv_dmer/models/markov.py src/sc_mv_dmer/models/state_prototypes.py tests/models/test_markov_bridge.py tests/models/test_markov_gradients.py
git commit -m "feat: add shared transition Markov bridge"
```

### Task 3: 实现 View Dropout、Sensor 分叉、Fusion 与 beta

**Files:**
- Create: `src/sc_mv_dmer/models/view_dropout.py`
- Create: `src/sc_mv_dmer/models/fusion.py`
- Create: `src/sc_mv_dmer/models/no_llm.py`
- Create: `tests/models/test_view_dropout_topology.py`
- Create: `tests/models/test_state_biased_fusion.py`
- Create: `tests/models/test_no_llm_boundary.py`

**Interfaces:**
- Consumes: encoded views, Sensor Heads, Markov outputs and RNG namespace.
- Produces: `NoLLMModel.forward(batch, mode) -> NoLLMPrediction` containing `Y_va`, Sensor outputs, Markov outputs and effective betas.

- [ ] **Step 1: Write topology and forbidden-path tests**

```python
def test_view_dropout_does_not_change_sensor_inputs() -> None:
    model = build_no_llm_model()
    a = model(batch_fixture(), forced_drop="mel")
    b = model(batch_fixture(), forced_drop=None)
    assert torch.equal(a.sensor_inputs.mel, b.sensor_inputs.mel)
    assert not torch.equal(a.markov_inputs.mel, b.markov_inputs.mel)

def test_no_llm_has_no_qwen_or_lm_head() -> None:
    roles = build_no_llm_model().capability_roles()
    assert roles.isdisjoint({"qwen", "lm_head", "parser", "cot"})
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/models/test_view_dropout_topology.py tests/models/test_state_biased_fusion.py tests/models/test_no_llm_boundary.py -v`

Expected: FAIL because topology is absent.

- [ ] **Step 3: Implement branch-safe dropout and anchored Fusion**

Fork handcrafted `E_v` before dropout; feed unchanged branch to Sensor and dropout-affected branch to Markov/Fusion. Never drop Deep. Concatenate four `[90,256]` outputs only through the frozen Fusion/post-concat contract, with Deep query anchor and per-handcrafted effective beta bias. Store `beta_parameterization` exactly as approved; reject silent parameterization changes.

- [ ] **Step 4: Run GREEN**

Run: `python -m pytest tests/models/test_view_dropout_topology.py tests/models/test_state_biased_fusion.py tests/models/test_no_llm_boundary.py -v`

Expected: PASS for branch equality, deep non-dropout, shape and forbidden-role checks.

- [ ] **Step 5: Future commit boundary**

```powershell
git add src/sc_mv_dmer/models tests/models/test_view_dropout_topology.py tests/models/test_state_biased_fusion.py tests/models/test_no_llm_boundary.py
git commit -m "feat: add state-biased no-llm fusion"
```

### Task 4: 实现 No-LLM training、validation curves 与 metric protocol

**Files:**
- Create: `src/sc_mv_dmer/training/no_llm_trainer.py`
- Create: `src/sc_mv_dmer/training/loss_profiles.py`
- Create: `src/sc_mv_dmer/metrics/va.py`
- Create: `configs/training/no-llm-v1.yaml`
- Create: `tests/training/test_no_llm_trainer.py`
- Create: `tests/metrics/test_va_metrics.py`

**Interfaces:**
- Consumes: No-LLM model, primary split, S5 and frozen loss/selection config.
- Produces: `train_no_llm(spec: TrainingRunSpec) -> TrainingRunResult`; `compute_va_metrics(...) -> VAMetrics`.

- [ ] **Step 1: Write loss and FP64 metric tests**

```python
def test_no_llm_loss_profile_matches_stage1_contract() -> None:
    profile = compile_loss_profile("A1_FULL_7B", "NO_LLM_GATE")
    assert profile.fixed_weights == {"VA":1.0, "state":0.01, "sensor":0.10}
    assert profile.kendall_terms == ()

def test_ccc_is_pooled_fp64_per_dimension() -> None:
    metrics = compute_va_metrics(pred_fixture(), target_fixture(), mask_fixture())
    assert metrics.dtype == "float64"
    assert set(metrics.ccc) == {"V", "A"}
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/training/test_no_llm_trainer.py tests/metrics/test_va_metrics.py -v`

Expected: FAIL because trainer/metric APIs are absent.

- [ ] **Step 3: Implement S5-isolated training runs**

Create one run_id per seed, keep split constant, save every normally completed validation checkpoint, beta values, Markov diagnostics, metrics, optimizer/checkpoint provenance and terminal manifest. Test is inaccessible to selection. Use the frozen selection protocol shared with CRNN.

- [ ] **Step 4: Run GREEN**

Run: `python -m pytest tests/training/test_no_llm_trainer.py tests/metrics/test_va_metrics.py -v`

Expected: PASS for S5 membership invariance, mask reduction and terminal immutability.

- [ ] **Step 5: Future commit boundary**

```powershell
git add src/sc_mv_dmer/training src/sc_mv_dmer/metrics/va.py configs/training/no-llm-v1.yaml tests/training tests/metrics
git commit -m "feat: train no-llm prototype with frozen metrics"
```

### Task 5: 冻结 CRNN reference 并执行 RG-04

**Files:**
- Create: `src/sc_mv_dmer/baselines/crnn.py`
- Create: `src/sc_mv_dmer/gates/rg04.py`
- Create: `configs/baselines/crnn-reference-v1.yaml`
- Create: `tests/gates/test_rg04.py`
- Generate during execution: `manifests/baselines/crnn-reference-v1.json`
- Generate during execution: `evidence/gates/rg-04/rg-04-attempt-v1.json`

**Interfaces:**
- Consumes: frozen CRNN reference bundle and No-LLM S5 results.
- Produces: `evaluate_rg04(reference: CRNNReferenceBundle, candidate: NoLLMEvidence) -> GateEvaluationAttempt`.

- [ ] **Step 1: Write per-dimension boundary tests**

```python
def test_rg04_near_pass_requires_both_dimensions() -> None:
    result = evaluate_rg04(rg04_fixture(delta_v=-0.02, delta_a=0.01))
    assert result.historical_verdict == "PASS"
    assert result.label == "NEAR"

def test_positive_average_cannot_hide_one_failed_dimension() -> None:
    result = evaluate_rg04(rg04_fixture(delta_v=-0.021, delta_a=0.50))
    assert result.historical_verdict == "FAIL"
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/gates/test_rg04.py -v`

Expected: FAIL because CRNN binding/evaluator are absent.

- [ ] **Step 3: Implement protocol-equivalence firewall**

Before reading CCC, assert identical validation split/hash, label domain, target normalization, masks, metric implementation/version, selection protocol and S5. Freeze CRNN artifact/checksum before candidate evaluation. Compare S5 arithmetic-mean CCC separately for V/A and preserve all seed values.

- [ ] **Step 4: Run GREEN and materialize reference before candidate Gate**

Run: `python -m pytest tests/gates/test_rg04.py -v`

Expected: PASS including exact `-0.02`, REACHED, NEAR and failure cases.

- [ ] **Step 5: Future commit boundary**

```powershell
git add src/sc_mv_dmer/baselines src/sc_mv_dmer/gates/rg04.py configs/baselines tests/gates/test_rg04.py
git commit -m "test: bind CRNN reference and RG04"
```

### Task 6: 执行 RG-05、RG-06 并封存 Step-4 evidence

**Files:**
- Create: `src/sc_mv_dmer/gates/rg05.py`
- Create: `src/sc_mv_dmer/gates/rg06.py`
- Create: `tests/gates/test_rg05.py`
- Create: `tests/gates/test_rg06.py`
- Generate during execution: `evidence/gates/rg-05/rg-05-attempt-v1.json`
- Generate during execution: `evidence/gates/rg-06/rg-06-attempt-v1.json`
- Generate during execution: `evidence/qualifications/step-4-no-llm.json`

**Interfaces:**
- Consumes: selected No-LLM checkpoints, full validation curves and frozen masks.
- Produces: `evaluate_rg05(...) -> GateEvaluationAttempt`; `evaluate_rg06(...) -> GateEvaluationAttempt`.

- [ ] **Step 1: Write collapse and beta-window tests**

```python
def test_rg05_requires_every_state_strictly_above_five_percent() -> None:
    assert evaluate_rg05(occupancy_fixture(one_cell=0.05)).historical_verdict == "FAIL"

def test_rg06_uses_last_three_normal_checkpoints_ddof_zero() -> None:
    result = evaluate_rg06(beta_curve_fixture(last3=[1.0, 1.0, 1.1]))
    assert result.criteria["mel"].population_ddof == 0
    assert result.criteria["mel"].passed is True
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/gates/test_rg05.py tests/gates/test_rg06.py -v`

Expected: FAIL because evaluators are absent.

- [ ] **Step 3: Implement exact operational criteria**

RG-05 uses post-Markov `gamma_tilde` hard argmax with smallest-index tie break on valid primary-validation 2 Hz frames; saves raw/post soft mass, hard counts/occupancy, A/entropy/L_state. RG-06 rejects fewer than 3 consecutive normal checkpoints, nonfinite values, any last-3 `<=0`, or any per-view CV `>0.10`; saves parameterization and direction hypothesis separately.

- [ ] **Step 4: Run full Step-4 verification**

Run: `python -m pytest tests/models tests/training/test_no_llm_trainer.py tests/metrics/test_va_metrics.py tests/gates/test_rg04.py tests/gates/test_rg05.py tests/gates/test_rg06.py -v`

Run: `python -m sc_mv_dmer.cli qualify-step --step 4 --mode formal --output evidence/qualifications/step-4-no-llm.json`

Expected: Step 4 PASS only if RG-04 and all applicable RG-05/06 attempts are current-effective; otherwise Qwen entry is blocked.

- [ ] **Step 5: Future commit boundary**

```powershell
git add src/sc_mv_dmer/gates tests/gates evidence/gates/rg-04 evidence/gates/rg-05 evidence/gates/rg-06 evidence/qualifications/step-4-no-llm.json
git commit -m "test: qualify no-llm research gates"
```
