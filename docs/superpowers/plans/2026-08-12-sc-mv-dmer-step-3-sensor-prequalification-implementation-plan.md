# Step 3 — Sensor Heads 独立预验证 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 仅使用 `E_mel/E_mfcc/E_chroma` 与冻结 pseudo-label 独立训练 Sensor Heads，按 RG-03 验证 mode CCC、energy/brightness MSE、key CE、梯度和 bundle provenance。

**Architecture:** Standalone model 包含三个 handcrafted encoder；Mel Sensor Head 从 `E_mel` 回归 RMS energy，MFCC Sensor Head 从 `E_mfcc` 回归 spectral-centroid brightness，Chroma Sensor Heads 从 `E_chroma` 回归 continuous mode strength 并分类 12-way tonic key。不构建 Deep、Markov、Fusion 或 Qwen。Data module 只从 train 拟合 normalization、constant baseline 和 Laplace-smoothed key prior；checkpoint selector 使用预注册 validation metric，test 全程隔离。

**Tech Stack:** Python、PyTorch、NumPy、Pydantic、pytest、safetensors。

## Global Constraints

- Sensor 只能直接消费原始 `E_mel/E_mfcc/E_chroma`；不得从 `F_fused`、Markov 或其派生输出取输入。
- `L_sensor` 必须能反向更新对应 handcrafted encoder。
- Mode 是连续 `[-1,1]` regression，validation `CCC > 0.7`。
- Energy/brightness Gate 要求 model MSE 至少按 `max(1e-8,1e-6*abs(baseline))` 严格优于 train-only constant baseline。
- Key 是 12-way tonic classification；train-only prior 使用 add-one smoothing `p_c=(n_c+1)/(N+C)`，model validation CE 必须按同一 tolerance 严格优于 prior。
- Key checkpoint selection/early stopping 只依据预登记 validation CE；test、accuracy、macro-F1 和人工挑选不参与。
- Formal FAIL 后不得自动调伪标签粒度；规则修改必须新 Decision/spec/config/cache/version。

---

### Task 1: 建立 Sensor dataset、train-only normalization 与 baselines

**Files:**
- Create: `src/sc_mv_dmer/sensors/data.py`
- Create: `src/sc_mv_dmer/sensors/normalization.py`
- Create: `src/sc_mv_dmer/sensors/baselines.py`
- Create: `tests/sensors/test_sensor_data.py`
- Create: `tests/sensors/test_sensor_baselines.py`

**Interfaces:**
- Consumes: `FourViewFeatureCacheManifest`, `PseudoLabelBundleManifest`, frozen split.
- Produces: `build_sensor_dataset(...) -> SensorDatasetBundle`; `fit_train_normalization(...) -> TargetNormalization`; `fit_train_baselines(...) -> SensorBaselineBundle`.

- [ ] **Step 1: Write leakage and smoothing tests**

```python
def test_normalization_uses_train_membership_only() -> None:
    norm = fit_train_normalization(dataset_fixture())
    assert norm.fit_split == "train"
    assert norm.fit_membership_hash == TRAIN_HASH

def test_key_prior_uses_add_one_smoothing() -> None:
    prior = fit_train_baselines(key_counts=[2, 0], class_count=2)
    assert prior.key_probabilities == pytest.approx([3/4, 1/4])
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/sensors/test_sensor_data.py tests/sensors/test_sensor_baselines.py -v`

Expected: FAIL because sensor data/baseline APIs are absent.

- [ ] **Step 3: Implement exact dataset contracts**

Preserve song/sample/segment, view, class vocabulary/order, target encoding, segment boundaries, valid/missing masks and cache checksums. Compute continuous-target constant means and normalization only from eligible Sensor-train targets; compute key prior from valid train 5-second segments with add-one smoothing.

- [ ] **Step 4: Run GREEN**

Run: `python -m pytest tests/sensors/test_sensor_data.py tests/sensors/test_sensor_baselines.py -v`

Expected: PASS, including zero-frequency class and test-leakage rejection.

- [ ] **Step 5: Future commit boundary**

```powershell
git add src/sc_mv_dmer/sensors tests/sensors
git commit -m "feat: bind sensor data and train-only baselines"
```

### Task 2: 实现 standalone handcrafted encoders 与 Sensor Heads

**Files:**
- Create: `src/sc_mv_dmer/models/handcrafted_encoders.py`
- Create: `src/sc_mv_dmer/models/sensor_heads.py`
- Create: `src/sc_mv_dmer/models/standalone_sensor.py`
- Create: `tests/models/test_standalone_sensor.py`
- Create: `tests/models/test_sensor_gradients.py`

**Interfaces:**
- Consumes: batched Mel/MFCC/Chroma tensors and masks.
- Produces: `StandaloneSensorModel.forward(batch: SensorBatch) -> SensorPrediction`; `assert_sensor_gradients(model, loss) -> GradientEvidence`.

- [ ] **Step 1: Write topology and gradient tests**

```python
def test_standalone_sensor_has_no_forbidden_modules() -> None:
    model = build_standalone_sensor(model_spec_fixture())
    names = {name for name, _ in model.named_modules()}
    assert not any("qwen" in n or "markov" in n or "fusion" in n for n in names)

def test_each_sensor_loss_reaches_its_encoder() -> None:
    evidence = gradient_probe(build_standalone_sensor(model_spec_fixture()), batch_fixture())
    assert evidence.required_encoder_roles_all_nonzero
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/models/test_standalone_sensor.py tests/models/test_sensor_gradients.py -v`

Expected: FAIL because model modules are absent.

- [ ] **Step 3: Implement minimal frozen topology**

Encode each handcrafted view to `[B,T,256]`; branch energy directly from `E_mel`、brightness directly from `E_mfcc`、continuous mode and 12-logit key directly from `E_chroma`. Emit target-specific masks. Store role names so gradient evidence can prove each head reaches its corresponding encoder and all forbidden roles are absent.

- [ ] **Step 4: Run GREEN**

Run: `python -m pytest tests/models/test_standalone_sensor.py tests/models/test_sensor_gradients.py -v`

Expected: PASS for shapes, forbidden-module absence and gradient ownership.

- [ ] **Step 5: Future commit boundary**

```powershell
git add src/sc_mv_dmer/models tests/models/test_standalone_sensor.py tests/models/test_sensor_gradients.py
git commit -m "feat: add standalone sensor topology"
```

### Task 3: 实现 Sensor loss、训练循环和预登记选模

**Files:**
- Create: `src/sc_mv_dmer/sensors/losses.py`
- Create: `src/sc_mv_dmer/sensors/trainer.py`
- Create: `src/sc_mv_dmer/sensors/checkpoints.py`
- Create: `configs/training/sensor-prequalification-v1.yaml`
- Create: `tests/sensors/test_sensor_losses.py`
- Create: `tests/sensors/test_sensor_checkpoint_selection.py`

**Interfaces:**
- Consumes: `StandaloneSensorModel`, Sensor loaders and frozen training config.
- Produces: `train_sensor(run: SensorRunSpec) -> SensorTrainingResult`; `select_sensor_checkpoint(curve: ValidationCurve, rule: SelectionRule) -> CheckpointRef`.

- [ ] **Step 1: Write masked-loss and CE-selection tests**

```python
def test_sensor_loss_is_flat_head_sum() -> None:
    loss = sensor_loss(prediction_fixture(), target_fixture())
    assert loss.total == pytest.approx(loss.rms + loss.brightness + loss.mode + loss.key)

def test_key_checkpoint_is_minimum_registered_validation_ce() -> None:
    selected = select_sensor_checkpoint(curve_with_key_ce([1.2, 0.8, 0.9]), frozen_rule())
    assert selected.validation_index == 1
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/sensors/test_sensor_losses.py tests/sensors/test_sensor_checkpoint_selection.py -v`

Expected: FAIL because loss/trainer/selector are absent.

- [ ] **Step 3: Implement denominator-aware training**

Use `L_sensor=L_rms+L_brightness+L_mode+L_key` with target-specific masks and logged numerators/denominators. Save every normally completed validation checkpoint, all curves, early-stop reason, selection inputs and checkpoint checksums. Never load test data in trainer or selector.

- [ ] **Step 4: Run GREEN**

Run: `python -m pytest tests/sensors/test_sensor_losses.py tests/sensors/test_sensor_checkpoint_selection.py -v`

Expected: PASS, including zero-eligibility and test-loader rejection.

- [ ] **Step 5: Future commit boundary**

```powershell
git add src/sc_mv_dmer/sensors configs/training/sensor-prequalification-v1.yaml tests/sensors
git commit -m "feat: train and select standalone sensors"
```

### Task 4: 实现 RG-03 指标与逐 criterion Gate evaluator

**Files:**
- Create: `src/sc_mv_dmer/metrics/regression.py`
- Create: `src/sc_mv_dmer/metrics/classification.py`
- Create: `src/sc_mv_dmer/gates/rg03.py`
- Create: `tests/gates/test_rg03.py`
- Generate during execution: `evidence/gates/rg-03/rg-03-attempt-v1.json`

**Interfaces:**
- Consumes: validation predictions, baselines, masks and training result.
- Produces: `evaluate_rg03(evidence: RG03Evidence) -> GateEvaluationAttempt`.

- [ ] **Step 1: Write exact threshold tests**

```python
def test_rg03_passes_only_when_all_sensor_criteria_pass() -> None:
    attempt = evaluate_rg03(evidence_fixture(mode_ccc=0.7001, energy=(0.9,1.0), brightness=(1.9,2.0), key_ce=(1.1,1.2)))
    assert attempt.historical_verdict == "PASS"

def test_mode_equal_point_seven_fails_strict_threshold() -> None:
    assert evaluate_rg03(evidence_fixture(mode_ccc=0.7)).historical_verdict == "FAIL"
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/gates/test_rg03.py -v`

Expected: FAIL because evaluator is absent.

- [ ] **Step 3: Implement FP64 metrics and tolerance formula**

Mode uses masked FP64 CCC. Energy/brightness and key compare exact model/baseline values using `model <= baseline - max(1e-8,1e-6*abs(baseline))`. Save accuracy, macro-F1, confusion matrix, supports, priors and smoothing as non-gating key diagnostics. Any nonfinite or provenance mismatch is non-PASS.

- [ ] **Step 4: Run GREEN**

Run: `python -m pytest tests/gates/test_rg03.py tests/sensors -v`

Expected: PASS for strict boundary cases and complete AND logic.

- [ ] **Step 5: Future commit boundary**

```powershell
git add src/sc_mv_dmer/metrics src/sc_mv_dmer/gates/rg03.py tests/gates/test_rg03.py
git commit -m "test: encode RG03 sensor criteria"
```

### Task 5: 封存 StandaloneSensorBundle 与 Step-3 qualification

**Files:**
- Create: `src/sc_mv_dmer/sensors/bundle.py`
- Create: `schemas/standalone_sensor_bundle.schema.json`
- Create: `tests/sensors/test_sensor_bundle.py`
- Generate during execution: `manifests/sensors/standalone-sensor-bundle-v1.json`
- Generate during execution: `evidence/qualifications/step-3-sensor.json`

**Interfaces:**
- Consumes: selected checkpoint, encoder/head state, normalization/baseline and RG-03 attempt.
- Produces: `build_sensor_bundle(...) -> StandaloneSensorBundle`.

- [ ] **Step 1: Write bundle provenance tests**

```python
def test_sensor_bundle_binds_all_upstream_identities() -> None:
    bundle = build_sensor_bundle(bundle_source_fixture())
    assert bundle.feature_cache_hash == FEATURE_HASH
    assert bundle.pseudo_label_hash == LABEL_HASH
    assert bundle.dimension_binding_hash == DIM_HASH
    assert bundle.rg03_evaluation_id
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/sensors/test_sensor_bundle.py -v`

Expected: FAIL because bundle builder is absent.

- [ ] **Step 3: Implement immutable bundle and future Event reference**

Store model/config/normalization/class/mask/segment/checkpoint/metric/curve/gradient/source-run identities and checksums. Expose a typed artifact ref usable by EventSequence provenance, but mark standalone predictions as prequalification evidence rather than final-inference authority.

- [ ] **Step 4: Run final Step-3 verification**

Run: `python -m pytest tests/sensors tests/models/test_standalone_sensor.py tests/models/test_sensor_gradients.py tests/gates/test_rg03.py -v`

Run: `python -m sc_mv_dmer.cli qualify-step --step 3 --mode formal --output evidence/qualifications/step-3-sensor.json`

Expected: PASS only when every RG-03 criterion and gradient/provenance check is current-effective.

- [ ] **Step 5: Future commit boundary**

```powershell
git add src/sc_mv_dmer/sensors/bundle.py schemas/standalone_sensor_bundle.schema.json tests/sensors/test_sensor_bundle.py manifests/sensors evidence/qualifications/step-3-sensor.json
git commit -m "test: qualify standalone sensor bundle"
```
