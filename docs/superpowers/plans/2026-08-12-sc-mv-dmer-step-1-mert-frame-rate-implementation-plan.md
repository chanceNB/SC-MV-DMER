# Step 1 — MERT 真实帧率与下游维度绑定 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 对 pin 后的真实 MERT artifact 执行只读 forward，测量实际输出帧率并编译全链 `DownstreamDimensionBinding`，关闭 RG-01 后才允许特征缓存和模型构建。

**Architecture:** Upstream inventory 固化 MERT model/processor/config bytes；measurement runner 只产出 raw observation；TimeGrid compiler 将实测 frame centers 映射到冻结 2 Hz；dimension compiler 从 measurement + frozen constants 生成唯一 shape DAG。任何下游模块只读取 manifest checksum，不自行重算维度。

**Tech Stack:** Python、PyTorch、Transformers、safetensors、NumPy、Pydantic、pytest、SHA-256。

## Global Constraints

- 方案中的 50 Hz 只能作为历史假设，不得成为 executable constant。
- 所有实测/派生/冻结字段分别标记 `MEASURED`、`DERIVED`、`FROZEN_CONSTANT`。
- Canonical TimeGrid 是 2 Hz；标准 DEAM excerpt `T=90`。
- Padding 不得当作 silence；frame/bin membership 必须可回查到 source sample/time。
- 未 pin upstream bytes、RG-01 非 PASS、dimension manifest 未封存或 checksum 漂移时，禁止 Step 2。
- 本计划不训练模型，只执行 read-only forward 与维度编译。

---

### Task 1: 绑定 MERT Upstream Model Manifest

**Files:**
- Create: `src/sc_mv_dmer/features/mert_upstream.py`
- Create: `configs/upstream/mert-primary.yaml`
- Create: `schemas/upstream_model_manifest.schema.json`
- Create: `tests/features/test_mert_upstream.py`
- Generate during execution: `reports/preflight/mert-snapshot-inventory.json`
- Generate during execution: `manifests/upstream/mert-primary-v1.json`

**Interfaces:**
- Consumes: `MertUpstreamConfig`, configured model root.
- Produces: `discover_mert(config: MertUpstreamConfig, model_root: Path) -> UpstreamInventory`; `bind_mert(inventory: UpstreamInventory) -> UpstreamModelManifest`.

- [ ] **Step 1: Write upstream identity tests**

```python
def test_mert_manifest_covers_processor_config_and_weights() -> None:
    manifest = bind_mert(inventory_fixture())
    assert {f.role for f in manifest.files} >= {"processor", "config", "weights"}
    assert all(len(f.sha256) == 64 for f in manifest.files)

def test_missing_weight_bytes_fail_closed() -> None:
    with pytest.raises(UpstreamArtifactError):
        bind_mert(inventory_fixture(missing="weights"))
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/features/test_mert_upstream.py -v`

Expected: FAIL because `mert_upstream` is absent.

- [ ] **Step 3: Implement read-only inventory and binding**

Record model ID, immutable revision when available, processor/config identity, selected layers 5/6 contract, hidden size, sample rate, source-relative file roles, byte sizes and SHA-256. Do not download automatically; missing local artifact produces a pre-flight blocker.

- [ ] **Step 4: Run GREEN and inventory configured model root**

Run: `python -m pytest tests/features/test_mert_upstream.py -v`

Run: `python -m sc_mv_dmer.cli discover-upstream --kind mert --root $env:SC_MV_DMER_MODEL_ROOT --output reports/preflight/mert-snapshot-inventory.json`

Expected: tests PASS; real inventory is complete or emits `PINNED_MERT_UPSTREAM_BLOCKED`.

- [ ] **Step 5: Future commit boundary**

```powershell
git add src/sc_mv_dmer/features/mert_upstream.py configs/upstream/mert-primary.yaml schemas/upstream_model_manifest.schema.json tests/features/test_mert_upstream.py
git commit -m "feat: bind MERT upstream artifact"
```

### Task 2: 测量真实 MERT frame observation

**Files:**
- Create: `src/sc_mv_dmer/features/mert_measure.py`
- Create: `schemas/mert_frame_observation.schema.json`
- Create: `tests/features/test_mert_measure.py`
- Create: `tests/features/audio_fixtures.py`
- Generate during execution: `evidence/rg-01/mert-frame-observations.jsonl`

**Interfaces:**
- Consumes: `UpstreamModelManifest`, registered `AudioFixture`.
- Produces: `measure_mert(model: MertBundle, audio: AudioFixture) -> MertFrameObservation`.

- [ ] **Step 1: Write observation tests with a deterministic fake bundle**

```python
def test_measurement_uses_sample_count_not_declared_duration() -> None:
    obs = measure_mert(fake_mert(frames=250), audio_fixture(samples=80000, sample_rate=16000))
    assert obs.input_duration_seconds == 5.0
    assert obs.output_frame_count == 250
    assert obs.actual_frame_rate_hz == 50.0

def test_nonfinite_hidden_state_is_rejected() -> None:
    with pytest.raises(NonFiniteMertOutputError):
        measure_mert(fake_mert(nonfinite=True), audio_fixture())
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/features/test_mert_measure.py -v`

Expected: FAIL with missing measurement API.

- [ ] **Step 3: Implement observation capture**

`tests/features/audio_fixtures.py` generates a deterministic 16 kHz five-second sine array in memory and assigns a stable fixture hash; no binary fixture is hand-edited. Record fixture/sample identity, exact input samples/sample rate/duration, attention/coverage mask, layer 5 and 6 tensor shapes, output frame count, computed frame rate, inferred effective stride, dtype/device/library environment and upstream checksum. Preserve raw observations; do not round values used by RG-01.

- [ ] **Step 4: Run GREEN and execute real registered fixtures**

Run: `python -m pytest tests/features/test_mert_measure.py -v`

Run: `python -m sc_mv_dmer.cli measure-mert --upstream manifests/upstream/mert-primary-v1.json --dataset manifests/datasets/deam-primary-v1.json --fixture-policy rg01-v1 --output evidence/rg-01/mert-frame-observations.jsonl`

Expected: tests PASS; every observation finite and provenance-complete.

- [ ] **Step 5: Future commit boundary**

```powershell
git add src/sc_mv_dmer/features/mert_measure.py schemas/mert_frame_observation.schema.json tests/features
git commit -m "feat: measure actual MERT frame rate"
```

### Task 3: 编译 MERT frame 到 canonical 2 Hz TimeGrid

**Files:**
- Create: `src/sc_mv_dmer/timegrid.py`
- Create: `src/sc_mv_dmer/features/mert_alignment.py`
- Create: `schemas/timegrid_binding.schema.json`
- Create: `tests/features/test_mert_alignment.py`

**Interfaces:**
- Consumes: `MertFrameObservation`, `InputCoverageMask`.
- Produces: `compile_mert_timegrid(observation: MertFrameObservation, grid: TimeGrid) -> MertTimeGridBinding`.

- [ ] **Step 1: Write exact boundary tests**

```python
def test_standard_excerpt_maps_to_90_bins_without_gap_or_overlap() -> None:
    binding = compile_mert_timegrid(observation_45s_fixture(), TimeGrid.hz2(duration_seconds=45))
    assert binding.output_length == 90
    assert binding.coverage_gap_count == 0
    assert binding.duplicate_membership_count == 0

def test_padding_frames_are_not_valid_audio() -> None:
    binding = compile_mert_timegrid(padded_observation_fixture(), TimeGrid.hz2(45))
    assert all(not binding.valid_frame_mask[i] for i in binding.padding_frame_indices)
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/features/test_mert_alignment.py -v`

Expected: FAIL because TimeGrid compiler does not exist.

- [ ] **Step 3: Implement half-open boundary membership**

Represent frame centers and 0.5-second output bins explicitly; use the frozen boundary rule from RG-01 authority. Store every input-frame to output-bin membership, valid coverage, padding exclusion and reduction rule. Reject ambiguous or uncovered valid frames.

- [ ] **Step 4: Run GREEN**

Run: `python -m pytest tests/features/test_mert_alignment.py -v`

Expected: PASS for exact-boundary, short/padded and standard 45-second fixtures.

- [ ] **Step 5: Future commit boundary**

```powershell
git add src/sc_mv_dmer/timegrid.py src/sc_mv_dmer/features/mert_alignment.py schemas/timegrid_binding.schema.json tests/features/test_mert_alignment.py
git commit -m "feat: bind MERT frames to 2Hz timegrid"
```

### Task 4: 编译全链 DownstreamDimensionBinding

**Files:**
- Create: `src/sc_mv_dmer/foundation/dimensions.py`
- Create: `configs/dimensions/sc-mv-dmer-primary-v1.yaml`
- Create: `schemas/downstream_dimension_binding.schema.json`
- Create: `tests/foundation/test_downstream_dimensions.py`
- Generate during execution: `manifests/dimensions/mert-downstream-dimensions-v1.json`

**Interfaces:**
- Consumes: measured observation aggregate, TimeGrid binding and frozen model constants.
- Produces: `compile_dimensions(source: DimensionSourceBundle) -> DownstreamDimensionBinding`; `assert_dimension_dependency(manifest_hash: str, consumer: ConsumerSpec) -> None`.

- [ ] **Step 1: Write shape-DAG tests**

```python
def test_dimension_binding_resolves_full_va_and_answer_path() -> None:
    dims = compile_dimensions(source_bundle_fixture())
    assert dims["X_deep"] == [90, 768]
    assert dims["fusion_concat"] == [90, 1024]
    assert dims["F_fused"] == [90, 256]
    assert dims["Y_va"] == [90, 2]
    assert dims["answer_time_positions"] == 90

def test_consumer_with_wrong_manifest_hash_fails() -> None:
    with pytest.raises(DownstreamDimensionBindingMismatch):
        assert_dimension_dependency("wrong", consumer_fixture())
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/foundation/test_downstream_dimensions.py -v`

Expected: FAIL because dimension compiler is absent.

- [ ] **Step 3: Implement source-tagged dimension DAG**

Include MERT input/output/stride/frame centers; TimesNet length/period/downsampling; `T=90`; `X_deep[90,768]`, `E_deep[90,256]`, `X_mel[90,128]`, `X_mfcc[90,40]`, `X_chroma[90,12]`, handcrafted `E_v[90,256]`, Markov gamma/state-bias, fusion `[90,1024]→[90,256]`, Sensor outputs, Qwen time-series tokens, `Y_va[90,2]` and ANSWER 90 positions. Attach dependency edges and source classification to every field.

- [ ] **Step 4: Run GREEN and materialize manifest**

Run: `python -m pytest tests/foundation/test_downstream_dimensions.py -v`

Run: `python -m sc_mv_dmer.cli compile-dimensions --observations evidence/rg-01/mert-frame-observations.jsonl --output manifests/dimensions/mert-downstream-dimensions-v1.json`

Expected: PASS and an immutable checksum-complete manifest.

- [ ] **Step 5: Future commit boundary**

```powershell
git add src/sc_mv_dmer/foundation/dimensions.py configs/dimensions schemas/downstream_dimension_binding.schema.json tests/foundation/test_downstream_dimensions.py
git commit -m "feat: compile downstream dimension binding"
```

### Task 5: 终结 RG-01 与 Step-1 qualification

**Files:**
- Create: `src/sc_mv_dmer/gates/rg01.py`
- Create: `tests/gates/test_rg01.py`
- Generate during execution: `evidence/gates/rg-01/rg-01-attempt-v1.json`
- Generate during execution: `evidence/qualifications/step-1-mert.json`

**Interfaces:**
- Consumes: upstream manifest, raw observations, TimeGrid binding and dimension manifest.
- Produces: `evaluate_rg01(evidence: RG01Evidence) -> GateEvaluationAttempt`.

- [ ] **Step 1: Write Gate completeness tests**

```python
def test_rg01_requires_actual_frame_rate_and_all_downstream_dimensions() -> None:
    attempt = evaluate_rg01(complete_evidence_fixture())
    assert attempt.historical_verdict == "PASS"
    assert attempt.effective_verdict == "PASS"

def test_missing_dimension_edge_is_non_pass() -> None:
    assert evaluate_rg01(evidence_missing("fusion_concat")).historical_verdict != "PASS"
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/gates/test_rg01.py -v`

Expected: FAIL because RG-01 evaluator is absent.

- [ ] **Step 3: Implement exact evidence checks**

Require finite/reproducible observations, pinned upstream, complete layer/coverage/timestamp evidence, exact 2 Hz mapping, complete dimension dependency DAG and checksums. Persist raw values and machine-readable criterion results; do not substitute rounded displayed rate.

- [ ] **Step 4: Run final Step-1 verification**

Run: `python -m pytest tests/features tests/foundation/test_downstream_dimensions.py tests/gates/test_rg01.py -v`

Run: `python -m sc_mv_dmer.cli qualify-step --step 1 --mode formal --output evidence/qualifications/step-1-mert.json`

Expected: PASS only when RG-01 is current-effective and dimension manifest is immutable.

- [ ] **Step 5: Future commit boundary**

```powershell
git add src/sc_mv_dmer/gates/rg01.py tests/gates/test_rg01.py evidence/gates/rg-01 evidence/qualifications/step-1-mert.json manifests/dimensions
git commit -m "test: qualify MERT rate and dimension graph"
```
