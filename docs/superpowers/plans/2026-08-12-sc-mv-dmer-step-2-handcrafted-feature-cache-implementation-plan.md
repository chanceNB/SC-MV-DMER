# Step 2 — 四视图、伪标签与不可变缓存 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 RG-01 与 `DownstreamDimensionBinding` 上离线提取 MERT deep、Mel、MFCC、Chroma 四视图和 RMS/brightness/mode/key 四类伪标签，生成 immutable cache、全量自动审计和 3 首歌曲人工对齐包。

**Architecture:** 每种 extractor 只产生带 raw-frame timestamp 的 typed record；统一 binner 依据 RG-01 half-open 0.5s windows 做 within-bin mean；content-addressed cache writer 原子封存 tensor 与 provenance。自动审计覆盖所有 formal eligible records，人工包只使用查看前冻结的 3 首 train songs。

**Tech Stack:** Python、NumPy、librosa、soundfile、PyTorch、Pydantic、matplotlib、pytest、SHA-256。

## Global Constraints

- Mel: 22.05 kHz, `n_fft=2048`, `hop_length=512`, 128 mel bins, power 2 → dB → `[90,128]`。
- MFCC: 与 Mel 共享 STFT timing，40 coefficients → `[90,40]`；c0 policy 必须进入 manifest。
- Chroma: CQT, `hop_length=512`, 12 chroma, `bins_per_octave=36`；raw-frame L1 normalization 后再 bin mean → `[90,12]`。
- 三手工视图消费同一 22.05 kHz waveform；Deep 与它们共享 dataset/song/sample、45s segment、TimeGrid 和 mask。
- 每个 eligible raw frame 恰好属于一个 bin；禁止 silent trim/pad/fill/interpolation/adaptive pooling/reshape 制造 90 bins。
- 自动审计覆盖 train/validation/test 全部 eligible records，但不读 target、不选模；人工 3 首只从 train 无放回预注册抽样。
- RG-02 人工 criterion 必须由真实 human reviewer 完成；Codex 只能生成包和计算 automatic verdict。
- Cache content/version 不可覆盖；任何 extractor、normalization、timestamp 或 source byte 改变必须新 version。

---

### Task 1: 定义 feature、pseudo-label 与 cache contract

**Files:**
- Create: `src/sc_mv_dmer/features/models.py`
- Create: `src/sc_mv_dmer/features/cache.py`
- Create: `src/sc_mv_dmer/data/coverage.py`
- Create: `configs/features/deam-four-view-v1.yaml`
- Create: `configs/features/deam-pseudo-label-v1.yaml`
- Create: `schemas/four_view_feature_record.schema.json`
- Create: `schemas/feature_cache_manifest.schema.json`
- Create: `schemas/pseudo_label_bundle.schema.json`
- Create: `tests/features/test_feature_contracts.py`
- Create: `tests/features/test_cache_immutability.py`
- Create: `tests/data/test_evaluation_input_coverage.py`

**Interfaces:**
- Produces: `FeatureFrameRecord`, `BinnedFeatureRecord`, `PseudoLabelRecord`; `compile_input_coverage(...) -> EvaluationInputCoverage`; `write_cache(records: Iterable[FeatureRecord], spec: CacheSpec) -> FourViewFeatureCacheManifest`.

- [ ] **Step 1: Write schema and immutability tests**

```python
def test_feature_record_requires_dimension_dependency_hash() -> None:
    with pytest.raises(ValidationError):
        FeatureFrameRecord(view="mel", values=np.zeros((10, 128)), timestamps=TIMES)

def test_finalized_cache_version_cannot_change_content(tmp_path: Path) -> None:
    manifest = write_fixture_cache(tmp_path)
    with pytest.raises(ImmutableCacheError):
        overwrite_same_version(manifest, changed_tensor())

def test_input_and_target_masks_remain_distinct() -> None:
    coverage = compile_input_coverage(coverage_fixture())
    assert coverage.input_coverage_mask_hash != coverage.target_validity_mask_hash
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/features/test_feature_contracts.py tests/features/test_cache_immutability.py tests/data/test_evaluation_input_coverage.py -v`

Expected: FAIL because models/cache do not exist.

- [ ] **Step 3: Implement typed records and atomic content-addressed writer**

Require dataset/song/sample, source-audio hash, view/rule version, implementation/library version, raw timestamps/support, TimeGrid/mask, dimension-manifest checksum, dtype/shape/finite status and content checksum. Compile `InputCoverageMask` independently from `TargetValidityMask` and from Sensor/Event validity; forbid one mask being substituted for another. Write to a staging directory, verify all checksums, then atomically finalize; reject same-version mutation.

- [ ] **Step 4: Run GREEN**

Run: `python -m pytest tests/features/test_feature_contracts.py tests/features/test_cache_immutability.py tests/data/test_evaluation_input_coverage.py -v`

Expected: PASS.

- [ ] **Step 5: Future commit boundary**

```powershell
git add src/sc_mv_dmer/features/models.py src/sc_mv_dmer/features/cache.py src/sc_mv_dmer/data/coverage.py configs/features schemas tests/features tests/data/test_evaluation_input_coverage.py
git commit -m "feat: define immutable feature cache contracts"
```

### Task 2: 提取并缓存 MERT Deep view

**Files:**
- Create: `src/sc_mv_dmer/features/deep.py`
- Create: `tests/features/test_deep_extractor.py`
- Generate during execution: `cache/features/deam-four-view-v1/deep/`

**Interfaces:**
- Consumes: pinned MERT manifest, RG-01 frame/TimeGrid binding, audio record.
- Produces: `extract_deep(record: AudioRecord, bundle: MertBundle, binding: DownstreamDimensionBinding) -> BinnedFeatureRecord`.

- [ ] **Step 1: Write layer/gating/alignment tests**

```python
def test_deep_extractor_uses_registered_layers_and_90_bins() -> None:
    result = extract_deep(audio_fixture(), fake_mert_bundle(), dimension_fixture())
    assert result.values.shape == (90, 768)
    assert result.layer_ids == (5, 6)
    assert result.timegrid_binding_hash == RG01_BINDING_HASH
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/features/test_deep_extractor.py -v`

Expected: FAIL because extractor is absent.

- [ ] **Step 3: Implement extraction from Step-1 geometry**

Use the pinned processor/resampler/layers and frozen layer-gating rule; persist raw valid-frame coverage and 2 Hz reduction membership. Do not estimate frame rate, create a second pooling rule or mix padding into bin means.

- [ ] **Step 4: Run GREEN**

Run: `python -m pytest tests/features/test_deep_extractor.py -v`

Expected: PASS for exact shape, layer, coverage and dependency hash.

- [ ] **Step 5: Future commit boundary**

```powershell
git add src/sc_mv_dmer/features/deep.py tests/features/test_deep_extractor.py
git commit -m "feat: extract RG01-bound deep features"
```

### Task 3: 提取 Mel、MFCC、Chroma raw frames

**Files:**
- Create: `src/sc_mv_dmer/features/handcrafted.py`
- Create: `src/sc_mv_dmer/features/resample.py`
- Create: `tests/features/test_handcrafted_extractors.py`
- Generate during execution: `cache/features/deam-four-view-v1/mel/`
- Generate during execution: `cache/features/deam-four-view-v1/mfcc/`
- Generate during execution: `cache/features/deam-four-view-v1/chroma/`

**Interfaces:**
- Consumes: one registered waveform and frozen extractor config.
- Produces: `extract_handcrafted(waveform: WaveformRecord, spec: HandcraftedSpec) -> Mapping[ViewName, FeatureFrameRecord]`.

- [ ] **Step 1: Write exact-parameter and shared-waveform tests**

```python
def test_handcrafted_shapes_and_shared_timing() -> None:
    out = extract_handcrafted(waveform_fixture(), frozen_handcrafted_spec())
    assert out["mel"].feature_dim == 128
    assert out["mfcc"].feature_dim == 40
    assert out["chroma"].feature_dim == 12
    assert out["mel"].waveform_hash == out["mfcc"].waveform_hash == out["chroma"].waveform_hash
    assert out["mel"].hop_length == out["mfcc"].hop_length == out["chroma"].hop_length == 512
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/features/test_handcrafted_extractors.py -v`

Expected: FAIL because extractor is absent.

- [ ] **Step 3: Implement exact frozen frontends**

Decode/channel policy and 22.05 kHz resampler identity enter provenance. Derive timestamps with integer/rational arithmetic from sample rate, hop, center/pad/support. Apply Chroma raw-frame L1 normalization before binning; record zero/near-zero handling from the versioned config and reject unregistered epsilon changes.

- [ ] **Step 4: Run GREEN**

Run: `python -m pytest tests/features/test_handcrafted_extractors.py -v`

Expected: PASS including Chroma nonzero-frame L1 tolerance and zero-frame diagnostics.

- [ ] **Step 5: Future commit boundary**

```powershell
git add src/sc_mv_dmer/features/handcrafted.py src/sc_mv_dmer/features/resample.py tests/features/test_handcrafted_extractors.py
git commit -m "feat: extract frozen handcrafted views"
```

### Task 4: 统一 2 Hz binning 并生成四类 pseudo-label

**Files:**
- Create: `src/sc_mv_dmer/features/binning.py`
- Create: `src/sc_mv_dmer/features/pseudo_labels.py`
- Create: `tests/features/test_feature_binning.py`
- Create: `tests/features/test_pseudo_labels.py`

**Interfaces:**
- Consumes: raw frames, RG-01 TimeGrid and registered pseudo-label rules.
- Produces: `bin_feature(record: FeatureFrameRecord, grid: TimeGrid) -> BinnedFeatureRecord`; `derive_pseudo_labels(waveform: WaveformRecord, spec: PseudoLabelSpec) -> PseudoLabelRecord`.

- [ ] **Step 1: Write exact-once and pseudo-label provenance tests**

```python
@pytest.mark.parametrize("view,dim", [("mel",128), ("mfcc",40), ("chroma",12)])
def test_every_eligible_frame_is_assigned_once(view: str, dim: int) -> None:
    result = bin_feature(raw_view_fixture(view, dim), TimeGrid.hz2(45))
    assert result.values.shape == (90, dim)
    assert result.unassigned_count == 0
    assert result.duplicate_count == 0
    assert sum(result.bin_counts) == result.raw_eligible_count
    assert min(result.bin_counts) > 0

def test_pseudo_labels_have_rule_and_segment_identity() -> None:
    labels = derive_pseudo_labels(waveform_fixture(), pseudo_label_spec())
    assert {x.name for x in labels.targets} == {"rms", "brightness", "mode", "key"}
    assert labels.rule_version and labels.segment_boundaries
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/features/test_feature_binning.py tests/features/test_pseudo_labels.py -v`

Expected: FAIL because binner/rules are absent.

- [ ] **Step 3: Implement deterministic binner and registered rule dispatch**

Assign by frame center to one half-open RG-01 bin and compute within-bin mean. Pseudo-label runner emits RMS energy, spectral-centroid brightness, continuous mode-strength target and 12-way tonic key segment target with rule version, segment boundaries, valid/missing masks and source checksum. It must not tune granularity from validation outcomes.

- [ ] **Step 4: Run GREEN**

Run: `python -m pytest tests/features/test_feature_binning.py tests/features/test_pseudo_labels.py -v`

Expected: PASS for all views, exact 90 bins and complete pseudo-label provenance.

- [ ] **Step 5: Future commit boundary**

```powershell
git add src/sc_mv_dmer/features/binning.py src/sc_mv_dmer/features/pseudo_labels.py tests/features/test_feature_binning.py tests/features/test_pseudo_labels.py
git commit -m "feat: bin features and derive sensor targets"
```

### Task 5: Materialize 全 population cache 与自动 RG-02 audit

**Files:**
- Create: `src/sc_mv_dmer/features/materialize.py`
- Create: `src/sc_mv_dmer/gates/rg02.py`
- Create: `tests/features/test_materialization.py`
- Create: `tests/gates/test_rg02_automatic.py`
- Generate during execution: `manifests/features/deam-four-view-cache-v1.json`
- Generate during execution: `manifests/features/deam-pseudo-label-bundle-v1.json`
- Generate during execution: `evidence/gates/rg-02/automatic-audit-v1.json`

**Interfaces:**
- Consumes: dataset manifest, four extractors, pseudo-label runner and dimension binding.
- Produces: `materialize_population(...) -> tuple[FourViewFeatureCacheManifest, PseudoLabelBundleManifest]`; `evaluate_rg02_automatic(...) -> CriterionResult`.

- [ ] **Step 1: Write full-population failure aggregation tests**

```python
def test_one_bad_record_fails_automatic_rg02() -> None:
    result = evaluate_rg02_automatic(cache_fixture(bad_record="song-17"))
    assert result.passed is False
    assert result.failure_sample_ids == ("song-17",)
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/features/test_materialization.py tests/gates/test_rg02_automatic.py -v`

Expected: FAIL because materializer/evaluator are absent.

- [ ] **Step 3: Implement resumable staging, not mutable final cache**

Allow restart only inside a non-final staging run with item-level checksums. Finalize only after every eligible record has all four views, pseudo labels, masks, shape/dtype/finite/time accounting and checksum validation. Any failure keeps partial evidence but does not publish a complete manifest.

- [ ] **Step 4: Run tests and real materialization**

Run: `python -m pytest tests/features/test_materialization.py tests/gates/test_rg02_automatic.py -v`

Run: `python -m sc_mv_dmer.cli materialize-features --dataset manifests/datasets/deam-primary-v1.json --dimensions manifests/dimensions/mert-downstream-dimensions-v1.json --profile deam-four-view-v1 --output manifests/features/deam-four-view-cache-v1.json`

Expected: tests PASS; materialization publishes only a checksum-complete cache.

- [ ] **Step 5: Future commit boundary**

```powershell
git add src/sc_mv_dmer/features/materialize.py src/sc_mv_dmer/gates/rg02.py tests/features/test_materialization.py tests/gates/test_rg02_automatic.py
git commit -m "feat: materialize and audit four-view cache"
```

### Task 6: 冻结 3 首 train-song 可视化人工审核包并终结 RG-02

**Files:**
- Create: `src/sc_mv_dmer/features/alignment_review.py`
- Create: `tests/features/test_alignment_review.py`
- Generate during execution: `evidence/gates/rg-02/manual-sample-v1.json`
- Generate during execution: `evidence/gates/rg-02/review-package-v1/`
- Generate during execution: `evidence/gates/rg-02/rg-02-attempt-v1.json`
- Generate during execution: `evidence/qualifications/step-2-features.json`

**Interfaces:**
- Consumes: immutable cache, frozen train split, automatic audit.
- Produces: `select_alignment_sample(...) -> AlignmentSample`; `build_alignment_package(...) -> HumanReviewPackage`; `finalize_rg02(...) -> GateEvaluationAttempt`.

- [ ] **Step 1: Write pre-view sampling and authority tests**

```python
def test_alignment_sample_is_three_unique_train_songs() -> None:
    sample = select_alignment_sample(train_ids(), rule_fixture())
    assert len(sample.song_ids) == len(set(sample.song_ids)) == 3
    assert set(sample.song_ids) <= set(train_ids())

def test_automatic_result_cannot_supply_human_approval() -> None:
    with pytest.raises(HumanAuthorityRequired):
        finalize_rg02(automatic_pass(), reviewer_records=[])
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/features/test_alignment_review.py -v`

Expected: FAIL because sampling/package builder is absent.

- [ ] **Step 3: Implement immutable review package**

Freeze population/rule/seed/IDs/order before rendering. Each 0–45s figure overlays waveform, 0.5s boundaries, Deep/Mel/MFCC/Chroma preregistered summaries, frame/bin counts, pseudo-label event timing, masks and hashes. Import real-human M1 identity, M2 start/end, M3 bin consistency, applicable M4 event timing, M5 no systematic offset; keep raw reviewer identity and notes append-only.

- [ ] **Step 4: Run tests and build package**

Run: `python -m pytest tests/features/test_alignment_review.py tests/gates/test_rg02_automatic.py -v`

Run: `python -m sc_mv_dmer.cli build-rg02-review --cache manifests/features/deam-four-view-cache-v1.json --split manifests/splits/deam-primary-song-80-10-10-v1.json --output evidence/gates/rg-02/review-package-v1`

Expected: package is built with checksum; Gate remains `PENDING_MANUAL_REVIEW` until three real-human reviews pass all applicable criteria.

- [ ] **Step 5: Finalize after human records and future commit**

Run: `python -m sc_mv_dmer.cli finalize-gate --gate RG-02 --evidence evidence/gates/rg-02 --output evidence/gates/rg-02/rg-02-attempt-v1.json`

Expected: PASS only for `AUTOMATIC_ALL_RECORDS_PASS AND MANUAL_3_SONGS_PASS`.

Run: `python -m sc_mv_dmer.cli qualify-step --step 2 --mode formal --output evidence/qualifications/step-2-features.json`

Expected: Step 2 is VALID only when RG-02 is current-effective and `EVALUATION_INPUT_COVERAGE_EXECUTABLE_BINDING` has a checksum-complete evidence record.

```powershell
git add src/sc_mv_dmer/features/alignment_review.py tests/features/test_alignment_review.py evidence/gates/rg-02 evidence/qualifications/step-2-features.json manifests/features
git commit -m "test: qualify four-view alignment and cache"
```
