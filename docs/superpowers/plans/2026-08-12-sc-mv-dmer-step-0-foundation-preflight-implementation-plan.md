# Step 0 — Foundation 与数据预检 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立机器可读 Foundation、run/evidence 生命周期、stable identity、冻结 DEAM primary population/split、annotation audit 和 formal/debug pre-flight，为实验 Step 1 提供唯一合法入口。

**Architecture:** `sc_mv_dmer.foundation` 负责 canonical config、manifest、capability、stage 和 effective validity；`sc_mv_dmer.data` 只从可配置 data root 发现并绑定 source bytes。人类可读 Research Spec 保持唯一研究语义权威，YAML/JSON 是版本化可执行映射。

**Tech Stack:** Python `>=3.10,<3.13`、Pydantic v2、PyYAML、pytest、Hypothesis、JSON Schema、SHA-256、PowerShell。

## Global Constraints

- 当前 `M0=READY_TO_CLOSE`，不是 PASS；不得伪造 human approval artifact 或 Gate attempt。
- Formal Git workspace 必须 clean；dirty 只能中止 formal 或显式降级 debug。
- `semantic_config_hash` 排除 runtime/machine/path/timestamp；`resolved_config_hash` 覆盖完整 canonical resolved config。
- Formal CLI 只能覆盖 runtime allowlist；任何 semantic CLI override 强制 `run_mode=debug` 并禁止 paper aggregation。
- Primary dataset membership 是冻结的 1,744 DEAM excerpts；唯一歌曲级 80/10/10 split 不随 S5 改变。
- RunSpec 在启动时产生；SUCCEEDED/FAILED/INTERRUPTED 都是不可逆 terminal；post-hoc evaluation 创建独立 evaluation run。
- Historical verdict 与 effective validity 分离；invalidation/supersession append-only 并按 dependency graph 传播。
- 不硬编码绝对 data path；路径 relocation 不得污染 semantic identity。
- 本计划的 Git 命令只作为未来获得实施与 Git 授权后的 commit boundary；规划阶段不执行。

---

### Task 1: 建立 Python 项目与测试基线

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/sc_mv_dmer/__init__.py`
- Create: `src/sc_mv_dmer/cli.py`
- Create: `tests/test_package_smoke.py`

**Interfaces:**
- Consumes: Python `>=3.10,<3.13`.
- Produces: importable package and `python -m sc_mv_dmer.cli` entrypoint.

- [ ] **Step 1: Write the failing smoke test**

```python
from sc_mv_dmer import __version__
from sc_mv_dmer.cli import app_name

def test_package_identity() -> None:
    assert __version__ == "0.1.0"
    assert app_name() == "sc-mv-dmer"
```

- [ ] **Step 2: Run it and verify RED**

Run: `python -m pytest tests/test_package_smoke.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'sc_mv_dmer'`.

- [ ] **Step 3: Add the minimal package**

`pyproject.toml` must use a `src` layout, expose dependency ranges from this plan, and configure pytest with `testpaths=["tests"]`. `__init__.py` exports `__version__ = "0.1.0"`; `cli.py` defines `app_name() -> str` returning `"sc-mv-dmer"`.

- [ ] **Step 4: Run GREEN**

Run: `python -m pytest tests/test_package_smoke.py -v`

Expected: PASS.

- [ ] **Step 5: Initialize Git only after the user explicitly starts implementation**

Run: `git rev-parse --is-inside-work-tree`

Expected before initialization: non-zero exit because the workspace is not yet a Git repository. After explicit implementation authorization, run `git init` once, then re-run the check and expect `true`. Do not run `git init` while merely reading or reviewing this plan.

- [ ] **Step 6: Future commit boundary**

```powershell
git add pyproject.toml .gitignore src/sc_mv_dmer/__init__.py src/sc_mv_dmer/cli.py tests/test_package_smoke.py
git commit -m "build: establish python project baseline"
```

### Task 2: 实现 stable identity 与双配置 hash

**Files:**
- Create: `src/sc_mv_dmer/foundation/canonical.py`
- Create: `src/sc_mv_dmer/foundation/config.py`
- Create: `src/sc_mv_dmer/foundation/identity.py`
- Create: `schemas/resolved_config.schema.json`
- Create: `configs/research/defaults.yaml`
- Create: `tests/foundation/test_canonical_config.py`
- Create: `tests/foundation/test_stable_identity.py`

**Interfaces:**
- Consumes: layered YAML paths and runtime CLI override mapping.
- Produces: `resolve_config(paths: Sequence[Path], overrides: Mapping[str, object], run_mode: RunMode) -> ResolvedConfigSnapshot`; `stable_id(namespace: str, parts: Sequence[str]) -> str`.

- [ ] **Step 1: Write deterministic-hash tests**

```python
def test_semantic_hash_ignores_runtime_and_absolute_paths() -> None:
    a = resolve_fixture(device="cuda:0", data_root="C:/data")
    b = resolve_fixture(device="cpu", data_root="D:/mirror")
    assert a.semantic_config_hash == b.semantic_config_hash
    assert a.resolved_config_hash != b.resolved_config_hash

def test_formal_semantic_override_is_rejected() -> None:
    with pytest.raises(FormalOverrideError):
        resolve_config(FILES, {"markov.shared_A": False}, RunMode.FORMAL)
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/foundation/test_canonical_config.py tests/foundation/test_stable_identity.py -v`

Expected: FAIL because resolver and identity functions do not exist.

- [ ] **Step 3: Implement canonical JSON and provenance chain**

Use UTF-8, recursively sorted keys, stable enum/bool/null/number encoding and SHA-256. Persist source-file path relative to repository, source-file SHA-256, merge order, CLI overrides, `config_schema_version`, `research_spec_version`, resolved canonical JSON, both hashes and `semantic_diff.json`. Reject unknown keys and runtime-profile writes to semantic fields.

- [ ] **Step 4: Run GREEN and property tests**

Run: `python -m pytest tests/foundation/test_canonical_config.py tests/foundation/test_stable_identity.py -v`

Expected: PASS including Windows/Linux path-relocation cases.

- [ ] **Step 5: Future commit boundary**

```powershell
git add src/sc_mv_dmer/foundation configs/research/defaults.yaml schemas/resolved_config.schema.json tests/foundation
git commit -m "feat: add canonical config and stable identity"
```

### Task 3: 实现 Run、Artifact、Gate 与 append-only validity

**Files:**
- Create: `src/sc_mv_dmer/foundation/manifests.py`
- Create: `src/sc_mv_dmer/foundation/lifecycle.py`
- Create: `src/sc_mv_dmer/foundation/validity.py`
- Create: `schemas/run_spec.schema.json`
- Create: `schemas/run_manifest.schema.json`
- Create: `schemas/invalidation_record.schema.json`
- Create: `schemas/gate_evaluation.schema.json`
- Create: `tests/foundation/test_run_lifecycle.py`
- Create: `tests/foundation/test_effective_validity.py`

**Interfaces:**
- Consumes: `ResolvedConfigSnapshot`, dependency refs and artifact checksums.
- Produces: `start_run(spec: RunSpec) -> ActiveRun`; `finalize_run(active: ActiveRun, terminal: TerminalState) -> RunManifest`; `resolve_effective_validity(node_id: str, registry: Registry) -> EffectiveValidity`.

- [ ] **Step 1: Write lifecycle and propagation tests**

```python
def test_terminal_run_cannot_be_reopened(tmp_path: Path) -> None:
    final = finalize_fixture(tmp_path, TerminalState.SUCCEEDED)
    with pytest.raises(TerminalRecordMutationError):
        reopen_run(final.run_id)

def test_invalidated_predecessor_makes_downstream_stale() -> None:
    registry = chain_fixture("gate-a", "stage-b")
    registry.invalidate("gate-a", reason="bad evidence")
    assert resolve_effective_validity("stage-b", registry).status == "STALE_DEPENDENCY"
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/foundation/test_run_lifecycle.py tests/foundation/test_effective_validity.py -v`

Expected: FAIL with missing manifest/lifecycle modules.

- [ ] **Step 3: Implement immutable terminal and dependency graph rules**

Run manifests record Git commit/dirty state, data/split/cache identities, seed, config hashes/snapshot/provenance, environment, command, metrics, checkpoint/artifact checksums, parent/continuation refs and terminal reason. Gate attempts store historical verdict, gate version/attempt/evaluation ID, evidence commit, evidence-bundle hash and manual/automatic authority; effective verdict is derived, never written back.

- [ ] **Step 4: Run GREEN**

Run: `python -m pytest tests/foundation/test_run_lifecycle.py tests/foundation/test_effective_validity.py -v`

Expected: PASS for INVALIDATED and STALE_DEPENDENCY propagation and superseding Gate definitions.

- [ ] **Step 5: Future commit boundary**

```powershell
git add src/sc_mv_dmer/foundation schemas tests/foundation
git commit -m "feat: enforce immutable run and evidence lifecycle"
```

### Task 4: 编译 Variant Capability 与 Stage Authority

**Files:**
- Create: `src/sc_mv_dmer/foundation/capabilities.py`
- Create: `src/sc_mv_dmer/foundation/stages.py`
- Create: `configs/catalog/variants.yaml`
- Create: `configs/catalog/stages.yaml`
- Create: `schemas/compiled_execution_capability.schema.json`
- Create: `tests/foundation/test_capability_compiler.py`
- Create: `tests/foundation/test_stage_authority.py`
- Generate during execution: `reports/preflight/stage-authority-coverage.json`

**Interfaces:**
- Consumes: frozen variant definition, semantic diff, dependency closure, stage/context profile.
- Produces: `compile_capabilities(...) -> CompiledExecutionCapabilityManifest`; `audit_stage_authority(...) -> StageAuthorityCoverage`.

- [ ] **Step 1: Write fail-closed compiler tests**

```python
def test_a10_has_no_generation_or_parser_capability() -> None:
    compiled = compile_fixture("A10_VA_ONLY_LLM", "stage2")
    assert compiled.generation == "NOT_APPLICABLE"
    assert compiled.parser == "NOT_APPLICABLE"

def test_unknown_stage_row_blocks_catalog_publication() -> None:
    with pytest.raises(FrozenContractCoverageError):
        publish_stage_catalog(incomplete_m0_m20_source())
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/foundation/test_capability_compiler.py tests/foundation/test_stage_authority.py -v`

Expected: FAIL because compiler is absent.

- [ ] **Step 3: Implement explicit compilation**

Represent scientific existence separately from context activation and readiness. Cover module/Sensor/Event producer-consumer-visibility/prompt/Qwen/LM/parser/generation/cache/stage/phase/final inference/handoff/parameter/Gate/artifact roles. If current authority does not enumerate a machine-transcribable M0–M20 row, generate the coverage report and stop with `IMPLEMENTATION_BLOCKED_BY_FROZEN_CONTRACT: SGA-M0M20-EV-3R+`.

- [ ] **Step 4: Run GREEN or record the prescribed blocker**

Run: `python -m pytest tests/foundation/test_capability_compiler.py tests/foundation/test_stage_authority.py -v`

Expected: unit fixtures PASS; real catalog publication either PASS with full authority references or emits the exact blocker without inferred rows.

- [ ] **Step 5: Future commit boundary**

```powershell
git add src/sc_mv_dmer/foundation configs/catalog schemas tests/foundation reports/preflight
git commit -m "feat: compile variant and stage capabilities"
```

### Task 5: 发现 DEAM source 并绑定 stable dataset/song/sample identity

**Files:**
- Create: `src/sc_mv_dmer/data/discovery.py`
- Create: `src/sc_mv_dmer/data/manifests.py`
- Create: `configs/datasets/deam.yaml`
- Create: `schemas/dataset_manifest.schema.json`
- Create: `tests/data/test_deam_discovery.py`
- Generate during execution: `reports/preflight/deam-source-inventory.json`
- Generate during execution: `manifests/datasets/deam-primary-v1.json`

**Interfaces:**
- Consumes: configured `data_root: Path` and source files.
- Produces: `discover_deam(config: DatasetConfig, data_root: Path) -> SourceInventory`; `bind_deam_primary(inventory: SourceInventory) -> DatasetManifest`.

- [ ] **Step 1: Write relocation and population tests**

```python
def test_data_root_relocation_preserves_ids(tmp_path: Path) -> None:
    left, right = mirrored_deam_fixtures(tmp_path)
    assert bind_deam_primary(discover_deam(CONFIG, left)).semantic_id == bind_deam_primary(discover_deam(CONFIG, right)).semantic_id

def test_primary_population_requires_exactly_1744_excerpts() -> None:
    with pytest.raises(PopulationMismatchError):
        bind_deam_primary(inventory_with_count(1743))
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/data/test_deam_discovery.py -v`

Expected: FAIL because discovery is absent.

- [ ] **Step 3: Implement read-only discovery and binding**

Inventory all source-relative roles and SHA-256 values. Never copy, rename, repair or delete source data. Derive stable IDs from registered dataset version and logical keys; exclude local absolute root. Duplicate logical IDs, missing audio/annotation pairs and count mismatch fail closed.

- [ ] **Step 4: Run GREEN; then execute PRE-FLIGHT DISCOVERY**

Run: `python -m pytest tests/data/test_deam_discovery.py -v`

Run: `python -m sc_mv_dmer.cli discover-data --dataset deam --data-root $env:SC_MV_DMER_DATA_ROOT --output reports/preflight/deam-source-inventory.json`

Expected: tests PASS; discovery either emits a checksum-complete 1,744-record inventory or a machine-readable non-PASS report. `SC_MV_DMER_DATA_ROOT` is supplied by the execution environment and is not committed.

- [ ] **Step 5: Future commit boundary**

```powershell
git add src/sc_mv_dmer/data configs/datasets/deam.yaml schemas/dataset_manifest.schema.json tests/data
git commit -m "feat: bind stable DEAM dataset identities"
```

### Task 6: 验证冻结 split、annotation、target domain 与 normalization

**Files:**
- Create: `src/sc_mv_dmer/data/splits.py`
- Create: `src/sc_mv_dmer/data/annotations.py`
- Create: `schemas/split_manifest.schema.json`
- Create: `tests/data/test_frozen_split.py`
- Create: `tests/data/test_annotation_audit.py`
- Generate during execution: `manifests/splits/deam-primary-song-80-10-10-v1.json`
- Generate during execution: `evidence/data/deam-annotation-audit-v1.json`

**Interfaces:**
- Consumes: bound DEAM manifest and an existing frozen split authority/artifact.
- Produces: `verify_frozen_split(...) -> SplitVerification`; `audit_annotations(...) -> AnnotationAudit`.

- [ ] **Step 1: Write membership and train-only tests**

```python
def test_seed_never_changes_split_membership() -> None:
    base = verify_fixture(seed=52826381)
    assert all(verify_fixture(seed=s).membership_hash == base.membership_hash for s in S5)

def test_target_normalization_uses_train_only() -> None:
    audit = audit_fixture()
    assert audit.normalization_source_splits == ("train",)
    assert audit.test_used_for_selection is False
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/data/test_frozen_split.py tests/data/test_annotation_audit.py -v`

Expected: FAIL because split and audit modules are absent.

- [ ] **Step 3: Implement verification without regeneration**

Verify disjointness, union coverage, song-level grouping, exact artifact hash, target finite/domain/time alignment, missingness, `TargetValidityMask`, train-only constant baseline and train-only normalization provenance. If the approved split artifact is not present, stop with a source-binding blocker; never synthesize a replacement split from S5 or observed results.

- [ ] **Step 4: Run GREEN**

Run: `python -m pytest tests/data/test_frozen_split.py tests/data/test_annotation_audit.py -v`

Expected: PASS for fixtures; real binding must preserve the registered hash.

- [ ] **Step 5: Future commit boundary**

```powershell
git add src/sc_mv_dmer/data schemas tests/data manifests/splits evidence/data
git commit -m "feat: verify frozen split and annotation contract"
```

### Task 7: 实现 formal/debug pre-flight 与 Step-0 evidence

**Files:**
- Create: `src/sc_mv_dmer/foundation/preflight.py`
- Create: `src/sc_mv_dmer/foundation/qualification.py`
- Create: `tests/foundation/test_formal_preflight.py`
- Create: `tests/foundation/test_step0_qualification.py`
- Generate during execution: `evidence/qualifications/step-0-foundation-preflight.json`

**Interfaces:**
- Consumes: config, Git state, dataset/split/audit, capability/stage and validity registries.
- Produces: `run_preflight(context: PreflightContext) -> PreflightResult`; `build_step0_qualification(...) -> FoundationPreflightBundle`.

- [ ] **Step 1: Write aggregate fail-closed tests**

```python
@pytest.mark.parametrize("fault", ["dirty_git", "split_hash", "semantic_cli", "invalid_dependency", "unregistered_data"])
def test_formal_preflight_rejects_fault(fault: str) -> None:
    assert run_preflight(context_with_fault(fault)).allowed is False

def test_debug_semantic_override_is_not_paper_eligible() -> None:
    result = run_preflight(debug_context_with_semantic_override())
    assert result.allowed is True
    assert result.paper_eligible is False
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/foundation/test_formal_preflight.py tests/foundation/test_step0_qualification.py -v`

Expected: FAIL because pre-flight is absent.

- [ ] **Step 3: Implement aggregate qualification**

Require clean Git for formal, exact data/split/config/schema versions, registered artifact identities, effective predecessor validity and allowed runtime overrides. Evidence bundle closes exactly `MACHINE_READABLE_MANIFEST_CONFIG_BINDING`, `CLEAN_GIT_PROVENANCE_PREFLIGHT`, `OQ-DATA-DEAM-TARGET-ARTIFACT-BINDING`, and `DEAM_ANNOTATION_AUDIT`; any manual criterion remains pending human authority.

- [ ] **Step 4: Run full Step-0 verification**

Run: `python -m pytest tests/foundation tests/data -v`

Run: `python -m sc_mv_dmer.cli qualify-step --step 0 --mode formal --output evidence/qualifications/step-0-foundation-preflight.json`

Expected: tests PASS; qualification returns VALID only when all four blocker records and dependencies are complete.

- [ ] **Step 5: Future commit boundary**

```powershell
git add src/sc_mv_dmer/foundation tests/foundation evidence/qualifications
git commit -m "feat: close foundation preflight dependencies"
```
