# 执行治理与机器可读 Foundation 实施计划

> **For agentic workers / 供代理执行者：** 必须使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans`，按任务逐项实施本计划。使用复选框（`- [ ]`）跟踪执行状态。

**Goal:**（目标） 建立类型化、机器可读的治理内核，使后续每个 SC-MV-DMER run、artifact、capability、stage 和 Gate dependency 均可审计。

**Architecture:**（架构） 一个轻依赖 Python package 统一负责 canonical ID、Pydantic manifest model、确定性 config hash、capability compilation、stage binding、append-only effective validity 以及 formal/debug pre-flight。人类可读 Foundation 文档继续作为研究语义 Source of Truth；生成的 JSON Schema 和版本化 YAML catalog 只是其可执行映射。

**Tech Stack:**（技术栈） Python 3.10.11；Pydantic `>=2.8,<3`；PyYAML `>=6.0.2,<7`；pytest `>=8.3,<9`；Hypothesis `>=6.112,<7`；标准库 hashlib/json/pathlib/subprocess。

## Global Constraints（全局约束）

- 本实施包不得重新解释 Foundation 科学语义；unknown key 和无权威引用的 contract value 必须 fail closed。
- Canonical JSON 使用 UTF-8、固定 key 排序、允许范围内的路径规范化和 SHA-256。
- `semantic_config_hash` 排除 runtime/machine/path 字段；`resolved_config_hash` 覆盖完整 resolved config。
- Formal CLI override 仅限 runtime 字段；semantic CLI override 强制 `run_mode=debug` 且禁止进入 paper aggregation。
- Terminal run、Gate attempt、artifact 和 invalidation 均为 append-only。
- 仅在取得明确执行授权后开始实施；本规划阶段不执行 `git init`。

---

### Task 1（任务 1）：建立仓库与 Python 基线

**Files:**（文件）
- Create（创建）：`.gitignore`
- Create（创建）：`pyproject.toml`
- Create（创建）：`src/sc_mv_dmer/__init__.py`
- Create（创建）：`tests/test_package_smoke.py`

**Interfaces:**（接口）
- Consumes（输入）：当前非 Git workspace 和已批准的执行授权。
- Produces（输出）：可导入的 `sc_mv_dmer` package 和 baseline Git provenance。

- [ ] **步骤 1：验证预期的 bootstrap 前失败**

Run（执行）：`powershell -NoProfile -Command "if (Test-Path .git) { exit 0 } else { Write-Error 'not a git repository'; exit 1 }"`

Expected（预期）：以 `not a git repository` 失败。

- [ ] **步骤 2：创建最小 package metadata 和 smoke test**

`pyproject.toml` 使用 `hatchling.build`，project name 为 `sc-mv-dmer`，`requires-python=">=3.10,<3.13"`，core dependency 采用本计划 Tech Stack 中的版本范围，package root 为 `src`，pytest `testpaths=["tests"]`。`src/sc_mv_dmer/__init__.py` 写入 `__version__ = "0.1.0"`；smoke test 导入 package 并断言该版本。

- [ ] **步骤 3：运行 smoke test**

Run（执行）：`python -m pip install -e ".[dev]"; python -m pytest tests/test_package_smoke.py -v`

Expected（预期）：PASS，1 个 test。

- [ ] **步骤 4：仅在获得执行授权后 bootstrap Git**

Run（执行）：`git init -b main; git add .gitignore pyproject.toml src/sc_mv_dmer/__init__.py tests/test_package_smoke.py docs; git commit -m "chore: establish governed project baseline"`

Expected（预期）：已生成 root commit，且 `git status --short` 为空。

### Task 2（任务 2）：稳定身份与 Canonical Config Hash

**Files:**（文件）
- Create（创建）：`src/sc_mv_dmer/contracts/ids.py`
- Create（创建）：`src/sc_mv_dmer/contracts/config.py`
- Create（创建）：`tests/contracts/test_ids.py`
- Create（创建）：`tests/contracts/test_config_hashes.py`

**Interfaces:**（接口）
- Consumes（输入）：mapping 和 scalar value。
- Produces（输出）：`StableId`、`ContractRef`、`canonical_json_bytes(value) -> bytes`、`semantic_config_hash(config) -> str`、`resolved_config_hash(config) -> str`。

- [ ] **步骤 1：先写失败的 identity/hash 测试**

```python
def test_semantic_hash_ignores_runtime_and_paths():
    a = {"dataset": {"id": "DEAM/v1"}, "runtime": {"device": "cuda:0"}, "data_root": "D:/a"}
    b = {"dataset": {"id": "DEAM/v1"}, "runtime": {"device": "cpu"}, "data_root": "E:/b"}
    assert semantic_config_hash(a) == semantic_config_hash(b)
    assert resolved_config_hash(a) != resolved_config_hash(b)

def test_stable_id_rejects_absolute_path():
    with pytest.raises(ValueError, match="path-dependent"):
        StableId(kind="sample", value="D:/data/song.wav")
```

- [ ] **步骤 2：验证 red state**

Run（执行）：`python -m pytest tests/contracts/test_ids.py tests/contracts/test_config_hashes.py -v`

Expected（预期）：FAIL，因为 `sc_mv_dmer.contracts.ids` 和 `.config` 尚不存在。

- [ ] **步骤 3：实现确定性类型**

为 `StableId(kind, value, version)` 和 `ContractRef(contract_id, version, checksum)` 实现 frozen Pydantic model。实现递归 canonicalization：固定 key 排序，统一 bool/null/number 表示，使用 UTF-8 JSON separators `(',', ':')`，semantic denylist 为 `{runtime, data_root, output_dir, timestamp, machine_name, gpu_type, checkpointing}`，输出 SHA-256 hex。

- [ ] **步骤 4：验证 green state 与 property stability**

Run（执行）：`python -m pytest tests/contracts/test_ids.py tests/contracts/test_config_hashes.py -v`

Expected（预期）：PASS；YAML key 顺序和 Windows/Linux 本地路径变化不改变 semantic identity。

- [ ] **步骤 5：Commit**

Run（执行）：`git add src/sc_mv_dmer/contracts tests/contracts; git commit -m "feat: add stable identities and config hashing"`

### Task 3（任务 3）：Manifest 与 Provenance 原语

**Files:**（文件）
- Create（创建）：`src/sc_mv_dmer/contracts/manifests.py`
- Create（创建）：`src/sc_mv_dmer/contracts/checksums.py`
- Create（创建）：`tests/contracts/test_manifests.py`
- Create（创建）：`schemas/run_spec.schema.json`
- Create（创建）：`schemas/run_manifest.schema.json`
- Create（创建）：`schemas/artifact_manifest.schema.json`

**Interfaces:**（接口）
- Produces（输出）：`ArtifactRef`、`ConfigSource`、`RunSpec`、`RunManifest`、`EvidenceBundle`、`sha256_file(path)`、`finalize_manifest(model, path)`。
- Invariant（不变量）：finalization 采用 exclusive creation，拒绝 overwrite。

- [ ] **步骤 1：先写失败的 immutability 测试**

```python
def test_final_manifest_cannot_be_overwritten(tmp_path):
    path = tmp_path / "run_manifest.json"
    finalize_manifest(valid_terminal_manifest(), path)
    with pytest.raises(FileExistsError):
        finalize_manifest(valid_terminal_manifest(), path)
```

- [ ] **步骤 2：验证 red state**

Run（执行）：`python -m pytest tests/contracts/test_manifests.py -v`

Expected（预期）：因 manifest module 缺失而 FAIL。

- [ ] **步骤 3：实现两阶段 run 与 artifact model**

字段必须覆盖 run ID、parent/continuation/source 关系、Git commit/dirty state、dataset/split/cache fingerprint、seed、semantic/resolved hash、config provenance chain、environment、command、metrics、checkpoint/artifact checksum、terminal status、failure evidence、schema/research version。Final record 使用 `open(path, 'x', encoding='utf-8')`。

- [ ] **步骤 4：导出 schema 并验证 round-trip**

Run（执行）：`python -c "from pathlib import Path; from sc_mv_dmer.contracts.manifests import export_schemas; export_schemas(Path('schemas'))"; python -m pytest tests/contracts/test_manifests.py -v`

Expected（预期）：schema 文件确定性一致，测试 PASS。

- [ ] **步骤 5：Commit**

Run（执行）：`git add src/sc_mv_dmer/contracts tests/contracts schemas; git commit -m "feat: add immutable manifest primitives"`

### Task 4（任务 4）：Capability 与 Execution Manifest Compiler

**Files:**（文件）
- Create（创建）：`src/sc_mv_dmer/contracts/capabilities.py`
- Create（创建）：`configs/catalog/variants.yaml`
- Create（创建）：`tests/contracts/test_capability_compiler.py`

**Interfaces:**（接口）
- Produces（输出）：`VariantCapabilityProfile`、`CompiledExecutionCapabilityManifest`、`ExecutionReadiness`、`FinalEventDisposition`、`compile_execution_manifest(variant, stage, context, dependencies)`。
- Consumes（输入）：A1–A13 精确 catalog value 和 `ContractRef`。

- [ ] **步骤 1：先写失败的 closed-world 测试**

```python
def test_a6_has_replay_without_local_sensor():
    result = compile_fixture("A6-v1", "stage2", "final_va")
    assert result.sensor is Capability.NOT_APPLICABLE_BY_DESIGN
    assert result.event_disposition is FinalEventDisposition.EXTERNAL_REPLAY_MODEL_VISIBLE

def test_a5_event_exists_but_is_not_visible():
    result = compile_fixture("A5-v1", "stage2", "final_generation")
    assert result.event_artifact is Activation.ACTIVE
    assert result.model_visible_event is Activation.NOT_APPLICABLE
```

- [ ] **步骤 2：验证 red state**

Run（执行）：`python -m pytest tests/contracts/test_capability_compiler.py -v`

Expected（预期）：因 compiler 缺失而 FAIL。

- [ ] **步骤 3：实现 enum、typed profile 与精确 closure**

拒绝 unknown variant、undeclared inheritance、将 blocked 当作 zero capability、以及 runtime mutation。编码 15 个 executable variant 和非 executable A8 parent；编译 module、Sensor/Event producer/consumer/visibility、prompt、Qwen、LM head、parser、generation、stage、phase、handoff、RG-08 与 forbidden role。

- [ ] **步骤 4：验证全部 catalog profile**

Run（执行）：`python -m pytest tests/contracts/test_capability_compiler.py -v`

Expected（预期）：A1–A13 fixture 与 fail-closed case 全部 PASS。

- [ ] **步骤 5：Commit**

Run（执行）：`git add src/sc_mv_dmer/contracts/capabilities.py configs/catalog/variants.yaml tests/contracts/test_capability_compiler.py; git commit -m "feat: compile variant execution capabilities"`

### Task 5（任务 5）：Append-only Effective Validity

**Files:**（文件）
- Create（创建）：`src/sc_mv_dmer/contracts/validity.py`
- Create（创建）：`tests/contracts/test_effective_validity.py`
- Create（创建）：`schemas/invalidation_record.schema.json`

**Interfaces:**（接口）
- Produces（输出）：`InvalidationRecord`、`SupersessionRecord`、`ValidityState`、`resolve_effective_validity(node_id, graph, registry)`。

- [ ] **步骤 1：先写失败的 propagation 测试**

```python
def test_invalid_predecessor_marks_downstream_stale_not_failed():
    state = resolve_fixture(invalidated="gate:RG-02-attempt-1")
    assert state["stage:M3"] is ValidityState.STALE_DEPENDENCY
    assert historical_verdict("stage:M3") == "SUCCEEDED"
```

- [ ] **步骤 2：验证 red state**

Run（执行）：`python -m pytest tests/contracts/test_effective_validity.py -v`

Expected（预期）：因 validity resolver 缺失而 FAIL。

- [ ] **步骤 3：实现 graph-local propagation**

返回 `VALID`、`INVALIDATED` 或 `STALE_DEPENDENCY`，并附 invalidated source、direct edge、reason path 和 replacement evidence。不得修改任何 historical record。

- [ ] **步骤 4：验证 green state**

Run（执行）：`python -m pytest tests/contracts/test_effective_validity.py -v`

Expected（预期）：direct invalidation、version supersession 和 unaffected branch case 均 PASS。

- [ ] **步骤 5：Commit**

Run（执行）：`git add src/sc_mv_dmer/contracts/validity.py tests/contracts/test_effective_validity.py schemas/invalidation_record.schema.json; git commit -m "feat: resolve dependency-aware effective validity"`

### Task 6（任务 6）：Stage Definition 与 M0–M20 Authority Coverage

**Files:**（文件）
- Create（创建）：`src/sc_mv_dmer/contracts/stages.py`
- Create（创建）：`configs/catalog/stages.yaml`
- Create（创建）：`tools/audit_stage_authority.py`
- Create（创建）：`tests/contracts/test_stage_catalog.py`
- Create at execution（执行时创建）：`reports/preflight/sga_m0_m20_source_coverage.json`

**Interfaces:**（接口）
- Produces（输出）：`TrainingStageDefinition`、`StageTransitionBinding`、`audit_stage_sources(docs) -> StageSourceCoverage`。
- Failure: 缺少权威 row 时输出 `IMPLEMENTATION_BLOCKED_BY_FROZEN_CONTRACT: SGA-M0M20-EV-3R+`。

- [ ] **步骤 1：先写失败的 21-stage source coverage 测试**

```python
def test_every_milestone_has_authority_citation():
    result = audit_stage_sources(Path("docs"))
    assert result.milestone_ids == [f"M{i}" for i in range(21)]
    assert result.missing_authority == []
```

- [ ] **步骤 2：运行 discovery test**

Run（执行）：`python -m pytest tests/contracts/test_stage_catalog.py::test_every_milestone_has_authority_citation -v`

Expected（预期）：按当前已检查文档，因缺少权威 stage row 而 FAIL。写入 JSON report，不得自行生成 catalog value。

- [ ] **步骤 3：执行 decision boundary**

如 report 列出 missing row，停止发布 stage catalog，并报告准确缺失字段：`predecessor`、`gate_definition`、`entry`、`exit_evidence`、`next_stage`。只有收到获批 authority artifact 才能恢复；如果届时文档已覆盖全部 row，则逐项转录，并附 source line/checksum citation。

- [ ] **步骤 4：获得 authority 后验证 catalog consistency**

Run（执行）：`python tools/audit_stage_authority.py --docs docs --catalog configs/catalog/stages.yaml --out reports/preflight/sga_m0_m20_source_coverage.json; python -m pytest tests/contracts/test_stage_catalog.py -v`

Expected（预期）：仅当 21 个 row 均有 source citation 且无 inferred value 时 PASS。

- [ ] **步骤 5：Commit**

Run（执行）：`git add src/sc_mv_dmer/contracts/stages.py configs/catalog/stages.yaml tools/audit_stage_authority.py tests/contracts/test_stage_catalog.py reports/preflight/sga_m0_m20_source_coverage.json; git commit -m "feat: bind stage catalog to authoritative sources"`

### Task 7（任务 7）：Formal/Debug Pre-flight 与 CLI Override Firewall

**Files:**（文件）
- Create（创建）：`src/sc_mv_dmer/preflight.py`
- Create（创建）：`src/sc_mv_dmer/cli.py`
- Create（创建）：`configs/research/defaults.yaml`
- Create（创建）：`configs/runtime/local_5060ti.yaml`
- Create（创建）：`tests/test_preflight.py`

**Interfaces:**（接口）
- Produces（输出）：`run_preflight(resolved_config, workspace, registries) -> PreflightResult` 和 CLI `sc-mv-dmer preflight`。

- [ ] **步骤 1：先写失败的 override 与 dirty-workspace 测试**

```python
def test_formal_semantic_cli_override_is_rejected():
    result = preflight_fixture(mode="formal", overrides={"markov.shared_A": False})
    assert result.failure_codes == ["FORMAL_SEMANTIC_OVERRIDE_FORBIDDEN"]

def test_debug_override_is_paper_ineligible():
    result = preflight_fixture(mode="debug", overrides={"markov.shared_A": False})
    assert result.paper_eligible is False
```

- [ ] **步骤 2：验证 red state**

Run（执行）：`python -m pytest tests/test_preflight.py -v`

Expected（预期）：因 preflight module 缺失而 FAIL。

- [ ] **步骤 3：实现 fail-closed validation**

验证 schema、unknown key、source chain、semantic diff、split/feature/model/artifact registration、effective validity、formal override allowlist、Git repository/clean state 和精确 contract version。输出包含 reason code 的机器可读结果。

- [ ] **步骤 4：验证 formal/debug 行为**

Run（执行）：`python -m pytest tests/test_preflight.py -v`

Expected（预期）：PASS；formal dirty state 失败，debug state 明确 paper-ineligible。

- [ ] **步骤 5：Commit**

Run（执行）：`git add src/sc_mv_dmer/preflight.py src/sc_mv_dmer/cli.py configs tests/test_preflight.py; git commit -m "feat: enforce formal and debug preflight"`

### Task 8（任务 8）：P0 Blocker Closure Evidence

**Files:**（文件）
- Create（创建）：`tools/audit_foundation_binding.py`
- Create（创建）：`tests/test_foundation_binding_audit.py`
- Create at execution（执行时创建）：`reports/qualification/p0_foundation_binding.json`

**Interfaces:**（接口）
- Produces（输出）：覆盖 `MACHINE_READABLE_MANIFEST_CONFIG_BINDING` 和 `CLEAN_GIT_PROVENANCE_PREFLIGHT` 的单一 evidence report。

- [ ] **步骤 1：先写失败的 report completeness 测试**

断言 report 包含 schema checksum、catalog checksum、config hash、formal/debug fixture、Git clean/dirty fixture、stage-source coverage、effective-validity case、command、environment 和 evidence checksum。

- [ ] **步骤 2：验证 red state**

Run（执行）：`python -m pytest tests/test_foundation_binding_audit.py -v`

Expected（预期）：因 report generator 缺失而 FAIL。

- [ ] **步骤 3：实现并运行 audit**

Run（执行）：`python tools/audit_foundation_binding.py --out reports/qualification/p0_foundation_binding.json`

Expected（预期）：已实现的 machine binding PASS；若 M0–M20 source coverage 仍不完整，则以精确 SGA failure 报告 P0 blocked，不得关闭该 blocker。

- [ ] **步骤 4：运行 P0 regression suite**

Run（执行）：`python -m pytest tests/contracts tests/test_preflight.py tests/test_foundation_binding_audit.py -v`

Expected（预期）：除被显式断言的 SGA source-coverage blocked case 外全部 PASS。

- [ ] **步骤 5：Commit**

Run（执行）：`git add tools/audit_foundation_binding.py tests/test_foundation_binding_audit.py reports/qualification/p0_foundation_binding.json; git commit -m "test: qualify machine-readable foundation"`
