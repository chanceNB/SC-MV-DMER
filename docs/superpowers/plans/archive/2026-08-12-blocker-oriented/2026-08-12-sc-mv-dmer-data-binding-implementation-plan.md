# SC-MV-DMER 数据绑定实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task；在每个已命名的 fail-closed boundary 停止。

**Goal:**（目标） 在不改变冻结科研语义的前提下，为 E0/E1/E2/E4/E6 分阶段提供 DEAM 与 PMEmo source artifact、identity、label domain、mask、split membership 和 evaluation-input coverage；E1.1 只依赖 exact probe 与至少一条 registered real DEAM 45s sample，不得把完整 population/split、PMEmo 或 secondary blocker合并为 MERT帧率打印的 predecessor。

**Architecture:**（架构） 本文件保留 P1 support tag，但按实验阶段切片消费 P0 identity/canonicalization/manifest/validity API。Dataset adapter 输出类型化 song/sample record；audit command 在不修改输入的前提下派生 evidence；coverage mask 与 target-validity mask 始终分离。E0先形成 `E0_E1_PROBE_READY`，只绑定E1.1真实输入；完整1,744 population与冻结split可并行准备，但到E2 formal extraction/cache前才强制关闭。DEAM target evidence按第一次使用边界释放，PMEmo只在E6 secondary evaluation前成为强制依赖。

**Tech Stack:**（技术栈） Python 3.10+、Pydantic、PyYAML、NumPy、pandas、pytest、JSON Schema、SHA-256。

## Global Constraints（全局约束）

- Data root 必须可配置；本地绝对路径不得进入 semantic identity。
- Primary population 是冻结的 1,744 个 DEAM excerpt，使用唯一、不可变的歌曲级 80/10/10 split；58 首 long song 是独立 auxiliary evaluation data。
- 绑定 canonical 2 Hz `TimeGrid`；PMEmo 使用冻结的 chorus-clip-local timeline、本地 `t=0` 和经审计的 first-15-second 规则。
- S5 seed 不得改变 dataset membership；test 不参与 selection 或 Gate decision。
- Timing、domain、mask、normalization provenance 和 hash 必须来自 evidence，不能依靠假设。
- 缺失或矛盾的来源事实必须 fail closed；不得替换数据或静默 coercion。
- 不执行 `git init`；下列 commit command 仅是获得授权后的未来执行边界。

## Experiment-stage placement（实验阶段定位）

| Support task | experiment_stage_owner | Entry Gate | Required-before | can_prepare_early | Exit Gate |
|---|---|---|---|---:|---|
| Tasks 1–2：identity、DEAM inventory、registered real 45s probe binding | E0（为 E1.1 提供输入） | P0 identity/manifest/evidence API可用 | E1.1 real MERT frame-rate print | false | `E0_E1_PROBE_READY` |
| Task 3：1,744 primary population + frozen split | E0后开始、E1并行准备 | P0 stable ID与source inventory可用 | E2 formal extraction/cache及后续完整population use | true | `E2_DEAM_PRIMARY_SPLIT_READY` |
| Task 4：annotation/normalization | E4 primary；coverage部分为E6 secondary | DEAM primary/split已绑定 | annotation audit在E4；secondary coverage在E6 | true | 分别终结，不组成单一AND |
| Task 5：PMEmo | E6 / PMEmo secondary | P0 stable ID即可开始准备 | 仅E6 PMEmo zero-shot | true | `PMEMO_ARTIFACT_HASH_DOMAIN_AUDIT` current-effective |
| Task 6：qualification assembly | E0、E2、E4、E6分阶段 | 对应scope输入完成 | 对应exact step | true | scope-local evidence bundle |

完整1,744 population/split、`PMEMO_ARTIFACT_HASH_DOMAIN_AUDIT` 和 `EVALUATION_INPUT_COVERAGE_EXECUTABLE_BINDING` 均不得阻塞 E1.1 MERT帧率打印；完整population/split必须在E2 formal work前关闭，PMEmo/secondary evidence只在其对应后期边界生效。Task 3/5可以 `PARALLEL_PREPARATION`，但 `PREPARATION != EXPERIMENT_STAGE_ENTRY`。

## Interface Contract（接口合同）

- `make_dataset_id(registration: DatasetRegistration) -> StableId`
- `make_song_id(dataset_id: StableId, logical_song_key: str) -> StableId`
- `make_sample_id(song_id: StableId, excerpt_key: str) -> StableId`
- `discover_dataset(config: DatasetConfig, data_root: Path) -> SourceInventory`
- `bind_rg01_real_probe(inventory: SourceInventory, probe_spec: ExactProbeSpec) -> RegisteredProbeManifest`
- `bind_deam_primary(inventory: SourceInventory) -> DatasetManifest`
- `verify_frozen_split(dataset: DatasetManifest, split: SplitManifest) -> SplitVerification`
- `audit_annotations(dataset: DatasetManifest, split: SplitManifest) -> AnnotationAudit`
- `compile_evaluation_coverage(dataset: DatasetManifest, audit: AnnotationAudit) -> EvaluationCoverage`
- `bind_pmemo(inventory: SourceInventory) -> tuple[DatasetManifest, ArtifactDomainAudit]`
- `build_data_qualification(registry: EvidenceRegistry) -> EvidenceBundle`

---

### Task 1（任务 1）：建立类型化 Dataset Identity 与 Manifest Contract

**Files:**（文件）
- Create（创建）：`src/sc_mv_dmer/data/identity.py`
- Create（创建）：`src/sc_mv_dmer/data/manifests.py`
- Create（创建）：`schemas/dataset_manifest.schema.json`
- Create（创建）：`tests/data/test_dataset_identity.py`
- Create（创建）：`tests/data/test_dataset_manifest.py`

**Test first:**（先写测试） 断言 `dataset_id`、`song_id` 和 `sample_id` 在 data-root relocation 后保持不变；duplicate identity、缺失 source hash、semantic absolute path 和 finalized-manifest mutation 均被拒绝。

**Run:**（执行） `python -m pytest tests/data/test_dataset_identity.py tests/data/test_dataset_manifest.py -q`

**Expected initial result:**（预期初始结果） FAIL，因为 identity 和 manifest module 尚不存在。

**Minimal implementation（最小实现）：** 使用 registered dataset version 和 source-relative logical key 构造 typed identity。Manifest 包含 immutable schema/version、source hash、population role、record identity、label/timing provenance 和 checksum，并复用 P0 canonical JSON/finalization API。

**Pass condition（通过条件）：** 测试 PASS；relocation 不改变 identity；overwrite attempt 抛出 typed error。

**Commit boundary:**（提交边界） `git add src/sc_mv_dmer/data schemas/dataset_manifest.schema.json tests/data && git commit -m "feat: define immutable dataset identity contracts"`

### Task 2（任务 2）：发现并绑定 DEAM Source Inventory

**Files:**（文件）
- Create（创建）：`src/sc_mv_dmer/data/discovery.py`
- Create（创建）：`src/sc_mv_dmer/data/probes.py`
- Create（创建）：`src/sc_mv_dmer/cli/discover_data.py`
- Create（创建）：`src/sc_mv_dmer/cli/bind_rg01_probe.py`
- Create（创建）：`configs/datasets/deam.yaml`
- Create（创建）：`configs/probes/rg01-exact-45s.yaml`
- Create（创建）：`schemas/registered_probe_manifest.schema.json`
- Create（创建）：`tests/data/test_deam_discovery.py`
- Create（创建）：`tests/data/test_rg01_probe_binding.py`
- Generate during execution（执行时生成）：`reports/preflight/deam_source_inventory.json`
- Generate during execution（执行时生成）：`manifests/probes/deam-rg01-real-45s-v1.json`

**PRE-FLIGHT DISCOVERY TASK:**（执行前发现任务） 对配置的 DEAM root 执行只读 inventory，记录 relative path、byte size、SHA-256、media/annotation role 和 duplicate-content group。不得复制、重命名或规范化 source。为 E1.1 额外绑定至少一条预先登记、非结果驱动选择的真实 DEAM 45s sample，保存 stable sample identity、source hash、segment/crop/resample provenance，并验证 exact probe specification 为 `45s @ 24,000Hz = 1,080,000 samples`；此步骤不加载 MERT、不执行 forward，也不需要完整1,744 population/split。

**Run:**（执行）
```powershell
if (-not $env:SC_MV_DMER_DATA_ROOT) { throw 'DEAM_SOURCE_ARTIFACTS_UNAVAILABLE: SC_MV_DMER_DATA_ROOT is unset' }
python -m sc_mv_dmer.cli.discover_data --dataset deam --config configs/datasets/deam.yaml --output reports/preflight/deam_source_inventory.json
python -m sc_mv_dmer.cli.bind_rg01_probe --inventory reports/preflight/deam_source_inventory.json --probe-spec configs/probes/rg01-exact-45s.yaml --output manifests/probes/deam-rg01-real-45s-v1.json
python -m pytest tests/data/test_deam_discovery.py tests/data/test_rg01_probe_binding.py -q
```

**Expected initial result:**（预期初始结果） Discovery 实现前测试 FAIL；未配置 artifact 时命令以 `DEAM_SOURCE_ARTIFACTS_UNAVAILABLE` 退出，且不写 formal manifest。

**Decision artifact:**（决策产物） `reports/preflight/deam_source_inventory.json` 与 `manifests/probes/deam-rg01-real-45s-v1.json`；E1.1 evidence以checksum引用两者。Probe manifest只授权真实帧率测量输入，不证明完整primary population/split已绑定。

**Minimal implementation:**（最小实现） 仅对允许的 source-relative pattern 做确定性 inventory；拒绝 ambiguous role、unhashed input、inaccessible file 和逃逸 data root 的路径。

**Pass condition:**（通过条件） 测试 PASS；重复扫描得到相同 inventory且不含绝对路径；registered real probe具有稳定 identity/source hash、exact 45s/24kHz/1,080,000-sample specification和完整provenance。未满足时只阻塞E1.1，不生成替代样本。

**Commit boundary:**（提交边界） `git add src/sc_mv_dmer/data/discovery.py src/sc_mv_dmer/data/probes.py src/sc_mv_dmer/cli/discover_data.py src/sc_mv_dmer/cli/bind_rg01_probe.py configs/datasets/deam.yaml configs/probes/rg01-exact-45s.yaml schemas/registered_probe_manifest.schema.json tests/data/test_deam_discovery.py tests/data/test_rg01_probe_binding.py && git commit -m "feat: add fail-closed dataset and RG01 probe discovery"`

### Task 3（任务 3）：绑定 Primary Population 与冻结 Split

**Files:**（文件）
- Create（创建）：`src/sc_mv_dmer/data/deam.py`
- Create（创建）：`src/sc_mv_dmer/data/splits.py`
- Create（创建）：`schemas/split_manifest.schema.json`
- Create（创建）：`tests/data/test_deam_population.py`
- Create（创建）：`tests/data/test_primary_split.py`
- Generate during execution（执行时生成）：`manifests/datasets/deam-primary-v1.json`
- Bind during execution（执行时绑定）：`manifests/splits/deam-primary-song-80-10-10-v1.json`

**Test first:**（先写测试） 断言 primary excerpt 恰为 1,744 个；train/validation/test 按歌曲互斥、完整且只分配一次；membership 不受 S5 影响；58 首 long song 全部排除。Registered split hash 不匹配时必须阻止 formal use。

**Run:**（执行）
```powershell
python -m pytest tests/data/test_deam_population.py tests/data/test_primary_split.py -q
python -m sc_mv_dmer.cli.bind_dataset --dataset deam-primary --inventory reports/preflight/deam_source_inventory.json --output manifests/datasets/deam-primary-v1.json
python -m sc_mv_dmer.cli.bind_split --dataset-manifest manifests/datasets/deam-primary-v1.json --registered-split manifests/splits/deam-primary-song-80-10-10-v1.json --verify-only
```

**Expected initial result:**（预期初始结果） Adapter 实现前测试 FAIL；若 approved membership/hash 不可用，则以 `FROZEN_SPLIT_MANIFEST_UNAVAILABLE` 退出，且不得生成 replacement split。

**Minimal implementation:**（最小实现） 解析稳定 excerpt/song identity，将 primary 与 long-song population 分类并验证既有 split。Split 登记后，formal-run capability 中不得存在 split creation。

**Scheduling constraint:** Task 3可在E1.1–E1.4期间并行准备，但不得成为E1.1 real-forward predecessor；其 current-effective completion是E2 formal extraction/cache及后续完整primary population use的前置。该时序调整不改变冻结的歌曲级80/10/10 membership或manifest identity。

**Pass condition:**（通过条件） Verifier 报告精确 1,744-record coverage、零 song overlap 和 registered split hash。

**Commit boundary:**（提交边界） `git add src/sc_mv_dmer/data/deam.py src/sc_mv_dmer/data/splits.py schemas/split_manifest.schema.json tests/data && git commit -m "feat: bind frozen DEAM population and split"`

### Task 4（任务 4）：审计 DEAM Annotation、Normalization 与 Coverage Mask

**Files:**（文件）
- Create（创建）：`src/sc_mv_dmer/data/annotations.py`
- Create（创建）：`src/sc_mv_dmer/data/coverage.py`
- Create（创建）：`src/sc_mv_dmer/data/audit.py`
- Create（创建）：`schemas/evaluation_coverage.schema.json`
- Create（创建）：`tests/data/test_deam_annotation_audit.py`
- Create（创建）：`tests/data/test_evaluation_coverage.py`
- Generate during execution（执行时生成）：`reports/data/deam_annotation_audit.json`
- Generate during execution（执行时生成）：`reports/data/deam_evaluation_coverage.json`

**Test first:**（先写测试） 覆盖 monotonic 2 Hz timing、finite V/A、duration、label domain、train-only normalization/constant baseline，以及相互独立的 `InputCoverageMask`、`TargetValidityMask`、Sensor validity 和 Event validity。Padding 是 invalid coverage，绝不能表示 valid silence。

**Run:**（执行）
```powershell
python -m pytest tests/data/test_deam_annotation_audit.py tests/data/test_evaluation_coverage.py -q
python -m sc_mv_dmer.cli.audit_annotations --dataset-manifest manifests/datasets/deam-primary-v1.json --split-manifest manifests/splits/deam-primary-song-80-10-10-v1.json --output reports/data/deam_annotation_audit.json
python -m sc_mv_dmer.cli.bind_coverage --dataset-manifest manifests/datasets/deam-primary-v1.json --annotation-audit reports/data/deam_annotation_audit.json --output reports/data/deam_evaluation_coverage.json
```

**Expected initial result:**（预期初始结果） Auditor 实现前 FAIL；真实执行遇到 implicit domain、time origin、mask 或 boundary 时，以 `DEAM_ANNOTATION_AUDIT_INCOMPLETE` 或 `EVALUATION_INPUT_COVERAGE_UNBOUND` 退出。

**Minimal implementation:**（最小实现） 输出 typed annotation/coverage record；train-only statistic 必须记录 split/hash provenance；提供具名 applicable-mask intersection，并拒绝 formal metric 中匿名组合 boolean array。

**Pass condition:**（通过条件） 测试 PASS；每个 record 和 grid point 均有明确归属，不推断或静默删除任何 row。

**Commit boundary:**（提交边界） `git add src/sc_mv_dmer/data/annotations.py src/sc_mv_dmer/data/coverage.py src/sc_mv_dmer/data/audit.py schemas/evaluation_coverage.schema.json tests/data && git commit -m "feat: bind DEAM target and coverage provenance"`

### Task 5（任务 5）：发现并绑定 PMEmo Auxiliary Artifact

**Files:**（文件）
- Create（创建）：`src/sc_mv_dmer/data/pmemo.py`
- Create（创建）：`configs/datasets/pmemo.yaml`
- Create（创建）：`tests/data/test_pmemo_binding.py`
- Generate during execution（执行时生成）：`reports/preflight/pmemo_source_inventory.json`
- Generate during execution（执行时生成）：`reports/data/pmemo_artifact_domain_audit.json`
- Generate during execution（执行时生成）：`manifests/datasets/pmemo-2019-eval-v1.json`

**PRE-FLIGHT DISCOVERY TASK:**（执行前发现任务） 从真实 release artifact 确定精确 hash domain、release provenance、clip/time origin、annotation domain 和 first-15-second coverage。

**Run:**（执行）
```powershell
python -m sc_mv_dmer.cli.discover_data --dataset pmemo --config configs/datasets/pmemo.yaml --output reports/preflight/pmemo_source_inventory.json
python -m pytest tests/data/test_pmemo_binding.py -q
python -m sc_mv_dmer.cli.bind_dataset --dataset pmemo-2019-eval --inventory reports/preflight/pmemo_source_inventory.json --output manifests/datasets/pmemo-2019-eval-v1.json --audit-output reports/data/pmemo_artifact_domain_audit.json
```

**Expected initial result:**（预期初始结果） Adapter 实现前测试 FAIL；缺少 authoritative artifact 或 hash/domain boundary 未解决时，以 `PMEMO_ARTIFACT_HASH_DOMAIN_UNRESOLVED` 退出，且不创建 formal manifest。

**Decision artifact:**（决策产物） Audit 必须指出每个 checksum 覆盖的物理 byte，并在应用冻结 VA mapping 前证明 `[0,1]` source-domain handling。

**Minimal implementation:**（最小实现） 实现 auxiliary-only adapter，使用 local clip time `t=0`，显式审计前 15 秒，验证 domain，并保存 deterministic mapping provenance。

**Pass condition:**（通过条件） Formal PMEmo manifest 绑定真实 hash、release identity、audited domain 和 coverage evidence，且没有 training role。

**Commit boundary:**（提交边界） `git add src/sc_mv_dmer/data/pmemo.py configs/datasets/pmemo.yaml tests/data/test_pmemo_binding.py && git commit -m "feat: bind PMEmo evaluation artifacts"`

### Task 6（任务 6）：生成 Stage-scoped Data Blocker Evidence

**Files:**（文件）
- Create（创建）：`src/sc_mv_dmer/data/qualification.py`
- Create（创建）：`tests/data/test_data_qualification_bundle.py`
- Generate during execution（执行时生成）：`evidence/p1/e0-e1-probe-qualification.json`
- Generate during execution（执行时生成）：`evidence/p1/e2-primary-data-qualification.json`
- Generate during execution（执行时生成）：`evidence/p1/e4-annotation-qualification.json`
- Generate during execution（执行时生成）：`evidence/p1/e6-secondary-data-qualification.json`

**Test first:**（先写测试） 验证四个scope互不错误耦合：E0/E1 probe bundle只要求 machine-readable evidence primitive、MERT upstream identity、exact 45s probe、至少一条registered real DEAM sample identity/hash和provenance；E2 primary bundle要求完整1,744 population与registered frozen split；E4 bundle要求 `DEAM_ANNOTATION_AUDIT`；E6 bundle要求 `PMEMO_ARTIFACT_HASH_DOMAIN_AUDIT` 与 `EVALUATION_INPUT_COVERAGE_EXECUTABLE_BINDING`。拒绝 missing、superseded 或 invalidated dependency，并断言E2/E4/E6 blocker缺失不会使E0 probe bundle失败。

**Run:**（执行）
```powershell
python -m pytest tests/data -q
python -m sc_mv_dmer.data.qualification --scope e0-e1-probe --output evidence/p1/e0-e1-probe-qualification.json
python -m sc_mv_dmer.data.qualification --scope e2-primary --output evidence/p1/e2-primary-data-qualification.json
python -m sc_mv_dmer.data.qualification --scope e4-annotation --output evidence/p1/e4-annotation-qualification.json
python -m sc_mv_dmer.data.qualification --scope e6-secondary --output evidence/p1/e6-secondary-data-qualification.json
python -m sc_mv_dmer.governance.verify_evidence evidence/p1/e0-e1-probe-qualification.json
```

**Expected initial result:**（预期初始结果） 每个scope只在自身真实 artifact/evidence 完整前失败；完整primary/split未完成时E2 bundle失败但E0 probe bundle仍可VALID；PMEmo不可用时E6 bundle失败，但E0 probe/E2 primary bundle不受影响。

**Minimal implementation:**（最小实现） 按scope构建append-only bundle，包含experiment stage、blocker ID、historical outcome、effective validity、command、environment、dependency hash和output checksum；禁止生成隐含“全P1必须PASS”的aggregate field。

**Pass condition:**（通过条件） 对应scope verifier只报告其required closure，且无unregistered input或跨scope假依赖。

**Commit boundary:**（提交边界） `git add src/sc_mv_dmer/data/qualification.py tests/data/test_data_qualification_bundle.py && git commit -m "test: qualify data binding closure"`

## Stage-scoped Exit Gates（分阶段退出门槛）

| Exit Gate | Required evidence | Unlock | 不要求 |
|---|---|---|---|
| `E0_E1_PROBE_READY` | machine-readable evidence primitive、pinned MERT upstream/processor identity、exact 45s/24kHz/1,080,000-sample probe、至少一条registered real DEAM sample identity/source hash与provenance | E1.1 real frame-rate print | 完整1,744 population/split、annotation、PMEmo、Qwen、Sensor、CoT、A5/A6 |
| `E2_DEAM_PRIMARY_SPLIT_READY` | DEAM stable identities、1,744 primary population、registered song-level 80/10/10 split、source checksums | E2 formal extraction/cache及后续完整population use | PMEmo、long58 coverage、Qwen、CoT |
| `E4_DEAM_TARGET_READY` | current-effective `DEAM_ANNOTATION_AUDIT`、target domain/mask/normalization provenance | E4 formal No-LLM target/metric use | PMEmo |
| `E6_SECONDARY_DATA_READY` | exact long58/PMEmo所需 coverage；PMEmo artifact/hash/domain audit | 对应E6 secondary row | 其他E6 ablation rows |

P1不再拥有一个全局Exit Gate。无法认证某一source artifact时，只阻塞明确消费该artifact的exact step，不改变Foundation contract，也不反向阻塞无依赖的早期primary阶段。
