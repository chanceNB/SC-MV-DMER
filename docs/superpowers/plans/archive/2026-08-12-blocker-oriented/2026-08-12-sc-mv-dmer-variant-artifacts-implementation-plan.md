# SC-MV-DMER 变体与 Artifact 绑定实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task；始终保持 canonical-base semantic diff 和 explicit dependency closure。

**Goal:**（目标） 为E5 canonical A1在Stage-2前提供Event/RG-07、deterministic CoT partition和immutable CoT artifact，并将A5 firewall/A6 replay严格定位到E6对应ablation prerequisite；不得让A5/A6阻塞A1 E1–E5。

**Architecture:**（架构） 本文件保留P4 support tag。Typed `EventSequence`和artifact manifest输入统一canonical prompt builder；canonical A1 Event/CoT tasks在E5.5–E5.7终结，并由E5.8 RG-09解锁Stage-2。A5/A6仍使用同一底层contract，但其真实qualification/binding只在E6各自entry前强制。

**Tech Stack:**（技术栈） Python 3.10+、Pydantic、Jinja2 或等价 deterministic serializer、NumPy、pytest、JSON Schema、SHA-256。

## Global Constraints（全局约束）

- 每个 variant 都是 canonical base + semantic diff + explicit dependency closure；不得使用 hidden flag bundle。
- 保留 EventSourceCurriculum B+、final inference FI-A+、PB-1+ logical epoch boundary、CS-3+ 和 A1–A13 全部冻结语义。
- CoT coverage selection 由 `COTC-SN21-DSDP-3R+` 管理；RG-07 验证 token length；RG-09 消费 immutable human-review package。
- Event provenance 包含 source kind、Sensor bundle identity/hash、symbolizer version 和 segment boundary。
- A5 shadow data 仅用于 audit，永不 model-visible；A6 使用 paired A1 replay，不能训练或调用本地 Sensor。
- CoT artifact 属于 derived/versioned data；final artifact 不可变，不能原地重新生成。
- 不执行 `git init`；commit 行仅是未来执行边界。

## Experiment-stage placement（实验阶段定位）

| Support task | experiment_stage_owner | Entry Gate | Required-before | can_prepare_early | Exit Gate |
|---|---|---|---|---:|---|
| Task 1 EventSequence common schema | E5.5；replay schema亦服务E6/A6 | E0 artifact primitives | E5.5 materialization | true | schema/serializer qualification |
| Task 2 A5 firewall | E6/A5 | A5 capability + pinned tokenizer | E6 A5 formal entry | true | `A5_MODEL_VISIBLE_INPUT_FIREWALL_QUALIFICATION` |
| Task 3 A6 replay | E6/A6 | current-effective paired A1 same-S5 artifact | E6 A6 formal entry | true（schema）；false（真实binding） | `A6_EVENT_REPLAY_BUNDLE_BINDING` |
| Task 4 CoT partition | E5.6 | DEAM train population/split current-effective | E5.7 | true | partition manifest valid |
| Task 5 A1 phase applicability | E5.5/E5.10/E5.11 | H3/Sensor snapshot identities | Event/CoT materialization和PB运行 | true | compiled artifact allowlist |
| Task 6 CoT materialization | E5.7 | E5.4–E5.6 current-effective | E5.8 RG-09 | true（代码）；false（final artifact） | `COT_ARTIFACT_MATERIALIZATION` |
| Task 7 qualification assembly | E5 CoT core + E6 A5/A6分离 | 对应artifact/variant | exact step | true | scope-local bundle |

E5.7 materialization PASS只解锁E5.8 RG-09，不直接解锁Stage-2。只有current-effective RG-09 PASS才能解锁依赖该CoT artifact的E5.9。A5/A6 fixture可以提前开发，但其blocker不在canonical A1 critical path。

## Interface Contract（接口合同）

- `validate_event_sequence(value: Mapping[str, object]) -> EventSequence`
- `resolve_event_replay(reference: ArtifactRef, validity: ValidityResolver) -> EventReplayBundle`
- `build_prompt(sample: PromptSample, visibility: PromptVisibilityProfile) -> SerializedPrompt`
- `qualify_a5_firewall(pair: PromptPair, tokenizer: TokenizerBinding) -> FirewallEvidence`
- `bind_a6_replay(candidate: PairedA1Candidate) -> EventReplayManifest`
- `partition_cot_coverage(train_ids: Sequence[StableId], rule: CoverageRule) -> CoTPartitionManifest`
- `compile_artifact_applicability(profile: VariantCapabilityProfile, context: StageContext) -> ArtifactApplicability`
- `materialize_cot(partition: CoTPartitionManifest, sources: SourceBundle) -> CoTArtifactManifest`
- `build_artifact_qualification(registry: EvidenceRegistry) -> EvidenceBundle`

---

### Task 1（任务 1）：实现 Typed EventSequence 与 Replay-Bundle Contract

**Files:**（文件）
- Create（创建）：`src/sc_mv_dmer/events/models.py`
- Create（创建）：`src/sc_mv_dmer/events/replay.py`
- Create（创建）：`schemas/event_sequence.schema.json`
- Create（创建）：`schemas/event_replay_bundle.schema.json`
- Create（创建）：`tests/events/test_event_sequence.py`
- Create（创建）：`tests/events/test_replay_bundle.py`

**Test first:**（先写测试） `event_source` 必须属于 `{pseudo_label,sensor_prediction}`；适用时必须提供 Sensor bundle identity/hash、symbolizer rule version、ordered segment boundary、sample/time-grid identity 和 checksum。Replay bundle 必须绑定 paired A1 variant/version、S5 seed、source run/checkpoint、phase、event record 和 dependency hash；mutation 或 cross-seed 使用必须失败。

**Run:**（执行） `python -m pytest tests/events/test_event_sequence.py tests/events/test_replay_bundle.py -q`

**Expected initial result:**（预期初始结果） FAIL，因为 event/replay model 尚不存在。

**Minimal implementation:**（最小实现） 实现 immutable schema、canonical serialization、provenance validator 和 resolver；resolver 必须先通过 P0 验证 effective validity 才能返回 replay record。

**Pass condition:**（通过条件） Valid fixture round-trip 后 hash 稳定，任何 source/seed/checkpoint mismatch 均 fail closed。

**Commit boundary:**（提交边界） `git add src/sc_mv_dmer/events schemas/event_sequence.schema.json schemas/event_replay_bundle.schema.json tests/events && git commit -m "feat: define immutable event replay contracts"`

### Task 2（任务 2）：实现 Canonical Prompt Builder 与 A5 Visibility Firewall

**Files:**（文件）
- Create（创建）：`src/sc_mv_dmer/prompts/builder.py`
- Create（创建）：`src/sc_mv_dmer/prompts/firewall.py`
- Create（创建）：`tests/prompts/test_a5_event_firewall.py`
- Create（创建）：`tests/prompts/test_prompt_serialization.py`
- Generate during execution（执行时生成）：`evidence/p4/a5-model-visible-input-firewall.json`

**Test first:**（先写测试） Render paired A1/A5 input；断言 A5 model-visible byte/token 不含任何 event field、event summary、event-derived count、mask、length、special token 或 formatting side channel。Shadow `EventSequence` 仅保留在非模型 audit evidence 中。非 event 字段必须相同，serialization 必须 canonical。

**Run:**（执行）
```powershell
python -m pytest tests/prompts/test_a5_event_firewall.py tests/prompts/test_prompt_serialization.py -q
python -m sc_mv_dmer.prompts.firewall qualify --variant A5_EVENT_INJECTION_DISABLED --output evidence/p4/a5-model-visible-input-firewall.json
```

**Expected initial result:**（预期初始结果） Prompt/firewall 实现前 FAIL；任何未分类 model-visible field 均以 `A5_MODEL_VISIBLE_INPUT_FIREWALL_VIOLATION` 退出。

**Minimal implementation:**（最小实现） 从 capability profile 编译 typed visibility allowlist 并据此构造 prompt content；比较 serialized byte 与 pinned-tokenizer ID；shadow 与 visible evidence 分开记录。

**Pass condition:**（通过条件） 测试与 qualification 证明 model-visible input 中 event-derived difference 为 0，同时保留 paired audit provenance。

**Commit boundary:**（提交边界） `git add src/sc_mv_dmer/prompts/builder.py src/sc_mv_dmer/prompts/firewall.py tests/prompts && git commit -m "test: qualify A5 prompt visibility firewall"`

### Task 3（任务 3）：绑定 A6 Paired Replay Source

**Files:**（文件）
- Create（创建）：`src/sc_mv_dmer/events/a6_binding.py`
- Create（创建）：`tests/events/test_a6_binding.py`
- Generate during execution（执行时生成）：`manifests/artifacts/a6-event-replay-v1.json`
- Generate during execution（执行时生成）：`evidence/p4/a6-event-replay-binding.json`
- Generate during discovery（发现阶段生成）：`reports/preflight/a6_paired_a1_candidate.json`

**Test first:**（先写测试） 断言 A6 不存在 Sensor head、Sensor loss、Sensor optimizer role 或 local event generation；它必须消费同一 S5 seed、registered curriculum phase 的 paired A1 replay bundle；source checkpoint 和 artifact hash 不可变。缺少 A1 evidence 时阻塞 A6。

**Run:**（执行）
```powershell
python -m pytest tests/events/test_a6_binding.py -q
python -m sc_mv_dmer.events.a6_binding discover --variant A6-v1 --registry manifests/runs --output reports/preflight/a6_paired_a1_candidate.json
python -m sc_mv_dmer.events.a6_binding bind --candidate reports/preflight/a6_paired_a1_candidate.json --output manifests/artifacts/a6-event-replay-v1.json --evidence evidence/p4/a6-event-replay-binding.json
```

**Expected initial result:**（预期初始结果） Binding 实现前 FAIL；无 valid paired A1 artifact 时以 `A6_PAIRED_EVENT_REPLAY_UNAVAILABLE` 退出，绝不生成 local replacement event。

**PRE-FLIGHT DISCOVERY TASK:**（执行前发现任务） 通过 P0 effective-validity API 解析 registered A1 run/checkpoint/phase，并在 binding 前记录完整 candidate dependency graph。

**Minimal implementation:**（最小实现） 验证 paired identity/capability，不复制 source data；生成只含 reference、source hash 和 applicability 的 immutable replay manifest。

**Pass condition:**（通过条件） A6 仅加载 paired reference，全部 local Sensor/event-generation entry point 在结构上不可用。

**Commit boundary:**（提交边界） `git add src/sc_mv_dmer/events/a6_binding.py tests/events/test_a6_binding.py && git commit -m "feat: bind A6 paired event replay"`

### Task 4（任务 4）：实现 Deterministic CoT Coverage Selection

**Files:**（文件）
- Create（创建）：`src/sc_mv_dmer/cot/coverage.py`
- Create（创建）：`schemas/cot_partition_manifest.schema.json`
- Create（创建）：`tests/cot/test_coverage_partition.py`
- Generate during execution（执行时生成）：`manifests/artifacts/cot-partition-v1.json`

**Test first:**（先写测试） 对 eligible training population size `N`，要求 `N_syn = floor(2N/3 + 0.5)`；使用冻结 identity domain 的 deterministic SHA-256 ranking；selection 与 S5 无关；synthetic/non-synthetic partition 完整且互斥；不得包含 validation/test membership。

**Run:**（执行）
```powershell
python -m pytest tests/cot/test_coverage_partition.py -q
python -m sc_mv_dmer.cot.coverage bind --dataset-manifest manifests/datasets/deam-primary-v1.json --split-manifest manifests/splits/deam-primary-song-80-10-10-v1.json --output manifests/artifacts/cot-partition-v1.json
```

**Expected initial result:**（预期初始结果） Partitioning 实现前 FAIL；如 frozen hash-domain component 缺失，以 `IMPLEMENTATION_BLOCKED_BY_FROZEN_CONTRACT: COT_COVERAGE_HASH_DOMAIN` 退出。

**Minimal implementation:**（最小实现） 按冻结 digest rule 对 train-only stable identity 排序，记录 domain input 和精确 boundary，并终结 partition manifest。

**Pass condition:**（通过条件） Golden/real manifest 均 deterministic、disjoint、complete，且跨 S5 seed 不变。

**Commit boundary:**（提交边界） `git add src/sc_mv_dmer/cot/coverage.py schemas/cot_partition_manifest.schema.json tests/cot/test_coverage_partition.py && git commit -m "feat: bind deterministic CoT coverage partition"`

### Task 5（任务 5）：编译 Phase 与 Variant Artifact Applicability

**Files:**（文件）
- Create（创建）：`src/sc_mv_dmer/artifacts/applicability.py`
- Create（创建）：`src/sc_mv_dmer/training/event_phase.py`
- Create（创建）：`tests/artifacts/test_variant_artifact_applicability.py`
- Create（创建）：`tests/training/test_event_phase_boundary.py`

**Test first:**（先写测试） 覆盖 A1 early pseudo/late predicted phase、A2 event-free、A5 shadow-only event、A6 paired replay 和 A8 scoped single-handcrafted-view。验证 PB-1+ 在 logical epoch 3-of-5 切换且不受 physical restart 影响；final FI-A+ 在适用时使用 Stage-2 integrated Sensor prediction。

**Run:**（执行） `python -m pytest tests/artifacts/test_variant_artifact_applicability.py tests/training/test_event_phase_boundary.py -q`

**Expected initial result:**（预期初始结果） Applicability compilation 实现前 FAIL。

**Minimal implementation:**（最小实现） 从 `VariantCapabilityProfile` 编译 required/optional/forbidden artifact role；在 durable `TrainingState` 中表示 phase state；在冻结 logical boundary 强制 source-policy transition。

**Pass condition:**（通过条件） 每个 primary variant fixture 都有唯一、无歧义的 artifact/source policy，forbidden dependency 无法解析。

**Commit boundary:**（提交边界） `git add src/sc_mv_dmer/artifacts/applicability.py src/sc_mv_dmer/training/event_phase.py tests/artifacts tests/training/test_event_phase_boundary.py && git commit -m "feat: compile variant artifact applicability"`

### Task 6（任务 6）：物化 Immutable CoT Artifact 与 Review Package

**Files:**（文件）
- Create（创建）：`src/sc_mv_dmer/cot/materialize.py`
- Create（创建）：`src/sc_mv_dmer/cot/review_package.py`
- Create（创建）：`schemas/cot_artifact_manifest.schema.json`
- Create（创建）：`tests/cot/test_cot_materialization.py`
- Create（创建）：`tests/cot/test_review_package.py`
- Generate during execution（执行时生成）：`manifests/artifacts/cot-synthesis-v1.json`
- Generate during execution（执行时生成）：`evidence/p4/cot-artifact-materialization.json`
- Generate during discovery（发现阶段生成）：`reports/preflight/cot_source_bundle.json`

**Test first:**（先写测试） 必须记录 source Sensor/event provenance、partition identity、prompt/tokenizer/upstream hash、synthesis rule/version、raw/parsed output、coverage、failure reason、artifact checksum 和 immutability。使用 P2 qualification 强制 RG-07 token-length criterion。Review package 暴露冻结 record，但不修改 source artifact。

**Run:**（执行）
```powershell
python -m pytest tests/cot -q
python -m sc_mv_dmer.cot.materialize discover --partition manifests/artifacts/cot-partition-v1.json --artifact-registry manifests/artifacts --output reports/preflight/cot_source_bundle.json
python -m sc_mv_dmer.cot.materialize run --partition manifests/artifacts/cot-partition-v1.json --source-bundle reports/preflight/cot_source_bundle.json --output manifests/artifacts/cot-synthesis-v1.json --evidence evidence/p4/cot-artifact-materialization.json
```

**Expected initial result:**（预期初始结果） Materialization 实现前 unit test FAIL；在 required upstream artifact 和 P2 tokenizer evidence 有效前，真实执行以 `COT_SOURCE_ARTIFACT_UNAVAILABLE` 退出，且不得调用 training。

**Minimal implementation:**（最小实现） 将 record 流式写入 content-addressed immutable shard；仅在完整 validation 后 finalize；生成人工 review 使用的 reference-only package。

**Pass condition:**（通过条件） Artifact完整覆盖冻结partition，通过适用RG-07 check，具备checksum且不可overwrite。该PASS的下一步是RG-09人工artifact review，不是Stage-2 entry。

**Commit boundary:**（提交边界） `git add src/sc_mv_dmer/cot schemas/cot_artifact_manifest.schema.json tests/cot && git commit -m "feat: materialize immutable CoT artifacts"`

### Task 7（任务 7）：生成 Stage/Variant-scoped Artifact Evidence

**Files:**（文件）
- Create（创建）：`src/sc_mv_dmer/artifacts/qualification.py`
- Create（创建）：`tests/artifacts/test_artifact_qualification_bundle.py`
- Generate during execution（执行时生成）：`evidence/p4/e5-cot-artifact-qualification.json`
- Generate during execution（执行时生成）：`evidence/p4/e6-a5-firewall-qualification.json`
- Generate during execution（执行时生成）：`evidence/p4/e6-a6-replay-qualification.json`

**Test first:**（先写测试） E5 CoT bundle只要求`COT_ARTIFACT_MATERIALIZATION`及其Event/RG-07/partition/provenance dependencies；E6 A5 bundle单独要求`A5_MODEL_VISIBLE_INPUT_FIREWALL_QUALIFICATION`；E6 A6 bundle单独要求`A6_EVENT_REPLAY_BUNDLE_BINDING`与paired A1 lineage。断言A5/A6缺失不使E5 CoT bundle失败；拒绝invalid dependency或不完整variant applicability。

**Run:**（执行）
```powershell
python -m pytest tests/events tests/prompts tests/cot tests/artifacts -q
python -m sc_mv_dmer.artifacts.qualification --scope e5-cot --output evidence/p4/e5-cot-artifact-qualification.json
python -m sc_mv_dmer.artifacts.qualification --scope e6-a5 --output evidence/p4/e6-a5-firewall-qualification.json
python -m sc_mv_dmer.artifacts.qualification --scope e6-a6 --output evidence/p4/e6-a6-replay-qualification.json
python -m sc_mv_dmer.governance.verify_evidence evidence/p4/e5-cot-artifact-qualification.json
```

**Expected initial result:**（预期初始结果） 每个scope在自己的真实binding record有效前保持non-PASS；E5 CoT不等待A5/A6。

**Minimal implementation:**（最小实现） 按stage/variant scope组装包含semantic diff、dependency graph、effective validity和checksum的append-only bundle；禁止生成全P4 aggregate PASS。

**Pass condition:**（通过条件） 对应scope verifier报告required closure，并证明无model-visible/provenance leakage或跨scope假依赖。

**Commit boundary:**（提交边界） `git add src/sc_mv_dmer/artifacts/qualification.py tests/artifacts/test_artifact_qualification_bundle.py && git commit -m "test: qualify variant artifact closure"`

## Stage/Variant-scoped Exit Gates（分阶段退出门槛）

| Exit Gate | Required evidence | Unlock | 不要求 |
|---|---|---|---|
| `E5_COT_ARTIFACT_READY_FOR_RG09` | A1 phase Event/RG-07、partition、immutable CoT/ANSWER artifact | E5.8 RG-09 | A5 firewall、A6 replay |
| `E6_A5_ARTIFACT_READY` | A5 visibility firewall + exact A5 semantic diff | E6 A5 | A6 |
| `E6_A6_ARTIFACT_READY` | paired same-S5 A1 replay + no local Sensor proof | E6 A6 | A5 |

P4不再拥有一个全局Exit Gate。缺失A1 source、tokenizer或applicability只阻塞消费该artifact的exact path；相关失败不改变variant语义。
