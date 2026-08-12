# SC-MV-DMER Qwen 绑定实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task；保留全部冻结 topology 和 fail-closed boundary。

**Goal:**（目标） 为 E5 的 canonical Qwen2.5-7B pipeline 分阶段 pin upstream、验证 Stage-1/Stage-2 loader 与 tokenizer budget，并将 A10-specific hidden-backbone qualification隔离到E6/A10；不得让A10 blocker阻塞A1 E5。

**Architecture:**（架构） 本文件保留P2 support tag。Versioned upstream manifest统一拥有model/tokenizer identity和hash；capability-compiled loader生成彼此独立的Stage-1、Stage-2 QLoRA和A10 hidden-backbone graph。Evidence按E5.1、E5.9和E6/A10分别终结，不生成要求A10与canonical A1共同PASS的全局P2 Gate。

**Tech Stack:**（技术栈） Python 3.10+、PyTorch、Transformers、PEFT、适用时的 bitsandbytes、safetensors、Pydantic、pytest、SHA-256。

## Global Constraints（全局约束）

- 仅使用冻结的 7B primary architecture；14B、DPO、Qwen2.5-Audio 和 contingency model 均 deferred 且不阻塞。
- 不自动下载模型；真实 qualification 只消费本地配置且已先 pin bytes/revision 的 snapshot。
- Stage 1 无 LoRA；Stage 2 遵循冻结的 `LQ7-R16A32-ZD-NF4-3R+` 与 `QKB-SPMS-3R++`。
- A10 直接加载 hidden backbone，无 LM head、无 generation path，并满足冻结 projection/equivalence criterion。
- Generated decoding 遵循 `GEN-DTB-VA90-3R++`；ANSWER parsing 与 full-grid evidence 遵循 `RG-10-v2`，其中 `T=90`。
- Tokenizer、chat template、special-token map 和 revision 均属于 upstream identity。
- 不执行 `git init`；下列 commit 仅是未来授权后的执行边界。

## Experiment-stage placement（实验阶段定位）

| Support task | experiment_stage_owner | Entry Gate | Required-before | can_prepare_early | Exit Gate |
|---|---|---|---|---:|---|
| Task 1 upstream manifest | E5.1 | E0 upstream primitives | E5.2 RG-08 Stage-1 P1/P2 | true | `E5_QWEN_UPSTREAM_READY` |
| Tasks 2–3 Stage-1 topology/loader | E5.1/E5.2 | pinned upstream | E5.2/E5.3 | true | `E5_STAGE1_LOADER_READY` |
| Tasks 2–3 Stage-2 topology/loader | E5.9 | H3、exact CoT artifact applicability可编译 | E5.9 RG-08 Stage-2 P1/P3 | true（代码）；false（final evidence） | `E5_STAGE2_LOADER_READY` |
| Task 5 tokenizer/ANSWER budget | E5.1/E5.5/E5.14 | pinned tokenizer | RG-07、CoT、final generation/RG-10 | true | `TOKENIZER_ANSWER_BUDGET_QUALIFICATION` |
| Task 4 A10 equivalence | E6/A10 | A1 Qwen binding可复用；A10 capability已编译 | 仅E6 A10 entry | true | `A10_PROJECTION_EQUIVALENCE_QUALIFICATION` |
| Task 6 qualification assembly | E5 core + E6/A10分离 | 对应role evidence | exact stage/variant | true | scope-local bundle |

`A10_PROJECTION_EQUIVALENCE_QUALIFICATION` 不在 canonical A1 E5 critical path。Stage-2 loader代码可提前准备，但其exact artifact/topology qualification不得冒充E5.9 entry PASS。

## Interface Contract（接口合同）

- `discover_upstream(config: UpstreamConfig, model_root: Path) -> UpstreamInventory`
- `bind_upstream(inventory: UpstreamInventory) -> UpstreamModelManifest`
- `compile_qwen_topology(profile: VariantCapabilityProfile, role: LoaderRole) -> QwenLoadSpec`
- `load_qwen(spec: QwenLoadSpec, upstream: UpstreamModelManifest) -> LoadedModelBundle`
- `qualify_loader(bundle: LoadedModelBundle, spec: QwenLoadSpec) -> LoaderQualification`
- `load_a10_backbone(upstream: UpstreamModelManifest) -> A10BackboneBundle`
- `qualify_a10_equivalence(reference: LoadedModelBundle, a10: A10BackboneBundle) -> ProjectionEquivalenceEvidence`
- `qualify_token_budget(fixtures: PromptFixtureSet, tokenizer: TokenizerBinding) -> TokenBudgetEvidence`
- `build_qwen_qualification(registry: EvidenceRegistry) -> EvidenceBundle`

---

### Task 1（任务 1）：定义并发现 Upstream Model Manifest

**Files:**（文件）
- Create（创建）：`src/sc_mv_dmer/models/upstream.py`
- Create（创建）：`schemas/upstream_model_manifest.schema.json`
- Create（创建）：`configs/upstream/qwen-7b-primary.yaml`
- Create（创建）：`tests/models/test_upstream_manifest.py`
- Generate during execution（执行时生成）：`reports/preflight/qwen_snapshot_inventory.json`
- Generate during execution（执行时生成）：`manifests/upstream/qwen-7b-primary-v1.json`

**PRE-FLIGHT DISCOVERY TASK:**（执行前发现任务） 对配置的本地 snapshot 和 tokenizer 执行 inventory，记录精确 model identifier、可用时的 immutable revision/commit、source-relative file role、file SHA-256、config/tokenizer/chat-template hash、license metadata 和 library compatibility input。

**Run:**（执行）
```powershell
if (-not $env:SC_MV_DMER_MODEL_ROOT) { throw 'MODEL_SNAPSHOT_UNAVAILABLE: SC_MV_DMER_MODEL_ROOT is unset' }
python -m sc_mv_dmer.models.upstream discover --config configs/upstream/qwen-7b-primary.yaml --output reports/preflight/qwen_snapshot_inventory.json
python -m pytest tests/models/test_upstream_manifest.py -q
python -m sc_mv_dmer.models.upstream bind --inventory reports/preflight/qwen_snapshot_inventory.json --output manifests/upstream/qwen-7b-primary-v1.json
```

**Expected initial result:**（预期初始结果） Manifest module 实现前测试 FAIL；缺少本地 byte 时以 `MODEL_SNAPSHOT_UNAVAILABLE` 退出；revision ambiguous 或 hash 不完整时以 `PINNED_UPSTREAM_IDENTITY_UNRESOLVED` 退出。

**Decision artifact:**（决策产物） `manifests/upstream/qwen-7b-primary-v1.json`；后续 loader 不得从 mutable cache alias 推断 upstream revision。

**Minimal implementation:**（最小实现） 实现 schema validation、deterministic inventory、hash verification 和 immutable binding；本地绝对 root 只进入 execution provenance。

**Pass condition:**（通过条件） 所有必需 model/tokenizer byte 均已 pin，重复 inventory 生成相同 manifest hash。

**Commit boundary:**（提交边界） `git add src/sc_mv_dmer/models/upstream.py schemas/upstream_model_manifest.schema.json configs/upstream/qwen-7b-primary.yaml tests/models/test_upstream_manifest.py && git commit -m "feat: bind pinned Qwen upstream manifest"`

### Task 2（任务 2）：从 Capability 编译冻结 Loader Topology

**Files:**（文件）
- Create（创建）：`src/sc_mv_dmer/models/capabilities.py`
- Create（创建）：`src/sc_mv_dmer/models/qwen_loader.py`
- Create（创建）：`src/sc_mv_dmer/models/topology.py`
- Create（创建）：`tests/models/test_qwen_topology.py`

**Test first:**（先写测试） 断言 Stage 1 无 LoRA 且 frozen/trainable role 正确；Stage 2 为 rank 16、alpha 32、LoRA dropout 0、NF4、frozen base，并具有精确 target-module binding；forbidden 或 unmatched target module 必须失败。Topology 必须从 variant capability profile 派生，不能依靠 ad-hoc flag。

**Run:**（执行） `python -m pytest tests/models/test_qwen_topology.py -q`

**Expected initial result:**（预期初始结果） FAIL，因为 topology compiler 和 loader 尚不存在。

**Minimal implementation:**（最小实现） 以 typed `VariantCapabilityProfile` 为输入，解析 Stage-1/Stage-2 load specification，对 pinned config 执行 target-module discovery，并在分配 model weight 前输出 machine-readable topology report。

**Pass condition:**（通过条件） Synthetic topology fixture PASS；unknown role fail closed；compiled Stage-1/Stage-2 report 与冻结合同一致。

**Commit boundary:**（提交边界） `git add src/sc_mv_dmer/models/capabilities.py src/sc_mv_dmer/models/qwen_loader.py src/sc_mv_dmer/models/topology.py tests/models/test_qwen_topology.py && git commit -m "feat: compile frozen Qwen loader topologies"`

### Task 3（任务 3）：执行 Qwen Loader Qualification

**Files:**（文件）
- Create（创建）：`src/sc_mv_dmer/models/qualify_loader.py`
- Create（创建）：`schemas/model_loader_qualification.schema.json`
- Create（创建）：`tests/models/test_loader_qualification.py`
- Generate during execution（执行时生成）：`evidence/p2/qwen-loader-qualification.json`

**Test first:**（先写测试） 使用 tiny local fixture，验证精确 dtype、quantization、base freezing、LoRA trainability、embedding handling、preparation order、module count 和 finite forward evidence。Mock 只能验证 control flow，不能关闭真实 blocker。

**Run:**（执行）
```powershell
python -m pytest tests/models/test_loader_qualification.py -q
python -m sc_mv_dmer.models.qualify_loader --upstream-manifest manifests/upstream/qwen-7b-primary-v1.json --roles stage1,stage2 --output evidence/p2/qwen-loader-qualification.json
```

**Expected initial result:**（预期初始结果） Qualification 实现前测试 FAIL；如本地 hardware/software 无法实例化每个 required role，则以 `QWEN_EXECUTABLE_LOADER_QUALIFICATION_INCOMPLETE` 退出。Partial observation 保留为 evidence，但不能关闭 blocker。

**Minimal implementation:**（最小实现） 实例化每个 compiled role，记录 resolved library/device/dtype state，检查 parameter/module，执行最小无训练 forward，并终结 immutable checksummed evidence。

**Pass condition:**（通过条件） 真实 qualification bundle 在同一 pinned upstream manifest 下验证全部适用 topology，输出 finite，trainability count 精确。

**Commit boundary:**（提交边界） `git add src/sc_mv_dmer/models/qualify_loader.py schemas/model_loader_qualification.schema.json tests/models/test_loader_qualification.py && git commit -m "test: qualify executable Qwen loading"`

### Task 4（任务 4）：验证 A10 Hidden-Backbone Equivalence Boundary

**Files:**（文件）
- Create（创建）：`src/sc_mv_dmer/models/a10_backbone.py`
- Create（创建）：`tests/models/test_a10_backbone.py`
- Generate during execution（执行时生成）：`evidence/p2/a10-projection-equivalence.json`

**Test first:**（先写测试） 断言 A10 只暴露 hidden state 和 VA head；任何 LM head、decoding、CoT、ANSWER 或 parser path 均不可达。在相同输入和冻结 shared weight 下，按冻结 numerical tolerance 比较指定 hidden representation 与 reference projection，并记录 comparison domain。

**Run:**（执行）
```powershell
python -m pytest tests/models/test_a10_backbone.py -q
python -m sc_mv_dmer.models.a10_backbone qualify --upstream-manifest manifests/upstream/qwen-7b-primary-v1.json --output evidence/p2/a10-projection-equivalence.json
```

**Expected initial result:**（预期初始结果） 在 A10 direct loading/comparison 实现前 FAIL。若缺少 frozen tolerance/source binding，以 `IMPLEMENTATION_BLOCKED_BY_FROZEN_CONTRACT: A10_PROJECTION_EQUIVALENCE` 退出，不得发明 threshold。

**Minimal implementation:**（最小实现） 使用 `A10-HBL-DSP-3R++` 定义的 direct hidden-backbone API，加载相同 snapshot byte，显式标识 projection point，并序列化 raw tensor checksum 与 computed deviation。

**Pass condition:**（通过条件） A10 topology 结构隔离，且 executable equivalence 满足已冻结 tolerance。

**Commit boundary:**（提交边界） `git add src/sc_mv_dmer/models/a10_backbone.py tests/models/test_a10_backbone.py && git commit -m "test: qualify A10 hidden-backbone equivalence"`

### Task 5（任务 5）：绑定 Tokenizer 与 ANSWER Budget

**Files:**（文件）
- Create（创建）：`src/sc_mv_dmer/prompts/tokenizer_budget.py`
- Create（创建）：`schemas/tokenizer_budget_qualification.schema.json`
- Create（创建）：`tests/prompts/test_tokenizer_budget.py`
- Generate during execution（执行时生成）：`evidence/p2/tokenizer-answer-budget-qualification.json`

**Test first:**（先写测试） 断言 canonical prompt serializer 使用 pinned tokenizer/template，分别计算 wrapper/content，绑定 full-grid `T=90`，保留冻结 `B_think=300`，测量 registered ANSWER budget，并在 truncation 或 template drift 时失败。RG-10-v2 parsing 覆盖完整 TimeGrid。

**Run:**（执行）
```powershell
python -m pytest tests/prompts/test_tokenizer_budget.py -q
python -m sc_mv_dmer.prompts.tokenizer_budget qualify --upstream-manifest manifests/upstream/qwen-7b-primary-v1.json --foundation-contract FOUNDATION_CONTRACT_INDEX.md --output evidence/p2/tokenizer-answer-budget-qualification.json
```

**Expected initial result:**（预期初始结果） Serializer 实现前 FAIL。若当前冻结文档未绑定可执行 ANSWER budget 值或 canonical fixture set，输出 `IMPLEMENTATION_BLOCKED_BY_FROZEN_CONTRACT: TOKENIZER_ANSWER_BUDGET`，并列出准确缺失 authority citation。

**Minimal implementation:**（最小实现） 实现确定性 serialization/token accounting，记录 fixture hash、per-field maximum、truncation check 和 tokenizer/template provenance；不得根据 formal outcome 调整 budget。

**Pass condition:**（通过条件） 每个 canonical fixture 均符合适用冻结 budget，无 hidden truncation，token count report 可复现。

**Commit boundary:**（提交边界） `git add src/sc_mv_dmer/prompts/tokenizer_budget.py schemas/tokenizer_budget_qualification.schema.json tests/prompts/test_tokenizer_budget.py && git commit -m "test: bind tokenizer and answer budgets"`

### Task 6（任务 6）：生成 Stage/Variant-scoped Qwen Binding Evidence

**Files:**（文件）
- Create（创建）：`src/sc_mv_dmer/models/qualification.py`
- Create（创建）：`tests/models/test_qwen_qualification_bundle.py`
- Generate during execution（执行时生成）：`evidence/p2/e5-qwen-core-qualification.json`
- Generate during execution（执行时生成）：`evidence/p2/e5-stage2-loader-qualification.json`
- Generate during execution（执行时生成）：`evidence/p2/e6-a10-qualification.json`

**Test first:**（先写测试） E5 core bundle必须包含 `PINNED_QWEN_UPSTREAM_MANIFEST`、Stage-1 applicable `QWEN_EXECUTABLE_LOADER_QUALIFICATION` 和 `TOKENIZER_ANSWER_BUDGET_QUALIFICATION`；E5 Stage-2 bundle只接受exact Stage-2 topology/role evidence；E6 A10 bundle单独要求 `A10_PROJECTION_EQUIVALENCE_QUALIFICATION`。断言A10缺失不会使E5 core bundle失败；拒绝仅fixture evidence、缺失upstream hash或invalidated prerequisite。

**Run:**（执行）
```powershell
python -m pytest tests/models tests/prompts -q
python -m sc_mv_dmer.models.qualification --scope e5-core --output evidence/p2/e5-qwen-core-qualification.json
python -m sc_mv_dmer.models.qualification --scope e5-stage2 --output evidence/p2/e5-stage2-loader-qualification.json
python -m sc_mv_dmer.models.qualification --scope e6-a10 --output evidence/p2/e6-a10-qualification.json
python -m sc_mv_dmer.governance.verify_evidence evidence/p2/e5-qwen-core-qualification.json
```

**Expected initial result:**（预期初始结果） 每个scope在自身required executable record有效前保持non-PASS；A10尚未执行时E5 core仍可PASS。

**Minimal implementation:**（最小实现） 使用P0 API按stage/variant scope组装带effective validity和dependency hash的append-only bundle；禁止写入全P2 aggregate PASS。

**Pass condition:**（通过条件） 对应scope verifier报告其required closure，全部绑定同一upstream manifest且无topology drift；不存在跨scope假依赖。

**Commit boundary:**（提交边界） `git add src/sc_mv_dmer/models/qualification.py tests/models/test_qwen_qualification_bundle.py && git commit -m "test: qualify Qwen binding closure"`

## Stage/Variant-scoped Exit Gates（分阶段退出门槛）

| Exit Gate | Required evidence | Unlock | 不要求 |
|---|---|---|---|
| `E5_QWEN_CORE_READY` | pinned upstream、Stage-1 loader、tokenizer/ANSWER budget | E5.2/E5.3与E5.5 | A10 equivalence |
| `E5_STAGE2_LOADER_READY` | exact Stage-2 QKB/LQ7 topology/loader evidence | E5.9 RG-08 Stage-2 qualification | A10 |
| `E6_A10_QWEN_READY` | direct hidden-backbone isolation/equivalence | E6 A10 | 其他ablation rows |

P2不再拥有一个全局Exit Gate。缺少model byte或frozen executable authority只阻塞消费该role的exact step；不得自动下载/替换model、增加tolerance或改变topology。
