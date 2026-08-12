# SC-MV-DMER 训练资格验证实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task；始终区分 durable replay state 与 per-invocation recomputation capsule。

**Goal:**（目标） 在不启动科研训练的fixture qualification阶段实现并验证冻结的RNG、replay、sampler、gradient、optimizer、scheduler、accumulation和numerical contract，并将证据插入E3/E4/E5第一次实际使用对应role/profile之前；不得把全部P3任务合并为所有训练阶段的单一前置包。

**Architecture:**（架构） 本文件保留P3 horizontal-support tag。Namespaced RNG service驱动global-permutation exact-once sampler和immutable `TrainingState`；Checkpoint recomputation使用invocation-local CRR capsule。Qualification按`stage_context + variant capability + numeric profile`分别终结，使Sensor、No-LLM、Qwen Stage-1/Stage-2只等待各自适用的证据。

**Tech Stack:**（技术栈） Python 3.10+、PyTorch、NumPy、pytest、Pydantic、safetensors、JSON Schema。

## Global Constraints（全局约束）

- Qualification 只使用 synthetic/minimal fixture；它不是 model training，也不能生成 paper evidence。
- 必须精确保留 `RNG-3+`、`SS-3+`、`TSC-GPESA-3R+`、`CRR-HRSC-3R++`、`SPP-3R`、`MMP-GT-3R+`、`HLAC-HDSA-3R+`、`DLC-P6-DAAI-3R+`、`OSNP-SLAW-3R++` 和 `OSNS-DB10-3R+`。
- CRR invocation capsule 是 ephemeral replay evidence；durable `TrainingState` 只包含 restart-required live state。
- View Dropout 在 checkpointed region 外采样；其 live RNG state 在 durable logical-step boundary 精确 commit 一次，包括 recomputation 情形。
- Frozen Qwen 可以保留在 autograd transit path 中，但 frozen parameter 不得被 optimizer 更新。
- 测试结果不得调整冻结 threshold、loss definition、reduction denominator 或 seed set。
- 不执行 `git init`；commit 行仅是未来执行边界。

## Experiment-stage placement（实验阶段定位）

| Support task | experiment_stage_owner | Entry Gate | Required-before | can_prepare_early | Exit Gate |
|---|---|---|---|---:|---|
| Task 1 RNG/S5 | E3–E6 horizontal | E0 identity API | first formal seeded training at E3 | true | `RNG_PROFILE_READY[context]` |
| Task 2 CRR | E5.2（及任何明确启用checkpoint的 earlier context） | exact checkpoint profile可编译 | E5.3 checkpointed Stage-1；E5.9 Stage-2 | true | `CRR_EXECUTABLE_QUALIFICATION[context]` |
| Task 3 TrainingState/sampler | E3–E6 horizontal | dataset/split/S5 identity | each formal train run | true | `TRAINING_STATE_READY[context]` |
| Task 4 gradient roles | E3、E4、E5分阶段 | exact model/capability fixture | first formal use of each role | true | `GRADIENT_FLOW_QUALIFICATION[context]` |
| Tasks 5–6 loss/numeric | E3、E4、E5分阶段 | exact active loss/numeric profile | first formal use of each profile | true | `NUMERICAL_POLICY_QUALIFICATION[context]` |
| Task 7 qualification assembly | E3/E4/E5 scopes | 对应fixture全部完成 | exact stage entry | true | scope-local evidence bundle |

Framework code可以在E0后 `PARALLEL_PREPARATION`；但是fixture qualification只解锁其明确的`context/profile`，不能借一个简化profile为所有后续stage提供PASS。

## Interface Contract（接口合同）

- `derive_s5(project_identity: str, version: str) -> tuple[int, int, int, int, int]`
- `rng_domain(seed: int, namespace: RNGNamespace) -> RNGHandle`
- `checkpoint_replay(invocation: InvocationIdentity, profile: ReplayProfile) -> ReplayContext`
- `capture_training_state(runtime: TrainingRuntime) -> TrainingState`
- `restore_training_state(state: TrainingState, runtime: TrainingRuntime) -> None`
- `compile_parameter_roles(profile: VariantCapabilityProfile, stage: StageContext) -> ParameterRoleManifest`
- `qualify_gradient_flow(model: nn.Module, roles: ParameterRoleManifest) -> GradientFlowEvidence`
- `reduce_loss_terms(terms: Sequence[LossContribution]) -> HierarchicalLossResult`
- `build_optimizer(model: nn.Module, roles: ParameterRoleManifest, policy: OptimizerPolicy) -> OptimizerBundle`
- `build_training_qualification(registry: EvidenceRegistry) -> EvidenceBundle`

---

### Task 1（任务 1）：实现 S5 Derivation 与 Namespaced RNG Domain

**Files:**（文件）
- Create（创建）：`src/sc_mv_dmer/training/seeds.py`
- Create（创建）：`src/sc_mv_dmer/training/rng.py`
- Create（创建）：`schemas/rng_state.schema.json`
- Create（创建）：`tests/training/test_seed_derivation.py`
- Create（创建）：`tests/training/test_rng_domains.py`

**Test first:**（先写测试） 复现 project-identity SHA-256 indexed S5 set，并断言 split membership 不变。验证 initialization、sampler、shuffle、dropout、view dropout、event curriculum 及其他 registered stochastic role 使用相互独立且确定的 domain；unknown/colliding namespace 必须失败。

**Run:**（执行） `python -m pytest tests/training/test_seed_derivation.py tests/training/test_rng_domains.py -q`

**Expected initial result:**（预期初始结果） FAIL，因为 seed 和 RNG service 尚不存在。

**Minimal implementation:**（最小实现） 实现 canonical domain derivation、Python/NumPy/PyTorch CPU/CUDA framework state capture/restore、namespace registry 和 checksummed serialization，并复用 P0 canonicalization。

**Pass condition:**（通过条件） Golden vector 在进程间稳定，各 domain 不消费其他 domain 的 stream。

**Commit boundary:**（提交边界） `git add src/sc_mv_dmer/training/seeds.py src/sc_mv_dmer/training/rng.py schemas/rng_state.schema.json tests/training && git commit -m "feat: implement S5 namespaced RNG domains"`

### Task 2（任务 2）：实现 CRR Invocation-Local Replay Capsule

**Files:**（文件）
- Create（创建）：`src/sc_mv_dmer/training/checkpoint_replay.py`
- Create（创建）：`schemas/crr_capsule.schema.json`
- Create（创建）：`tests/training/test_checkpoint_replay.py`

**Test first:**（先写测试） 对 original checkpointed forward 和 recomputation，要求 invocation identity、entry RNG reference、exit RNG reference/hash、shadow value、stochastic trace、profile 和 checksum 完全一致。Nested/interleaved invocation 必须隔离。断言 capsule 不进入 durable `TrainingState`；View Dropout 不在 checkpoint body 内；尽管有 recomputation，live RNG 仍只 commit 一次。

**Run:**（执行） `python -m pytest tests/training/test_checkpoint_replay.py -q`

**Expected initial result:**（预期初始结果） FAIL，因为 replay capsule 尚不存在。

**Minimal implementation:**（最小实现） 实现 invocation-scoped context manager，捕获/恢复 required RNG domain，记录 original/recompute trace，验证 equality 并输出 typed evidence；通过显式 adapter 接入 checkpoint callback。

**Pass condition:**（通过条件） Recompute output/stochastic trace 一致，invocation 间无 state leak，durable-state schema 拒绝 capsule。

**Commit boundary:**（提交边界） `git add src/sc_mv_dmer/training/checkpoint_replay.py schemas/crr_capsule.schema.json tests/training/test_checkpoint_replay.py && git commit -m "feat: implement checkpoint RNG replay capsules"`

### Task 3（任务 3）：实现 Durable TrainingState 与 Exact-Once Sampler Restart

**Files:**（文件）
- Create（创建）：`src/sc_mv_dmer/training/state.py`
- Create（创建）：`src/sc_mv_dmer/training/sampler.py`
- Create（创建）：`schemas/training_state.schema.json`
- Create（创建）：`tests/training/test_training_state.py`
- Create（创建）：`tests/training/test_sampler_restart.py`

**Test first:**（先写测试） 必须覆盖 model、optimizer、scheduler、committed update index、sampler permutation/cursor、live RNG state、stage/epoch/phase state、referenced artifact identity，以及适用时的 scaler state。Resume 必须产生与 uninterrupted execution 相同的剩余 exact-once sequence 和 update trajectory；terminal run 不得重新打开。

**Run:**（执行） `python -m pytest tests/training/test_training_state.py tests/training/test_sampler_restart.py -q`

**Expected initial result:**（预期初始结果） FAIL，因为 durable state 和 sampler 尚不存在。

**Minimal implementation:**（最小实现） 实现 global-permutation exact-once sampling 和通过 continuation provenance 关联的 atomic immutable checkpoint；只恢复 committed state，并拒绝 split/fingerprint/config mismatch。

**Pass condition:**（通过条件） Interrupted fixture execution 与 uninterrupted order/state hash 一致，duplicate/skipped sample 均为 0。

**Commit boundary:**（提交边界） `git add src/sc_mv_dmer/training/state.py src/sc_mv_dmer/training/sampler.py schemas/training_state.schema.json tests/training && git commit -m "feat: bind durable training replay state"`

### Task 4（任务 4）：编译 Module Mode 并验证 Gradient Flow

**Files:**（文件）
- Create（创建）：`src/sc_mv_dmer/training/roles.py`
- Create（创建）：`src/sc_mv_dmer/training/gradients.py`
- Create（创建）：`tests/training/test_module_modes.py`
- Create（创建）：`tests/training/test_gradient_flow.py`
- Generate during execution（执行时生成）：`evidence/p3/gradient-flow-qualification.json`

**Test first:**（先写测试） 从 variant/stage capability profile 编译 `train/eval`、`requires_grad`、optimizer membership、expected gradient state 和 autograd transit。断言 Sensor loss 直接更新对应 handcrafted encoder；Sensor 不消费 fused/Markov output；View Dropout 只作用于进入 Markov/Fusion 的 handcrafted path；deep view 永不 dropout。Frozen Qwen base 不得有 parameter grad，允许的 LoRA/head parameter 必须有 grad。

**Run:**（执行）
```powershell
python -m pytest tests/training/test_module_modes.py tests/training/test_gradient_flow.py -q
python -m sc_mv_dmer.training.gradients qualify --profiles configs/experiments --output evidence/p3/gradient-flow-qualification.json
```

**Expected initial result:**（预期初始结果） Role compilation 实现前 FAIL。Profile 缺少 authoritative stage/capability row 时，以 `IMPLEMENTATION_BLOCKED_BY_FROZEN_CONTRACT: PARAMETER_ROLE_APPLICABILITY` 退出。

**Minimal implementation:**（最小实现） 编译冻结 SPP/MMP role matrix；对每个适用 role 执行一次 synthetic forward/backward；报告 expected/observed gradient state，包括 `grad is None`、finite norm 和 optimizer membership。

**Pass condition:**（通过条件） 每个适用 role 均符合合同，任何 mismatch 都是 hard qualification failure。

**Commit boundary:**（提交边界） `git add src/sc_mv_dmer/training/roles.py src/sc_mv_dmer/training/gradients.py tests/training && git commit -m "test: qualify module modes and gradient flow"`

### Task 5（任务 5）：实现 Hierarchical Loss Accumulation 与 Denominator State

**Files:**（文件）
- Create（创建）：`src/sc_mv_dmer/losses/accumulation.py`
- Create（创建）：`src/sc_mv_dmer/losses/reductions.py`
- Create（创建）：`tests/losses/test_hierarchical_accumulation.py`
- Create（创建）：`tests/losses/test_denominator_aware_reduction.py`

**Test first:**（先写测试） 对每个冻结 active loss family，比对 microbatch accumulation 与数学等价 full batch。覆盖 valid count 不等、parser failure、masked row、absent loss term、terminal partial window 和 zero-denominator policy。验证 term-level numerator/denominator 先于 hierarchy 与 Kendall weighting 聚合。

**Run:**（执行） `python -m pytest tests/losses/test_hierarchical_accumulation.py tests/losses/test_denominator_aware_reduction.py -q`

**Expected initial result:**（预期初始结果） FAIL，因为 reduction primitive 尚不存在。

**Minimal implementation:**（最小实现） 将每个 loss contribution 表示为 typed numerator、denominator、applicability 和 diagnostics；按 `HLAC-HDSA-3R+` 与 `DLC-P6-DAAI-3R+` 聚合；parser-failure handling 遵循其冻结 validity contract。

**Pass condition:**（通过条件） 所有 fixture 中 full-batch 与 accumulated result 在冻结 numerical tolerance 内一致，任何维度都不能被其他维度的 denominator 掩盖。

**Commit boundary:**（提交边界） `git add src/sc_mv_dmer/losses/accumulation.py src/sc_mv_dmer/losses/reductions.py tests/losses && git commit -m "feat: implement denominator-aware loss accumulation"`

### Task 6（任务 6）：实现 Optimizer、Scheduler 与 Numerical Policy

**Files:**（文件）
- Create（创建）：`src/sc_mv_dmer/training/optimizer.py`
- Create（创建）：`src/sc_mv_dmer/training/scheduler.py`
- Create（创建）：`src/sc_mv_dmer/training/numerics.py`
- Create（创建）：`tests/training/test_optimizer_groups.py`
- Create（创建）：`tests/training/test_scheduler_updates.py`
- Create（创建）：`tests/training/test_numerical_policy.py`

**Test first:**（先写测试） 验证冻结 AdamW value 和 decay exclusion、无 frozen parameter membership、update-indexed warmup/schedule、clipping position、terminal partial accumulation、BF16 eligibility、FP32 island、finite guard 和精确 scheduler restart。Markov/CCC/HSIC/Kendall 与 normalization-sensitive operation 必须在冻结 precision domain 中执行。

**Run:**（执行） `python -m pytest tests/training/test_optimizer_groups.py tests/training/test_scheduler_updates.py tests/training/test_numerical_policy.py -q`

**Expected initial result:**（预期初始结果） FAIL，因为 policy module 尚不存在。

**Minimal implementation:**（最小实现） 从 role 编译 optimizer group；scheduler 仅在 committed optimizer update 后 step；在 call boundary 强制 precision island；输出 structured finite/overflow diagnostic。

**Pass condition:**（通过条件） Golden trajectory、dtype assertion、restart equivalence 和 finite check 全部 PASS。

**Commit boundary:**（提交边界） `git add src/sc_mv_dmer/training/optimizer.py src/sc_mv_dmer/training/scheduler.py src/sc_mv_dmer/training/numerics.py tests/training && git commit -m "feat: implement frozen numerical training policy"`

### Task 7（任务 7）：生成 Context-scoped Training Qualification Evidence

**Files:**（文件）
- Create（创建）：`src/sc_mv_dmer/training/qualification.py`
- Create（创建）：`tests/training/test_training_qualification_bundle.py`
- Generate during execution（执行时生成）：`evidence/p3/e3-sensor-training-qualification.json`
- Generate during execution（执行时生成）：`evidence/p3/e4-no-llm-training-qualification.json`
- Generate during execution（执行时生成）：`evidence/p3/e5-stage1-training-qualification.json`
- Generate during execution（执行时生成）：`evidence/p3/e5-stage2-training-qualification.json`

**Test first:**（先写测试） 对每个context验证required closure：E3 Sensor要求其gradient/numeric/sampler subset；E4 No-LLM要求Markov/Fusion/ViewDropout/loss/numeric subset；E5 Stage-1/Stage-2分别要求其CRR/gradient/loss/optimizer/topology subset。每个bundle包含environment、fixture、semantic/resolved hash、contract ID、raw diagnostic和dependency validity；断言E5 Stage-2未完成不使E3/E4 bundle失败。

**Run:**（执行）
```powershell
python -m pytest tests/training tests/losses -q
python -m sc_mv_dmer.training.qualification --context e3-sensor --output evidence/p3/e3-sensor-training-qualification.json
python -m sc_mv_dmer.training.qualification --context e4-no-llm --output evidence/p3/e4-no-llm-training-qualification.json
python -m sc_mv_dmer.training.qualification --context e5-stage1 --output evidence/p3/e5-stage1-training-qualification.json
python -m sc_mv_dmer.training.qualification --context e5-stage2 --output evidence/p3/e5-stage2-training-qualification.json
python -m sc_mv_dmer.governance.verify_evidence evidence/p3/e5-stage1-training-qualification.json
```

**Expected initial result:**（预期初始结果） 每个context在自己的executable checks成功前保持non-PASS；其他later context缺失不得污染已完成的earlier context。

**Minimal implementation:**（最小实现） 按context组装immutable append-only evidence，显式区分fixture qualification与未来formal-run observation；禁止产生全P3 aggregate PASS。

**Pass condition:**（通过条件） 对应context verifier报告全部applicable closure，且无缺失contract/dependency hash或borrowed incompatible profile。

**Commit boundary:**（提交边界） `git add src/sc_mv_dmer/training/qualification.py tests/training/test_training_qualification_bundle.py && git commit -m "test: qualify training reproducibility contracts"`

## Context-scoped Exit Gates（分阶段退出门槛）

| Exit Gate | Required evidence | Unlock |
|---|---|---|
| `E3_TRAINING_RUNTIME_READY` | Sensor-applicable RNG/sampler/gradient/numeric qualification | E3 formal Sensor pre-validation |
| `E4_TRAINING_RUNTIME_READY` | No-LLM Markov/Fusion/ViewDropout/loss/numeric qualification | E4 formal No-LLM |
| `E5_STAGE1_RUNTIME_READY` | Stage-1 exact topology的CRR/gradient/optimizer/numeric qualification | E5.2/E5.3 |
| `E5_STAGE2_RUNTIME_READY` | Stage-2 exact QLoRA/artifact topology的CRR/gradient/loss/optimizer/numeric qualification | E5.9/E5.10 |

P3不授权科研训练，只证明某个明确context的冻结执行机制可运行。不存在“全部P3 closure完成后才允许任意实验”的全局Gate。
