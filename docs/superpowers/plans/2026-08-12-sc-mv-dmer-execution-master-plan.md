# SC-MV-DMER 实验步骤型执行总计划 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 严格按照《方案 E》§8.3 的实验顺序，在本机优先完成 7B primary implementation，并以 current-effective Gate evidence 逐步放行 MERT、四视图缓存、Sensor、No-LLM、Qwen Stage 1/2、CoT、A1–A13 和外部评估。

**Architecture:** Step 0 只建立治理与数据基础；Step 1–4 依次完成帧率/维度、四视图缓存、Sensor 预验证和 No-LLM Gate；Step 5 完成 Qwen 7B 接入、Stage 1 与 H3 handoff；Step 6 先批准 phase-matched CoT artifact，再执行正式 Stage 2、消融和评估。每一步只消费前序不可变、未 invalidated 且 current-effective 的 evidence。

**Tech Stack:** Windows PowerShell；Python `>=3.10,<3.13`；Pydantic `>=2.8,<3`；PyYAML `>=6.0.2,<7`；pytest `>=8.3,<9`；Hypothesis `>=6.112,<7`；NumPy、pandas、librosa、matplotlib；PyTorch、Transformers、PEFT、bitsandbytes、safetensors 的确切版本由 executable environment manifest 冻结。

## Global Constraints

- `docs/research/RESEARCH_SPEC.md` 是研究语义 Source of Truth；任何研究语义变更必须走 `DECISIONS.md → RESEARCH_SPEC.md → config/schema → version bump`。
- 唯一 primary split 是歌曲级 80/10/10 manifest；S5 固定为 `[52826381,128866372,1616929435,1871035633,1460830465]`，seed 不改变 dataset membership。
- Formal run 必须 clean Git workspace；当前规划不执行代码、训练、`git init`、commit 或外部运行。各子计划中的 commit 是未来获授权后的边界。
- 本机 RTX 5060 Ti 16GB 优先；本机无法合法完成的 exact run 保存不可变失败/资源证据并创建新 external run，禁止重新打开 terminal run。
- 本机 memory evidence 不能替代 canonical RTX 4090 24GB RG-08 authority；canonical applicable-role peak 要求严格 `<22*2^30` bytes。
- Stage 2 synthetic-supervised songs 必须在开跑前拥有 phase/variant-matched immutable approved CoT/ANSWER artifact；缺失时以 `STAGE2_BLOCKED_BY_COT_ARTIFACT` fail closed。
- Formal test 不参与选模、Gate 阈值或调参；final checkpoint 只按冻结 CS-3+ terminal rule designation。
- A1–A13、rank、loss、warmup、S5、split、Gate threshold、Event visibility、ANSWER 长度、PMEmo mapping 和 A13 objective 不得由实施静默改变。
- 14B、DPO、Qwen2.5-Audio 和 contingency experiments 为 `DEFERRED_NON_BLOCKING`，不得阻塞 7B primary。
- 如冻结权威缺失或冲突，输出 `IMPLEMENTATION_BLOCKED_BY_FROZEN_CONTRACT: <contract_id>`，不得自行补造。
- 测试片段中的 `*_fixture()`/`*_fixture` helper 必须在同一测试文件或该步骤明确创建的相邻 fixture module 中实现；禁止假定仓库中存在未登记的隐藏 fixture。

---

## 计划文件与唯一执行顺序

| 顺序 | 计划文件 | 核心输出 | 进入 Gate |
|---|---|---|---|
| 0 | `2026-08-12-sc-mv-dmer-step-0-foundation-preflight-implementation-plan.md` | 治理内核、DEAM manifest/split/audit、formal pre-flight | Foundation 文档已冻结 |
| 1 | `2026-08-12-sc-mv-dmer-step-1-mert-frame-rate-implementation-plan.md` | RG-01、`DownstreamDimensionBinding` | Step 0 effective PASS |
| 2 | `2026-08-12-sc-mv-dmer-step-2-handcrafted-feature-cache-implementation-plan.md` | four-view cache、pseudo-label、RG-02 | Step 1 effective PASS |
| 3 | `2026-08-12-sc-mv-dmer-step-3-sensor-prequalification-implementation-plan.md` | standalone Sensor bundle/evidence | Step 2 effective PASS |
| 4 | `2026-08-12-sc-mv-dmer-step-4-no-llm-prototype-implementation-plan.md` | No-LLM model、RG-04/05/06 | Step 3 effective PASS |
| 5 | `2026-08-12-sc-mv-dmer-step-5-qwen-7b-stage1-handoff-implementation-plan.md` | Qwen binding、Stage-1 terminal、H3、Stage-2 entry | Step 4 No-LLM Gate PASS |
| 6 | `2026-08-12-sc-mv-dmer-step-6-cot-stage2-ablation-evaluation-implementation-plan.md` | CoT、formal Stage 2、A1–A13、evaluation/aggregation | Step 5 handoff + approved artifacts |

归档的 blocker-oriented plans 位于 `docs/superpowers/plans/archive/2026-08-12-blocker-oriented/`，只作历史审计，不是执行入口。

## 18 个 OPEN blocker 归属

| Blocker | Owner |
|---|---|
| `MACHINE_READABLE_MANIFEST_CONFIG_BINDING` | Step 0 |
| `CLEAN_GIT_PROVENANCE_PREFLIGHT` | Step 0 |
| `OQ-DATA-DEAM-TARGET-ARTIFACT-BINDING` | Step 0 |
| `DEAM_ANNOTATION_AUDIT` | Step 0 |
| `EVALUATION_INPUT_COVERAGE_EXECUTABLE_BINDING` | Step 2 |
| `PINNED_QWEN_UPSTREAM_MANIFEST` | Step 5 |
| `QWEN_EXECUTABLE_LOADER_QUALIFICATION` | Step 5 |
| `A10_PROJECTION_EQUIVALENCE_QUALIFICATION` | Step 5 |
| `TOKENIZER_ANSWER_BUDGET_QUALIFICATION` | Step 5 |
| `CRR_EXECUTABLE_QUALIFICATION` | Step 5 |
| `GRADIENT_FLOW_QUALIFICATION` | Step 5 |
| `NUMERICAL_POLICY_QUALIFICATION` | Step 5 |
| `RG08_CANONICAL_HARDWARE_EVIDENCE` | Step 5 entry roles + Step 6 artifact-dependent roles |
| `A5_MODEL_VISIBLE_INPUT_FIREWALL_QUALIFICATION` | Step 6 |
| `A6_EVENT_REPLAY_BUNDLE_BINDING` | Step 6 |
| `COT_ARTIFACT_MATERIALIZATION` | Step 6 |
| `PMEMO_ARTIFACT_HASH_DOMAIN_AUDIT` | Step 6 |
| `RG09_HUMAN_REVIEW_EXECUTION` | Step 6 |

## 《方案 E》附录 B 十项验证的执行位置

| 附录 B 验证项 | 计划任务 | 放行证据 |
|---|---|---|
| 1. MERT 真实输出帧率及全部下游维度 | Step 1 Tasks 2–5 | RG-01 + `DownstreamDimensionBinding` |
| 2. 四视图分箱与 3 首歌曲对齐 | Step 2 Tasks 2–6 | RG-02 automatic + real-human review |
| 3. Sensor Heads 独立预训练 | Step 3 Tasks 1–5 | RG-03 + `StandaloneSensorBundle` |
| 4. No-LLM 精度摸底 | Step 4 Tasks 1–5 | RG-04 REACHED/NEAR PASS |
| 5. Markov 状态使用率 | Step 4 Task 6 | RG-05 occupancy evidence |
| 6. beta 学习曲线 | Step 4 Task 6 | RG-06 last-3/CV evidence |
| 7. Event 文本长度 | Step 6 Task 3 | RG-07 all required sources PASS |
| 8. 4090 memory | Step 5 Task 7 + Step 6 Task 5 | RG-08 all applicable required roles |
| 9. CoT 100 条质量审核 | Step 6 Task 3 | RG-09 real-human attempt |
| 10. ANSWER parser / `L_consist` | Step 5 Task 3 + Step 6 Task 5 | qualification fixture + formal RG-10-v2 |

### Task 1: 执行 Step 0 Foundation 与数据预检

**Files:**
- Follow: `docs/superpowers/plans/2026-08-12-sc-mv-dmer-step-0-foundation-preflight-implementation-plan.md`
- Produce during execution: `evidence/qualifications/step-0-foundation-preflight.json`

**Interfaces:**
- Consumes: frozen Foundation documents.
- Produces: `FoundationPreflightBundle` and stable dataset/split identities.

- [ ] **Step 1:** 完整执行 Step 0 子计划的任务与 red/green tests。
- [ ] **Step 2:** 运行 `python -m sc_mv_dmer.cli qualify-step --step 0 --mode formal --output evidence/qualifications/step-0-foundation-preflight.json`。
- [ ] **Step 3:** 确认输出 `effective_status=VALID`、4 个 Step-0 blocker 全部 closed；否则停止，不进入 Step 1。
- [ ] **Step 4:** 未来获 Git 授权后提交 Step 0 边界：`git add pyproject.toml src tests configs schemas manifests evidence`，随后单独运行 `git commit -m "feat: establish foundation preflight"`。

### Task 2: 执行 Step 1 MERT 帧率与下游维度

**Files:**
- Follow: `docs/superpowers/plans/2026-08-12-sc-mv-dmer-step-1-mert-frame-rate-implementation-plan.md`
- Produce: `manifests/dimensions/mert-downstream-dimensions-v1.json`

**Interfaces:**
- Consumes: `FoundationPreflightBundle`.
- Produces: `DownstreamDimensionBinding` and RG-01 attempt.

- [ ] **Step 1:** 执行 Step 1 子计划，不使用方案中的 50 Hz 示意值替代实测。
- [ ] **Step 2:** 运行 `python -m sc_mv_dmer.cli qualify-step --step 1 --mode formal --output evidence/qualifications/step-1-mert.json`。
- [ ] **Step 3:** 验证 RG-01 current-effective PASS、dimension manifest checksum 已封存；否则停止。
- [ ] **Step 4:** 未来提交消息使用 `feat: bind measured MERT dimensions`。

### Task 3: 执行 Step 2 四视图与伪标签缓存

**Files:**
- Follow: `docs/superpowers/plans/2026-08-12-sc-mv-dmer-step-2-handcrafted-feature-cache-implementation-plan.md`
- Produce: `manifests/features/deam-four-view-cache-v1.json`

**Interfaces:**
- Consumes: exact Step-1 dimension checksum.
- Produces: `FourViewFeatureCacheManifest`, `PseudoLabelBundleManifest`, RG-02 evidence.

- [ ] **Step 1:** 执行 Step 2 子计划，对冻结 population 提取 MERT deep/Mel/MFCC/chroma 与四类 pseudo-label。
- [ ] **Step 2:** 运行 `python -m sc_mv_dmer.cli qualify-step --step 2 --mode formal --output evidence/qualifications/step-2-features.json`。
- [ ] **Step 3:** 验证至少 3 首预注册歌曲 alignment evidence 和 cache immutability；否则停止。
- [ ] **Step 4:** 未来提交消息使用 `feat: materialize immutable four-view cache`。

### Task 4: 执行 Step 3 Sensor Heads 独立预验证

**Files:**
- Follow: `docs/superpowers/plans/2026-08-12-sc-mv-dmer-step-3-sensor-prequalification-implementation-plan.md`
- Produce: `manifests/sensors/standalone-sensor-bundle-v1.json`

**Interfaces:**
- Consumes: feature/pseudo-label manifests.
- Produces: `StandaloneSensorBundle` and Sensor Gate evidence.

- [ ] **Step 1:** 按子计划训练 standalone Sensor，不连接 Markov/Fusion/Qwen。
- [ ] **Step 2:** 运行 `python -m sc_mv_dmer.cli qualify-step --step 3 --mode formal --output evidence/qualifications/step-3-sensor.json`。
- [ ] **Step 3:** 验证所有适用 criterion PASS、test 未参与、encoder gradient contract 有效；否则保留 FAIL 并停止。
- [ ] **Step 4:** 未来提交消息使用 `feat: qualify standalone sensor heads`。

### Task 5: 执行 Step 4 No-LLM 原型

**Files:**
- Follow: `docs/superpowers/plans/2026-08-12-sc-mv-dmer-step-4-no-llm-prototype-implementation-plan.md`
- Produce: `evidence/gates/rg-04/`, `evidence/gates/rg-05/`, `evidence/gates/rg-06/`

**Interfaces:**
- Consumes: Steps 1–3 current-effective artifacts.
- Produces: No-LLM terminal model and Gate attempts.

- [ ] **Step 1:** 执行四视图、Markov、Fusion、direct VA 训练与 CRNN 对齐比较。
- [ ] **Step 2:** 运行 `python -m sc_mv_dmer.cli qualify-step --step 4 --mode formal --output evidence/qualifications/step-4-no-llm.json`。
- [ ] **Step 3:** 确认逐维 RG-04、适用 RG-05/06 全部 current-effective PASS；否则禁止 Qwen integration。
- [ ] **Step 4:** 未来提交消息使用 `feat: qualify no-llm acoustic prototype`。

### Task 6: 执行 Step 5 Qwen 7B Stage 1 与 handoff

**Files:**
- Follow: `docs/superpowers/plans/2026-08-12-sc-mv-dmer-step-5-qwen-7b-stage1-handoff-implementation-plan.md`
- Produce: `manifests/upstream/qwen-7b-primary-v1.json`, Stage-1 run manifests, H3 packages.

**Interfaces:**
- Consumes: Step-4 PASS and pinned local Qwen snapshot.
- Produces: `H3StageHandoffPackage` and `Stage2EntryQualificationBundle`.

- [ ] **Step 1:** 执行 loader/双头/training-infrastructure qualification 与正式 Stage 1。
- [ ] **Step 2:** 运行 `python -m sc_mv_dmer.cli qualify-step --step 5 --mode formal --output evidence/qualifications/step-5-stage1.json`。
- [ ] **Step 3:** 确认 artifact-dependent RG-08 roles 明确为 `PENDING_STEP_6_ARTIFACT`，而不是伪 PASS。
- [ ] **Step 4:** 未来提交消息使用 `feat: bind qwen stage1 handoff`。

### Task 7: 执行 Step 6 CoT、Stage 2、消融与评估

**Files:**
- Follow: `docs/superpowers/plans/2026-08-12-sc-mv-dmer-step-6-cot-stage2-ablation-evaluation-implementation-plan.md`
- Produce: CoT/Event artifacts, Stage-2 terminal runs, A1–A13 evidence, external evaluations, aggregation manifest.

**Interfaces:**
- Consumes: H3 packages and phase-specific Event sources.
- Produces: paper-eligible `ExperimentEvidenceMatrix` only when every dependency is current-effective.

- [ ] **Step 1:** 先物化并审核 CoT artifacts，再启动任何 formal Stage-2 run。
- [ ] **Step 2:** 在本机按资源允许顺序执行；资源不足时终结本机 run，并从不可变 handoff 在其他环境创建新 run_id。
- [ ] **Step 3:** 完成 A1 S5、所需 A-variants、long58/PMEmo、RG-07/08/09/10 和 effective-validity aggregation。
- [ ] **Step 4:** 运行 `python -m sc_mv_dmer.cli qualify-step --step 6 --mode formal --output evidence/qualifications/step-6-final.json`。
- [ ] **Step 5:** 只有 `COMPLETE_S5=true`、required Gates current-effective 且 invalidation 传播清零时，未来提交 `feat: complete primary experiment evidence`。

## 最终验证命令

```powershell
python -m pytest -q
python -m sc_mv_dmer.cli validate-foundation --mode formal
python -m sc_mv_dmer.cli validate-dependency-graph --require-steps 0,1,2,3,4,5,6
python -m sc_mv_dmer.cli aggregate --profile paper --require-complete-s5 --dry-run
```

预期：pytest 全部 PASS；前三个 validator 返回 exit code 0；aggregation dry-run 只在 exact S5 与 current-effective dependency closure 完整时返回 `PAPER_AGGREGATION_ELIGIBLE`。
