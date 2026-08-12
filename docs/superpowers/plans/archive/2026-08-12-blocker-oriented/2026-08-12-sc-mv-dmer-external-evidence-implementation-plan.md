# SC-MV-DMER 外部证据实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task；按E5 exact Gate boundary执行，prerequisite resolver只要求当前role/artifact的精确依赖。

**Goal:**（目标） 在E5 Stage-1 entry、CoT→Stage-2 boundary、Stage-2 entry和final generation boundary分别收集canonical RG-08/真实人工RG-09 evidence；不得把两类Gate整体推迟到所有训练/消融之后，也不得把本地proxy或AI review冒充formal authority。

**Architecture:**（架构） 本文件保留P5 support tag。Prerequisite resolver根据`variant + exact stage/context`派生RG-08 role；runner/finalizer按E5.2、E5.9、E5.14分别产生attempt。RG-09在E5.7 CoT artifact完成后立即生成100条blinded package，并在E5.8终结；只有其current-effective PASS才解锁E5.9。

**Tech Stack:**（技术栈） Python 3.10+、PyTorch CUDA API、NVML binding、Pydantic、pytest、JSON Schema、static HTML/CSV review export。

## Global Constraints（全局约束）

- 本计划不是统一late-bound package；每个exact Gate attempt只要求自己的P0/P1/P2/P3/P4依赖current-effective。
- Canonical RG-08 evidence 在冻结 RTX 4090 24GB profile 上测量；每个适用 required role 必须低于 `22 * 2^30` byte。
- 本机 RTX 5060 Ti 结果仅是 local feasibility diagnostic，不能关闭 canonical RG-08。
- PASS 依据所有适用 required role，不要求 universal P1–P4。A10 保持 Stage-1 P1/P2、Stage-2 P1/P3、final FVA，P4 为 N/A。
- RG-09 必须由真实独立人工 authority 完成。Codex 或其他模型可以准备 evidence 和推荐 verdict，但不能冒充 reviewer/adjudicator。
- 外部执行必须创建新 run/evaluation attempt；不得重新打开 terminal run 或修改 historical evidence。
- 不执行 `git init`；下列 commit 仅是未来授权后的执行边界。

## Experiment-stage placement（实验阶段定位）

| Support task | experiment_stage_owner | Entry Gate | Required-before | can_prepare_early | Exit Gate |
|---|---|---|---|---:|---|
| Task 1 resolver/applicability | E5.2/E5.9/E5.14；E6/A10 | capability + exact context | each RG-08 role attempt | true | deterministic role/prerequisite manifest |
| Tasks 2–3 RG-08 Stage-1 P1/P2 | E5.2 | E5.1 loader/tokenizer + E5 Stage-1 runtime | E5.3 Stage-1 formal training | true（runner）；false（canonical evidence） | `E5_RG08_STAGE1_PASS` |
| Tasks 2–3 RG-08 Stage-2 P1/P3 | E5.9 | H3、RG-09 PASS、exact CoT/topology/runtime | E5.10 Stage-2 | true（runner）；false（canonical evidence） | `E5_RG08_STAGE2_PASS` |
| Tasks 2–3 RG-08 final P4 | E5.14 | E5.12 terminal + E5.13 final Event/prompt | final generation/evaluation | true（runner）；false（canonical evidence） | `E5_RG08_FINAL_PASS` |
| Tasks 2–3 A10 P1/P2/P3/FVA（P4 N/A） | E6/A10 | A10-specific topology/equivalence | E6 A10 exact boundaries | true | A10 scope-local RG-08 attempt |
| Tasks 4–5 RG-09 | E5.8 | E5.7 immutable eligible CoT artifact | E5.9 Stage-2 entry | true（tooling）；false（真实review） | `RG09_HUMAN_REVIEW_PASS` |
| Task 6 qualification assembly | 各exact boundary | 对应attempt | exact next step | true | scope-local blocker/Gate evidence |

RG-09不等待A5/A6或任何ablation。RG-08 Stage-1 roles不等待CoT；Stage-2 roles必须使用exact approved CoT/topology；final role不允许借用早期memory probe。A10属于E6且P4=`NOT_APPLICABLE_BY_DESIGN`。

## Interface Contract（接口合同）

- `resolve_gate_prerequisites(gate: GateRef, context: EvaluationContext) -> PrerequisiteManifest`
- `compile_rg08_roles(profile: VariantCapabilityProfile, context: StageContext) -> tuple[RG08Role, ...]`
- `measure_rg08_role(role: RG08Role, runtime: HardwareProfile) -> RG08RoleEvidence`
- `finalize_rg08(run: RunManifest, definition: GateDefinition) -> GateEvaluationAttempt`
- `select_rg09_records(cot: CoTArtifactManifest, rule: ReviewSamplingRule) -> ReviewSample`
- `build_review_package(sample: ReviewSample) -> HumanReviewPackage`
- `import_human_reviews(package: HumanReviewPackage, submissions: Sequence[HumanReviewRecord]) -> HumanReviewEvidence`
- `build_external_gate_qualification(registry: EvidenceRegistry) -> EvidenceBundle`

---

### Task 1（任务 1）：实现 External-Evidence Prerequisite Resolver

**Files:**（文件）
- Create（创建）：`src/sc_mv_dmer/gates/prerequisites.py`
- Create（创建）：`src/sc_mv_dmer/gates/applicability.py`
- Create（创建）：`tests/gates/test_external_prerequisites.py`
- Create（创建）：`tests/gates/test_rg08_applicability.py`

**Test first:**（先写测试） 必须验证 active Gate definition、predecessor effective PASS、未 invalidated 的 upstream evidence、pinned upstream/data/artifact identity、clean formal provenance 和 compiled stage/variant role applicability。覆盖 A10 明确 P4 N/A，以及只有 generated-output context 才适用的 P4 role。

**Run:**（执行） `python -m pytest tests/gates/test_external_prerequisites.py tests/gates/test_rg08_applicability.py -q`

**Expected initial result:**（预期初始结果） FAIL，因为 prerequisite/applicability resolver 尚不存在。

**Minimal implementation:**（最小实现） 查询 P0 effective-validity graph，编译冻结 RG-08 progressive topology，输出 signed input manifest；如失败，则输出逐项列出 invalid dependency 的 structured refusal。

**Pass condition:**（通过条件） Resolver 不调度 stale、invalidated、unbound 或 non-applicable role，并生成 deterministic role matrix。

**Commit boundary:**（提交边界） `git add src/sc_mv_dmer/gates/prerequisites.py src/sc_mv_dmer/gates/applicability.py tests/gates && git commit -m "feat: resolve external gate prerequisites"`

### Task 2（任务 2）：实现 RG-08 Measurement Runner

**Files:**（文件）
- Create（创建）：`src/sc_mv_dmer/gates/rg08.py`
- Create（创建）：`src/sc_mv_dmer/runtime/gpu_memory.py`
- Create（创建）：`schemas/rg08_evidence.schema.json`
- Create（创建）：`tests/gates/test_rg08_measurement.py`
- Create（创建）：`configs/runtime/rtx4090-24gb-canonical.yaml`

**Test first:**（先写测试） 验证 byte unit、device identity、driver/runtime/library provenance、warm-up exclusion policy、peak reset boundary、allocated/reserved/NVML sampling、OOM capture、role completeness，以及与 `22 * 2^30` 的严格比较。任何适用 role 缺失都禁止 PASS。

**Run:**（执行）
```powershell
python -m pytest tests/gates/test_rg08_measurement.py -q
$contexts = @('e5-stage1','e5-stage2','e5-final','e6-a10')
foreach ($ctx in $contexts) {
  python -m sc_mv_dmer.gates.prerequisites resolve --gate RG-08 --context $ctx --output "reports/preflight/rg08-$ctx-prerequisites.json"
  python -m sc_mv_dmer.gates.rg08 run --input-manifest "reports/preflight/rg08-$ctx-prerequisites.json" --runtime configs/runtime/rtx4090-24gb-canonical.yaml --run-root runs/evaluations --run-id-output "evidence/p5/rg08-$ctx-run-id.txt"
}
```

**Expected initial result:**（预期初始结果） Runner 实现前 unit test FAIL。在当前 non-canonical host 上，formal execution 以 `RG08_CANONICAL_HARDWARE_REQUIRED` 退出；diagnostic run 必须显式标为 debug 和 paper-ineligible。

**PRE-FLIGHT DISCOVERY TASK:**（执行前发现任务） 在 external host 加载 weight 前，记录 GPU UUID/model/VRAM、driver、CUDA、PyTorch、Transformers、PEFT、bitsandbytes、OS、launch command、source commit/clean state 和 input bundle hash。

**Minimal implementation:**（最小实现） 执行冻结 Gate 所要求的每个适用 no-training forward/backward/optimizer-shape role；在定义的 boundary reset/synchronize measurement；保存 raw trace；以 terminal status 终结 evaluation run。

**Pass condition:**（通过条件） 每个 applicable required role 都有 finite、完整 evidence，且 peak memory 严格低于冻结 limit；否则保留 immutable FAIL/BLOCKED evidence。

**Commit boundary:**（提交边界） `git add src/sc_mv_dmer/gates/rg08.py src/sc_mv_dmer/runtime/gpu_memory.py schemas/rg08_evidence.schema.json tests/gates/test_rg08_measurement.py configs/runtime/rtx4090-24gb-canonical.yaml && git commit -m "feat: add canonical RG08 measurement runner"`

### Task 3（任务 3）：终结 RG-08 Gate Evaluation Attempt

**Files:**（文件）
- Create（创建）：`src/sc_mv_dmer/gates/finalize_rg08.py`
- Create（创建）：`tests/gates/test_rg08_finalization.py`
- Generate externally（外部执行生成）：`evidence/p5/rg08-e5-stage1-canonical-evidence.json`
- Generate externally（外部执行生成）：`evidence/p5/rg08-e5-stage2-canonical-evidence.json`
- Generate externally（外部执行生成）：`evidence/p5/rg08-e5-final-canonical-evidence.json`
- Generate externally（外部执行生成）：`evidence/p5/rg08-e6-a10-canonical-evidence.json`
- Generate externally（外部执行生成）：`gates/evaluations/` 下一个 append-only record，其 concrete ID 由 finalizer 返回

**Test first:**（先写测试） 区分 historical verdict 与 effective verdict；必须包含 gate version、attempt、evidence bundle hash、automatic authority、active definition 和全部 applicable role result。验证 append-only finalization 与 downstream invalidation propagation。

**Run:**（执行）
```powershell
python -m pytest tests/gates/test_rg08_finalization.py -q
$contexts = @('e5-stage1','e5-stage2','e5-final','e6-a10')
foreach ($ctx in $contexts) {
  python -m sc_mv_dmer.gates.finalize_rg08 --context $ctx --run-id-file "evidence/p5/rg08-$ctx-run-id.txt" --run-registry runs/evaluations --gate-registry gates/evaluations --output "evidence/p5/rg08-$ctx-canonical-evidence.json"
}
```

**Expected initial result:**（预期初始结果） 缺失或 non-canonical evidence 时，以 `RG08_EVIDENCE_NOT_FORMALLY_ELIGIBLE` 拒绝 finalization。

**Minimal implementation:**（最小实现） 验证terminal run和exact stage/context role set，计算每项criterion和scope-local automatic verdict，然后append Gate attempt/dependency edge；不得修改source run，也不得把Stage-1/Stage-2/final role合成一个延迟到最后才生成的attempt。

**Pass condition:**（通过条件） 已生成完整不可变 Gate attempt，且 P0 effective-validity resolver 与其状态一致。

**Commit boundary:**（提交边界） `git add src/sc_mv_dmer/gates/finalize_rg08.py tests/gates/test_rg08_finalization.py && git commit -m "feat: finalize RG08 evaluation attempts"`

### Task 4（任务 4）：构建 Deterministic RG-09 Review Sample 与 Package

**Files:**（文件）
- Create（创建）：`src/sc_mv_dmer/gates/rg09.py`
- Create（创建）：`src/sc_mv_dmer/review/package.py`
- Create（创建）：`schemas/rg09_review_package.schema.json`
- Create（创建）：`tests/gates/test_rg09_sampling.py`
- Create（创建）：`tests/review/test_review_package.py`
- Generate during execution（执行时生成）：`evidence/p5/e5-rg09-review-package.json`

**Test first:**（先写测试） 必须至少有 100 条 eligible immutable synthetic record；按冻结 deterministic method 精确选择 100 条；排除 ineligible/invalidated item；保存 source hash；对 reviewer-independent field 做 blind；package release 后不可变。

**Run:**（执行）
```powershell
python -m pytest tests/gates/test_rg09_sampling.py tests/review/test_review_package.py -q
python -m sc_mv_dmer.gates.rg09 prepare --cot-manifest manifests/artifacts/cot-synthesis-v1.json --output evidence/p5/e5-rg09-review-package.json
```

**Expected initial result:**（预期初始结果） Sampling 实现前测试 FAIL。Eligible record 少于 100 条时以 `RG09_INSUFFICIENT_ELIGIBLE_RECORDS` 退出；缺少冻结 sampling authority 时以 `IMPLEMENTATION_BLOCKED_BY_FROZEN_CONTRACT: RG09_SAMPLE_SELECTION` 退出。

**Minimal implementation:**（最小实现） 解析 current-effective source record，应用冻结 hash-based selection，为两个 reviewer 和 adjudication 分别生成 blinded form，并为每个 view 计算 checksum。

**Pass condition:**（通过条件） 两份独立 blinded reviewer form 包含完全相同的 100 个 source identity，且无 mutable source payload。

**Commit boundary:**（提交边界） `git add src/sc_mv_dmer/gates/rg09.py src/sc_mv_dmer/review/package.py schemas/rg09_review_package.schema.json tests/gates/test_rg09_sampling.py tests/review/test_review_package.py && git commit -m "feat: prepare deterministic RG09 review package"`

### Task 5（任务 5）：收集并验证真实人工 Review Authority

**Files:**（文件）
- Create（创建）：`src/sc_mv_dmer/review/import_reviews.py`
- Create（创建）：`src/sc_mv_dmer/review/adjudication.py`
- Create（创建）：`schemas/human_review_record.schema.json`
- Create（创建）：`tests/review/test_human_review_import.py`
- Create（创建）：`tests/review/test_adjudication.py`
- Generate externally（外部执行生成）：`evidence/p5/e5-rg09-human-review-evidence.json`

**Test first:**（先写测试） 要求两个不同的 registered human reviewer identity、independent submission、timestamp、package hash 和完整 criterion answer；检测 conflict；冻结 policy 要求时必须有不同的 human adjudicator。拒绝 model-generated、duplicated、edited-in-place 或 self-approved record。

**Run:**（执行）
```powershell
python -m pytest tests/review/test_human_review_import.py tests/review/test_adjudication.py -q
python -m sc_mv_dmer.review.import_reviews --package evidence/p5/e5-rg09-review-package.json --review-a external/reviews/rg09/reviewer-a.json --review-b external/reviews/rg09/reviewer-b.json --adjudication external/reviews/rg09/adjudication.json --output evidence/p5/e5-rg09-human-review-evidence.json
```

**Expected initial result:**（预期初始结果） 在两份真实完整 submission 和任何 required adjudication 到位前，以 `RG09_HUMAN_AUTHORITY_INCOMPLETE` 退出；这是刻意保留的 external coordination boundary。

**Minimal implementation:**（最小实现） 验证 identity/independence；以 checksum 保留 original submission；计算 agreement/criterion summary；可以输出 recommendation，但 human-authority field 只能来自 authenticated human input。

**Pass condition:**（通过条件） Evidence bundle 与 package 一致，且不存在 unresolved required review conflict。

**Commit boundary:**（提交边界） `git add src/sc_mv_dmer/review schemas/human_review_record.schema.json tests/review && git commit -m "feat: validate RG09 human review evidence"`

### Task 6（任务 6）：终结 Exact-boundary External Gate Evidence

**Files:**（文件）
- Create（创建）：`src/sc_mv_dmer/gates/qualification.py`
- Create（创建）：`tests/gates/test_external_evidence_bundle.py`
- Generate externally（外部执行生成）：`evidence/p5/e5-rg08-stage1-qualification.json`
- Generate externally（外部执行生成）：`evidence/p5/e5-rg09-stage2-unlock.json`
- Generate externally（外部执行生成）：`evidence/p5/e5-rg08-stage2-qualification.json`
- Generate externally（外部执行生成）：`evidence/p5/e5-rg08-final-qualification.json`
- Generate externally（外部执行生成）：`evidence/p5/e6-a10-rg08-qualification.json`

**Test first:**（先写测试） 按exact boundary验证`RG08_CANONICAL_HARDWARE_EVIDENCE` role subset与`RG09_HUMAN_REVIEW_EXECUTION`：Stage-1 bundle不得要求RG-09；Stage-2 unlock必须要求RG-09；final bundle不得借Stage-1 peak；A10 bundle要求P1/P2/P3/FVA且P4 N/A。验证authority、active Gate version、attempt、dependency validity、checksum和historical/effective verdict分离。

**Run:**（执行）
```powershell
python -m pytest tests/gates tests/review -q
python -m sc_mv_dmer.gates.qualification --scope e5-stage1 --rg08 evidence/p5/rg08-e5-stage1-canonical-evidence.json --output evidence/p5/e5-rg08-stage1-qualification.json
python -m sc_mv_dmer.gates.qualification --scope e5-stage2-unlock --rg09 evidence/p5/e5-rg09-human-review-evidence.json --output evidence/p5/e5-rg09-stage2-unlock.json
python -m sc_mv_dmer.gates.qualification --scope e5-stage2 --rg08 evidence/p5/rg08-e5-stage2-canonical-evidence.json --output evidence/p5/e5-rg08-stage2-qualification.json
python -m sc_mv_dmer.governance.verify_evidence evidence/p5/e5-rg09-stage2-unlock.json
```

**Expected initial result:**（预期初始结果） 每个scope在自己的external evidence真实完整且current-effective前保持non-PASS；later/final/A10 evidence缺失不使Stage-1 scope失败。

**Minimal implementation:**（最小实现） 为每个exact boundary创建append-only Gate evaluation attempt和blocker-closure record，并连接dependency graph；禁止全P5 aggregate PASS，不得修改任何既有run/Gate record。

**Pass condition:**（通过条件） 对应scope verifier报告required effective closure，automatic/manual authority正确，且无stale或borrowed incompatible evidence。

**Commit boundary:**（提交边界） `git add src/sc_mv_dmer/gates/qualification.py tests/gates/test_external_evidence_bundle.py && git commit -m "test: qualify external gate evidence"`

## Exact-boundary Exit Gates（精确边界退出门槛）

| Exit Gate | Required evidence | Unlock |
|---|---|---|
| `E5_RG08_STAGE1_PASS` | canonical Stage-1 P1/P2 applicable roles | E5.3 Stage-1 formal training |
| `E5_RG09_COT_PASS` | exact CoT artifact的真实双reviewer/adjudicated RG-09 PASS | E5.9 Stage-2 qualification |
| `E5_RG08_STAGE2_PASS` | canonical Stage-2 P1/P3 on exact topology/artifact | E5.10 Stage-2 |
| `E5_RG08_FINAL_PASS` | applicable P4 final generation role | E5.14/E5.15 final evidence |
| `E6_A10_RG08_PASS` | A10 P1/P2/P3/FVA；P4 N/A | E6 A10对应边界 |

P5不再拥有统一Exit Gate。本地proxy measurement或model-authored review不能改变formal authority；某个later scope未完成只阻塞其exact next step。
