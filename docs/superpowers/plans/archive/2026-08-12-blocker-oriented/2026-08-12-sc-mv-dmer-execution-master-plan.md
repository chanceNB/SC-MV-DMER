# SC-MV-DMER 实验阶段顺序执行总计划 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改变冻结科研语义的前提下，以 `E0→E1→E2→E3→E4→E5→E6` 为唯一正式实验执行主线，逐 Gate 解锁 7B primary、S5、消融和 secondary evaluation。

**Architecture:** E0 只建立 E1.1 所需的最低治理、MERT upstream identity 和至少一条 registered real DEAM 45s probe identity；E1 严格按“先实测、后反推、再绑定时间轴、最后关闭 RG-01”执行；E2–E4 依次完成手工特征、Sensor 和 No-LLM；E5 完成完整 Qwen2.5-7B Stage-1→Event/CoT→Stage-2→final evaluation；E6 完成 A1 COMPLETE_S5、A1–A13 消融和 secondary evaluation。原 P0–P5 文件只提供 support tasks，不拥有阶段 entry/exit authority。

**Tech Stack:** Windows PowerShell；Python 3.10.11；Pydantic v2、PyYAML、NumPy、pandas、pytest；PyTorch、Transformers、PEFT、bitsandbytes、safetensors、NVML；确切 executable environment 由 run manifest 固化。

## Global Constraints

- Foundation research semantics 保持 `APPROVED / FROZEN`；实施不得修改 rank、loss weight、warmup、S5、split、Gate threshold、Event visibility、A1–A13、ANSWER 长度、PMEmo mapping 或 A13 objective。
- Primary split 是唯一、不可变的歌曲级 80/10/10 manifest；S5 是 `[52826381,128866372,1616929435,1871035633,1460830465]`，不改变 dataset membership。
- 工程代码完成不等于阶段解锁；下一阶段只接受 current-effective predecessor PASS。
- Formal run 需要 clean Git、registered artifacts、resolved/semantic config hashes、terminal immutable manifest 和 effective-validity dependency closure。
- RG-08 canonical authority 是 RTX 4090 24GB applicable-role evidence；本机 RTX 5060 Ti 16GB 仅能形成 local feasibility 或 blocked-compute evidence。
- 真实人工 criterion 不能由 Codex/LLM 代签；RG-02/RG-09 未完成人工 authority 时保持 non-PASS。
- 14B、DPO、Qwen2.5-Audio 和 contingency experiments 不进入 7B primary AND chain。
- `PARALLEL_PREPARATION` 只允许提前开发/发现，不产生阶段 entry 权；无法完成旁路准备不得阻塞无依赖的 primary critical path。
- E1 遵循 `MEASURE FIRST → DERIVE SECOND`：真实 MERT forward 之前及 E1.1 内不得把 `MERT_FPS=50`、`T_raw=2250` 或任何历史估算帧率当作 authority，也不得预先实例化依赖这些估计的 TimesNet、reduction、2Hz mapping 或 downstream temporal dimensions。
- 如权威缺失或冲突，输出 `IMPLEMENTATION_BLOCKED_BY_FROZEN_CONTRACT: <contract_id>`，不得猜测。
- 本轮只重构计划；禁止 `git init`、代码实现、data build、model download/load、training 和 Gate execution。

---

## 1. Highest-order execution authority

```text
E0  Execution Bootstrap / Pre-flight
 ↓
E1  MERT Frame-Rate Print → Downstream Dimension Derivation → 2Hz Binding → RG-01
 ↓
E2  Handcrafted Offline Features + Alignment + RG-02
 ↓
E3  Sensor Heads Independent Pre-validation + RG-03
 ↓
E4  No-LLM Main Model + RG-04 + applicable RG-05/RG-06
 ↓
E5  ChatTS + Qwen2.5-7B full staged pipeline
 ↓
E6  COMPLETE_S5 + Ablations + Secondary Evaluation + optional 14B
```

P0–P5 仅是以下 support work-package tag：

| Tag | Support plan | 不拥有的权限 |
|---|---|---|
| P0 | `2026-08-12-sc-mv-dmer-execution-governance-implementation-plan.md` | 不得要求 late artifact 才解锁 E1 |
| P1 | `2026-08-12-sc-mv-dmer-data-binding-implementation-plan.md` | 不得用 PMEmo 阻塞 DEAM/MERT 主线 |
| P2 | `2026-08-12-sc-mv-dmer-qwen-binding-implementation-plan.md` | 不得用 A10 qualification 阻塞 canonical A1 E5 |
| P3 | `2026-08-12-sc-mv-dmer-training-qualification-implementation-plan.md` | 不得把全部横向 qualification 合并成单一前置包 |
| P4 | `2026-08-12-sc-mv-dmer-variant-artifacts-implementation-plan.md` | 不得用 A5/A6 阻塞 canonical A1 E5 |
| P5 | `2026-08-12-sc-mv-dmer-external-evidence-implementation-plan.md` | 不得把 RG-08/RG-09统一推迟到所有训练之后 |

## 2. E0 — Execution Bootstrap / Pre-flight

**Research objective:** 无论文结果；建立 E1 所需最低可执行治理边界。

**Entry dependencies:** 已冻结的 `DECISIONS.md`、`RESEARCH_SPEC.md`、Foundation Design、Experiment Matrix；明确实施授权才允许未来 Git bootstrap。

**Implementation tasks:** package/repository baseline；stable IDs；canonical JSON 与双 hash；RunSpec/RunManifest/artifact/Gate/invalidation primitives；formal/debug provenance；upstream-manifest primitive schema；为 E1.1 登记 pinned MERT/processor identity、exact 45s probe contract，以及至少一条 registered real DEAM 45s sample 的 stable identity、source hash 与 preprocessing provenance。完整 1,744 population 和 frozen split identity 可在 E1 并行准备，但不是 E1.1 的 predecessor。

**Relevant existing sub-plan tasks:** P0 Tasks 1–8；P1 Tasks 1–2 的 identity/source/probe-binding 部分。P1 Task 3 的完整 population/split 是 E1 parallel preparation，并在 E2 前关闭。

**Formal experiment:** 无。只运行 pre-flight qualification fixtures 和只读 DEAM inventory/binding。

**Required evidence:** machine-readable Foundation binding；clean/dirty formal/debug evidence；pinned MERT upstream/processor identities；exact `45s @ 24,000Hz = 1,080,000 samples` probe specification；至少一条 registered real DEAM 45s sample identity/hash；required preprocessing/provenance；effective-validity resolver evidence。

**Gate:** `E0_PRE_FLIGHT`，不是 Research Gate，也不产生 paper result。

**PASS condition:** `MACHINE_READABLE_MANIFEST_CONFIG_BINDING`、`CLEAN_GIT_PROVENANCE_PREFLIGHT` 和 `E0_E1_PROBE_READY` 当前有效；不得等待完整 1,744 population/split、annotation audit、PMEmo、secondary coverage、Qwen、Sensor、CoT、A5/A6 或 RG-09。`E0_E1_PROBE_READY` 只证明 E1.1 输入和 provenance 已绑定，不冒充完整 `OQ-DATA-DEAM-TARGET-ARTIFACT-BINDING` closure。

**FAIL/BLOCKED behavior:** 保存 typed blocker 和 source-coverage report；禁止生成替代 split、硬编码路径或伪造 M0 approval。

**Next-stage unlock:** 只解锁 E1.1 `MERT_REAL_OUTPUT_FRAME_RATE_PRINT`。

## 3. E1 — MERT Measure-first Qualification / RG-01

**Research objective:** 严格执行 `MEASURE FIRST → DERIVE SECOND`：先通过真实 MERT forward 获得帧数/帧率，再反推全部下游维度并验证 2Hz/90-bin 时间轴。

**Entry dependencies:** E0 PASS；machine-readable evidence primitive 可用；pinned MERT upstream/processor identity 可用；exact 45s/24kHz/1,080,000-sample probe contract已绑定；至少一条 registered real DEAM 45s sample 具有 stable identity、source hash 和 required provenance。完整 1,744 population、80/10/10 split、annotation audit、PMEmo、Qwen、Sensor、CoT、A5/A6 均不是 E1.1 predecessor。

### E1.1 — `MERT_REAL_OUTPUT_FRAME_RATE_PRINT`

**Purpose:** 完成 research plan §8.3 Step 1 和附录 B #1 所要求的真正第一步；只测量 MERT 真实输出，不推断或写死 downstream dimension。

**Planned execution:** 加载 pinned MERT 与 pinned processor；分别对冻结 exact-length waveform probe 和至少一条 registered real DEAM 45s sample 执行真实 forward。每条输入必须是 `45s`、`24,000Hz`、`1,080,000 samples`。必须实际打印并保存：

- `input_sample_rate`
- `input_sample_count`
- `input_shape`
- `hidden_states_count`
- `research_layer_5_actual_index`
- `research_layer_6_actual_index`
- `layer5_shape`
- `layer6_shape`
- `observed_frame_count = T_raw`
- `hidden_dimension = D`
- `observed_fps = T_raw / 45.0`
- `finite_status`

**Independent evidence:** 生成不可变 `MERTFrameRateVerificationEvidence`，至少包含 MERT model identity/revision/hash、processor identity/revision/hash、probe identity/hash、sample rate/count/duration、layer mapping、raw tensor shapes、`T_raw`、`D_hidden`、`observed_fps`、stdout/raw report、environment identity、timestamp 和 checksum；另生成独立的人类可读《MERT 真实输出帧率打印验证》报告，显式展示“实测帧数 = ?”与“实测帧率 = ? Hz”，不得只把结果藏在 JSON 中。

**PASS iff:** 真实 forward 成功；layer 5/6 identity 与 actual indices 明确；`T_layer5 == T_layer6`；`D == 768`（当前冻结 expected hidden dimension）；所有 outputs finite；`T_raw` 来自真实 forward；`observed_fps` 已按 `T_raw/45.0` 计算；evidence/provenance/checksum 完整。任一条件失败即 immutable non-PASS 并 `STOP_BEFORE_E1_2`。

**Prohibition:** E1.1 之前及 E1.1 内禁止把 README、论文描述、model config 猜测、`MERT_FPS=50`、`T_raw=2250` 或其他历史估算作为真实 frame count；只有 real-forward observation 拥有 authority。

### E1.2 — `MERT_DOWNSTREAM_DIMENSION_DERIVATION`

**Entry:** 仅 current-effective E1.1 PASS。

**Planned derivation:** 只读取 E1.1 的 actual `T_raw`、actual `observed_fps` 和 pinned upstream temporal geometry（kernels/strides/padding/dilation/reductions），反推并保存：

- TimesNet raw temporal input dimension；
- TimesNet period interpretation/set（严格服从冻结 TimesNet contract）；
- required temporal reduction；
- MERT-to-2Hz reduction relationship；
- canonical downstream `T`；
- Deep projection dimensions；
- all downstream temporal dimensions。

**Evidence/output:** 生成不可变 `DownstreamDimensionBinding`，逐边记录 source observation、公式、单位、shape、contract/version 和 checksum。任何 downstream target（包括 `T=90`）不得 trim、pad、interpolate、reshape 或改写 E1.1 的 `T_raw`；唯一合法方向为 `MERT actual output → deterministic downstream mapping`。

### E1.3 — `MERT_TIMEGRID_2HZ_BINDING`

**Entry:** current-effective E1.2 completion，且 `DownstreamDimensionBinding` 可验证。

**Planned verification:** 从 actual upstream temporal geometry 推导 effective stride、receptive field、first support、frame centers/timestamps 和 expected count；要求 expected count 等于 E1.1 observed count。随后按 sample-domain deterministic arithmetic 绑定 0.5s 半开 bins，形成 2Hz/90-bin TimeGrid，并验证所有 eligible frames exactly once：`unassigned=0`、`duplicate=0`、`sum(bin_frame_counts)=T_raw`，且 RG-01 要求的 90 个 bins 全部 valid/nonempty。禁止 silent trim/pad/drop/clamp、copy/fill/interpolation 或 adaptive pooling 掩盖问题。

### E1.4 — `RG01_FORMAL_CLOSURE`

**Entry/evidence consumption:** 同时消费 current-effective E1.1 `MERTFrameRateVerificationEvidence`、E1.2 `DownstreamDimensionBinding` 和 E1.3 TimeGrid/all-frame-accounting evidence；执行冻结 RG-01 definition 的正式 evaluation attempt。

**PASS condition:** E1.1–E1.3 全部有效且满足冻结 RG-01；Gate attempt 的 historical verdict 为 PASS，且 dependency-aware resolver 给出 current-effective PASS。只有该状态才解锁 E2；FAIL/BLOCKED/INVALIDATED/STALE_DEPENDENCY 均不得继续。

**Relevant existing sub-plan tasks:** P0 manifest/provenance APIs；P1 registered real DEAM probe binding；实验步骤型 Step-1 MERT measure/derive/time-grid/Gate tasks。

**Formal experiment:** E1.1 对 frozen exact-length waveform probe 与至少一条 registered real DEAM 45s sample 做 real forward；E1.4 创建 RG-01 evaluation attempt；全阶段不训练模型。

**Required evidence:** E1.1 独立帧率打印 evidence/report；E1.2 downstream dimension DAG/binding；E1.3 raw frame support/timestamps、expected-vs-observed count、all-frame accounting、90-bin mapping/mask；E1.4 Gate attempt、dependency lineage 和 checksums。

**Gate:** 只有 E1.4 执行 RG-01；E1.1–E1.3 是不可跳过的有序 evidence producers，不得合并成“先写 downstream、后补验证”。

**PASS condition:** 所有 required probes finite、structurally reproducible；真实 frame count 与 temporal-geometry expected count 相等；2Hz mapping 无有效 frame gap/duplicate；dimension graph 完整；RG-01 current-effective PASS。

**FAIL/BLOCKED behavior:** 保存 immutable failure；E1.1 失败立即停止 E1.2；任一子阶段 non-PASS 均不得跳到后续子阶段；不得以方案中的约 50Hz 假设、插值、reshape 或 padding-as-silence 修复。

**Next-stage unlock:** 仅 E1.4 RG-01 current-effective PASS 后允许 E2 formal extraction/cache。

## 4. E2 — Handcrafted Offline Features / Alignment / RG-02

**Research objective:** 建立 Mel/MFCC/Chroma 的 2 Hz/90-bin不可变缓存，并证明其与 Deep/波形时间轴一致。

**Entry dependencies:** current-effective RG-01 PASS、exact TimeGrid/dimension checksum、DEAM source identities，以及 current-effective `E2_DEAM_PRIMARY_SPLIT_READY`（完整1,744 population与冻结歌曲级80/10/10 split）。该完整绑定可在E1期间并行准备，但E2 formal extraction/cache不得绕过。

**Implementation tasks:** Mel、MFCC、Chroma offline extraction；raw timestamp/accounting；90-bin mean；cache immutability；cross-view automatic audit；查看前冻结 3 首 train songs；生成 waveform/四视图 human review package。

**Relevant existing sub-plan tasks:** P1 Task 4 的 coverage/mask support；实验步骤型 Step-2 extractor/cache/review tasks。

**Formal experiment:** 全部 eligible records automatic qualification + 3-song real-human alignment review。

**Required evidence:** feature configs/library versions、逐 record shapes/checksums/bin accounting、Chroma L1 diagnostics、frozen sample IDs/order、M1–M5 human records。

**Gate:** RG-02。

**PASS condition:** `AUTOMATIC_ALL_RECORDS_PASS AND MANUAL_3_SONGS_PASS`，predecessor RG-01 仍 current-effective。

**FAIL/BLOCKED behavior:** 任一 record 或 manual criterion 失败即 non-PASS；不得换 3 首样本、silent trim/pad/fill 或重写 finalized cache。

**Next-stage unlock:** RG-02 PASS 后才允许 E3 formal Sensor pre-validation。

## 5. E3 — Sensor Heads Independent Pre-validation / RG-03

**Research objective:** 只用 pseudo labels 验证 RMS、brightness、mode、key Sensor Heads 可独立学习并向对应 handcrafted encoder 回传梯度。

**Entry dependencies:** current-effective RG-02 PASS；feature/pseudo-label cache；train-only normalization/prior provenance；Sensor train/validation split。

**Implementation tasks:** standalone `E_mel/E_mfcc/E_chroma` encoders/heads；train-only baselines；mode/RMS/brightness/key losses和 curves；gradient qualification；bundle finalization。禁止 Markov、Fusion、ChatTS、Qwen。

**Relevant existing sub-plan tasks:** P3 Task 4 的 Sensor gradient subset；实验步骤型 Step-3 Sensor tasks。

**Formal experiment:** pseudo-label-only Sensor training/validation；test 不参与选模、threshold 或 Gate。

**Required evidence:** normalization/prior/class/mask/segment provenance、完整 curves、selected checkpoint、mode CCC、MSE/CE baseline comparisons、gradient ownership、SensorBundle checksum。

**Gate:** RG-03。

**PASS condition:** mode `CCC>0.7`，energy/brightness MSE 严格优于 train-only constant，key CE 严格优于 add-one train prior，且所有适用 provenance/gradient criteria PASS。

**FAIL/BLOCKED behavior:** 保存 FAIL；不得观察结果后改变伪标签粒度或使用 test 调参。

**Next-stage unlock:** RG-03 PASS 后冻结可引用 Sensor evidence，并解锁 E4。

## 6. E4 — No-LLM Main Model / RG-04, RG-05, RG-06

**Research objective:** 在没有 ChatTS/Qwen 的情况下验证完整声学主线达到或接近冻结 CRNN reference，并验证 Markov/beta 健康度。

**Entry dependencies:** current-effective RG-03 PASS；RG-01/02/03 evidence；冻结 CRNN reference/protocol/S5。

**Implementation tasks:** four-view encoders、Sensor auxiliary supervision、Markov、ViewDropout、state-conditioned fusion、post-concat projector、No-LLM VA head；训练数值/gradient qualification 的 E4-applicable subset。

**Relevant existing sub-plan tasks:** P3 Tasks 1、3–7 的 E4-applicable subset；实验步骤型 Step-4 No-LLM tasks。

**Formal experiment:** No-LLM S5 protocol，使用相同 primary validation split/label/mask/metric/selection；test 不判 Gate。

**Required evidence:** per-seed V/A CCC；CRNN protocol identity；state occupancy/A/entropy；三条完整 beta curves；Sensor/ViewDropout topology；terminal model/checksums。

**Gate:** RG-04 为强制 predecessor；applicable RG-05、RG-06 也必须形成 current-effective attempt。

**PASS condition:** `Delta_V>=-0.02 AND Delta_A>=-0.02`，并满足 applicable RG-05/06；禁止 V/A 或跨视图平均掩盖失败。

**FAIL/BLOCKED behavior:** 保存 exact FAIL/INSUFFICIENT_EVIDENCE；不得重选 CRNN、阈值、checkpoint window 或 beta observations。

**Next-stage unlock:** RG-04 PASS 才允许 E5 ChatTS/Qwen integration。

## 7. E5 — ChatTS + Qwen2.5-7B Full Pipeline

**Research objective:** 按冻结 Stage/Event/CoT contract 完成 canonical A1 7B 的 Stage-1→CoT Gate→Stage-2→final evaluation pipeline。

**Entry dependencies:** current-effective RG-04 PASS 与 applicable RG-05/06；E0 governance；pinned artifacts/feature/Sensor evidence。

**Implementation tasks:** 按下表依次执行 E5.1–E5.15；每个 exact step 单独解析 predecessor、产生不可变 run/evidence，并只解锁下一项。

**Relevant existing sub-plan tasks:** P2 的 Qwen core Tasks 1–3/5；P3 对应 Stage-1/Stage-2 context tasks；P4 Event/CoT Tasks 1/4–6；P5 RG-08/RG-09 exact-boundary Tasks 1–6。具体放置如下表。

| Exact step | Implementation scope | Required-before / Gate | Support tasks |
|---|---|---|---|
| E5.1 | Pin Qwen snapshot/tokenizer；QKB/LQ7 binding；full dual-head loader | E5.2 前关闭 pinned upstream、loader、token budget applicable subset | P2 Tasks 1–3、5 |
| E5.2 | RG-08 Stage-1 P1/P2；CRR/gradient/numeric Stage-1 qualification | Stage-1 formal entry；canonical role evidence必须 current-effective | P3 applicable Tasks；P5 RG-08 Tasks 1–3 |
| E5.3 | Stage-1 formal training；Qwen base frozen；no LoRA；no CoT loss | E5.2 PASS | Stage trainer + P3 runtime |
| E5.4 | immutable H3 handoff + Stage-1 terminal Sensor snapshot | E5.3 terminal valid | training/artifact manifests |
| E5.5 | pseudo and Stage-1-Sensor EventSequence materialization + RG-07 | CoT materialization 前 | P4 Tasks 1、5；P2 Task 5 |
| E5.6 | `COTC-SN21-DSDP-3R+` deterministic partition | train population/split current-effective | P4 Task 4 |
| E5.7 | phase/variant-matched CoT/ANSWER synthesis/materialization | E5.4–E5.6 | P4 Task 6 |
| E5.8 | RG-09 exactly-100 real-human artifact review | 任何依赖该 artifact 的 Stage 2 前；必须 PASS | P5 Tasks 4–5 |
| E5.9 | RG-08 Stage-2 P1/P3 on exact artifact/topology | Stage-2 entry；RG-09 PASS | P5 Tasks 1–3 + P3 runtime |
| E5.10 | Stage-2 logical epochs 1–3，pseudo-grounded | E5.9 PASS，新 run_id | PB-1+ runtime |
| E5.11 | epochs 4–5，Stage-1-Sensor-snapshot grounded | epoch 3 fully committed | PB-1+ runtime |
| E5.12 | designate epoch-5 terminal checkpoint only | epoch 5 committed；CS-3+ | checkpoint/runtime tasks |
| E5.13 | terminal own-Sensor/Event materialization；final Event disposition | E5.12 | FI-A+ artifact tasks |
| E5.14 | RG-10-v2 full validation population；RG-08 P4 where applicable | exact final prompt/Event/generation | P2 parser/tokenizer + P5 RG-08 |
| E5.15 | primary DEAM final evaluation | applicable E5 Gates current-effective | evaluation/metric tasks |

**Formal experiment:** canonical A1 7B pipeline runs。E5 负责形成至少一条完整、current-effective formal A1 lineage；E6 负责补齐并聚合 exact S5。E5 中已经合法完成的 seed run 可进入 E6 S5，不重跑、不覆盖。

**Required evidence:** Qwen/upstream/topology；Stage-1/P1/P2；H3/Sensor snapshot；Event/RG-07；partition/CoT；RG-09；Stage-2 P1/P3；PB/CS terminal；final Event；P4/RG-10；primary metrics与完整 dependency graph。

**Gate:** 分布式 Gate chain：`E5_RG08_STAGE1_PASS`；E5.5 RG-07；`E5_RG09_COT_PASS`；`E5_RG08_STAGE2_PASS`；E5.14 RG-10-v2 与 `E5_RG08_FINAL_PASS`。

**PASS condition:** E5.1–E5.15 全部满足其精确 predecessor；不得用 late Gate 的未来承诺提前启动 Stage 2。

**FAIL/BLOCKED behavior:** 对应 exact step 停止并终结 immutable evidence；本机资源不足创建 external continuation/new run_id；不得把 entire E5/其他无关变体回滚或篡改。

**Next-stage unlock:** current-effective canonical A1 pipeline成功后进入 E6；未完成的 exact S5 在 E6补齐。

## 8. E6 — Full Matrix / S5 / Ablations / Secondary Evaluation

**Research objective:** 完成论文级 primary matrix、S5统计和 required secondary evaluation；optional 14B 不阻塞。

**Entry dependencies:** E5 canonical A1 pipeline current-effective；所有 E6 variant/evaluation只要求自己的 dependency closure。

**Implementation tasks:**

1. 优先完成 A1 exact `COMPLETE_S5`。
2. 执行 A2、A3、A4、A5、A6、A7、A8-MEL、A8-MFCC、A8-CHROMA、A9、A10、A11、A13 的 COMPLETE_S5。
3. A5 在其 entry 前执行 firewall；A6 在其 entry 前绑定 paired same-S5 A1 replay；二者不阻塞 A1。
4. A12 等待 A1 COMPLETE_S5 + A2 COMPLETE_S5；运行 `A12-HGST-Z0-3R++`。`TRIGGERED` 时执行 A12 COMPLETE_S5，否则记录 `NOT_TRIGGERED`。
5. A1 primary evidence 完成后执行 DEAM long58 `SECONDARY_REQUIRED`。
6. PMEmo artifact/domain audit通过后，按 `PME-ZST19-A1S5-3R++` 执行 A1-S5 zero-shot secondary evaluation。
7. 资源允许且独立 catalog/Gate批准后可执行 14B scaling；否则保持 `DEFERRED_NON_BLOCKING`。

**Relevant existing sub-plan tasks:** P2 Task 4（A10）；P4 Tasks 2–3（A5/A6）；P1 Task 5（PMEmo）；variant compiler/runner/evaluation/aggregation tasks。

**Formal experiment:** exact S5 trainable variants、triggered A12、long58、PMEmo；14B是独立 optional identity。

**Required evidence:** per-seed terminal run/checkpoint/config/artifact/Gate；S5 mean/sample-std；paired deltas；A12 trigger；secondary evaluation manifests；effective-validity exclusions。

**Gate:** 每个 variant/evaluation 使用自己的 applicable Gate set；不存在“所有 E6 side work 必须完成才允许其他 E6 row运行”的总 Gate。

**PASS condition:** A1及论文声明所需 rows exact COMPLETE_S5、current-effective；required long58/PMEmo完成；A12状态为 TRIGGERED+COMPLETE_S5 或合法 NOT_TRIGGERED。

**FAIL/BLOCKED behavior:** 单一 ablation/secondary blocker只阻塞对应 row/claim；不得反向 invalidates 无依赖的 E1–E5 historical execution。依赖真正失效时由 effective-validity graph传播。

**Next-stage unlock:** paper/report aggregation；14B未完成不阻塞 7B primary closure。

## 9. Primary experimental critical path

```text
E0
↓
E1.1 / MERT真实输出帧率打印
↓
E1.2 / 按实测值反推全部下游维度
↓
E1.3 / 2Hz、90-bin时间轴验证
↓
E1.4 / RG-01 formal closure
↓
E2 / RG-02
↓
E3 / RG-03
↓
E4 / RG-04 (+ applicable RG-05/RG-06)
↓
E5.1–E5.3 / Qwen binding + RG-08 Stage1 + Stage1
↓
E5.4–E5.8 / H3 + Event + RG-07 + CoT + RG-09
↓
E5.9–E5.12 / RG-08 Stage2 + epochs1–5 + terminal
↓
E5.13–E5.15 / final Event + RG-10/RG-08 final + primary evaluation
↓
E6 / A1 COMPLETE_S5 + ablations + secondary evaluation
```

## 10. Parallel preparation lanes

| Preparation | Earliest preparation | Formal use boundary | 不得阻塞 |
|---|---|---|---|
| DEAM 1,744 primary population + frozen 80/10/10 split binding | E0后，可与E1并行 | E2 formal extraction/cache及后续完整 population use | E1.1帧率打印、E1.2维度反推、E1.3时间轴验证 |
| Qwen snapshot inventory | E0后 | E5.1 | E1–E4 |
| training fixture/CRR/numeric code | E0后 | 对应 E3/E4/E5 formal entry | 已满足前置的 earlier stage |
| PMEmo inventory/domain audit | E0后 | E6 PMEmo | E1–E5、A1 primary |
| A5 firewall fixtures | E0/P2接口可用后 | E6 A5 | canonical A1 E1–E5 |
| A6 replay schema/fixtures | E0后 | E6 A6，真实 binding 等 A1 artifact | canonical A1 E1–E5 |
| RG-09 package/import tooling | E0后 | E5.8，真实 package 等 E5.7 | E1–E4、Stage-1 |
| 14B planning | 7B设计接口稳定后 | 7B primary完成且独立批准后 | 7B primary closure |

`PARALLEL_PREPARATION != EXPERIMENT_STAGE_ENTRY`。

## 11. Re-audited OPEN blocker ownership

| OPEN blocker | experiment_stage_owner | blocks_primary_critical_path | required_before_exact_step | can_prepare_early |
|---|---|---:|---|---:|
| `MACHINE_READABLE_MANIFEST_CONFIG_BINDING` | E0 | true | E0 formal pre-flight / E1 entry | false |
| `CLEAN_GIT_PROVENANCE_PREFLIGHT` | E0 | true | first formal run and every later formal run | false |
| `OQ-DATA-DEAM-TARGET-ARTIFACT-BINDING` | E0/E2/E4 staged | true | E1.1只需registered real 45s probe identity/hash sub-record；完整primary population/split在E2前；target/annotation scope在E4前 | true |
| `DEAM_ANNOTATION_AUDIT` | E4 | true | E4 No-LLM formal target/metric use | true |
| `PMEMO_ARTIFACT_HASH_DOMAIN_AUDIT` | E6 / PMEmo | false | E6 PMEmo zero-shot evaluation | true |
| `EVALUATION_INPUT_COVERAGE_EXECUTABLE_BINDING` | E6 / secondary | false | E6 long58/PMEmo execution | true |
| `PINNED_QWEN_UPSTREAM_MANIFEST` | E5.1 | true | E5.2 RG-08 Stage-1 P1/P2 | true |
| `QWEN_EXECUTABLE_LOADER_QUALIFICATION` | E5.1 | true | E5.2 and E5.3 Stage-1 entry | true |
| `A10_PROJECTION_EQUIVALENCE_QUALIFICATION` | E6 / A10 | false | E6 A10 Stage-1/Stage-2 entry | true |
| `TOKENIZER_ANSWER_BUDGET_QUALIFICATION` | E5.1/E5.5 | true | E5.5 RG-07 and E5.7 CoT/final generation | true |
| `CRR_EXECUTABLE_QUALIFICATION` | E5.2 | true | E5.3 checkpointed Stage-1 training | true |
| `GRADIENT_FLOW_QUALIFICATION` | E3/E4/E5 horizontal | true | first formal stage using each compiled role | true |
| `NUMERICAL_POLICY_QUALIFICATION` | E3/E4/E5 horizontal | true | first formal stage using each numeric profile | true |
| `A5_MODEL_VISIBLE_INPUT_FIREWALL_QUALIFICATION` | E6 / A5 | false | E6 A5 entry | true |
| `A6_EVENT_REPLAY_BUNDLE_BINDING` | E6 / A6 | false | E6 A6 entry after paired A1 evidence | true |
| `COT_ARTIFACT_MATERIALIZATION` | E5.7 | true | E5.8 RG-09 and E5.9 Stage-2 entry | true |
| `RG08_CANONICAL_HARDWARE_EVIDENCE` | E5.2/E5.9/E5.14 | true | each exact Stage-1/Stage-2/final role boundary | true |
| `RG09_HUMAN_REVIEW_EXECUTION` | E5.8 | true | E5.9 Stage-2 entry for artifact-consuming topology | true |

## 12. Gate-driven execution checklist

- [ ] E0 evidence只检查 E1.1 所需 machine-readable primitive、MERT upstream identity、exact 45s probe、至少一条 registered real DEAM sample identity/hash 和 provenance；不把完整 population/split 或 late artifacts设为 E1.1 predecessor。
- [ ] E1.1 未 current-effective PASS 时，不生成 `DownstreamDimensionBinding`，不实例化基于假定帧率的 TimesNet/reduction/2Hz/downstream temporal dimensions。
- [ ] RG-01未 current-effective PASS 时，E2 formal work不创建 run_spec。
- [ ] RG-02未 PASS 时，E3不训练 Sensor。
- [ ] RG-03未 PASS 时，E4不训练 No-LLM。
- [ ] RG-04未 PASS 时，E5不加载 formal Qwen topology。
- [ ] E5.2 RG-08 Stage-1 roles未 PASS 时，E5.3不启动。
- [ ] E5.8 RG-09未 PASS 时，E5.9/Stage-2不启动。
- [ ] E5.9 RG-08 Stage-2 roles未 PASS 时，E5.10不启动。
- [ ] E5 final Gates/current-effective lineage未完成时，E6不启动formal matrix。
- [ ] 每个 E6 blocker只影响自己的 variant/evaluation row，除非 dependency graph证明更广传播。

## 13. Research Plan / Appendix B Traceability

| Authoritative item | Exact plan mapping | Completion meaning |
|---|---|---|
| Research Plan §8.3 Step 1：环境验证，MERT 实测帧率后反推全部下游维度 | E1.1 + E1.2 + E1.3 + E1.4 | real forward observation、derived downstream binding、2Hz/90-bin validation 与 RG-01 current-effective closure 依次完成 |
| Appendix B #1：“MERT 真实输出帧率打印验证” | E1.1 | actual frame count/fps 已打印并保存于独立 machine-readable evidence 与 human-readable report |
| Appendix B #1：“得到实测帧率并反推下游全部维度”中的反推部分 | E1.2 | `DownstreamDimensionBinding` 只由 E1.1 actual observation 与 pinned temporal geometry派生 |
| RG-01 expanded engineering qualification | E1.1–E1.4 | 冻结 RG-01 的 probe、geometry、TimeGrid、evidence 和 effective-validity要求全部关闭 |

## 14. Support-plan execution rule

执行者按本 Master 的 exact stage选择 support task，而不是整份 support plan一次性执行/一次性 PASS。每个子计划中的 P0–P5 名称只是 provenance tag；如果其旧 package-level Exit Gate 与本 Master冲突，以本 Master 的分阶段 Entry/Exit Gate为准。任何真正的语义冲突仍必须 fail closed，不能用时序重写覆盖研究合同。
