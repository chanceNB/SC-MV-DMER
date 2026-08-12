# SC-MV-DMER 实验步骤型执行计划重排设计

**状态：** APPROVED / FROZEN（Revision 2，用户于 2026-08-12 批准）

**日期：** 2026-08-12

**目的：** 将当前按 blocker 技术类别组织的 P0–P5 实施计划，重排为与《方案 E》§8.3 完全对应的实验执行顺序，同时保留全部已冻结 Foundation 语义、Research Gate、provenance 和 18 个 OPEN blocker 的关闭要求。

## 1. 重排原则

1. 新计划以“实验先后顺序”为主导航，不再以 blocker 所属技术领域作为主导航。
2. 原方案的六个实验步骤保持原顺序；仅在其前增加不可省略的 Step 0，负责正式执行所需的治理和数据基础。
3. 任何步骤都必须以前一步的当前有效退出证据为进入条件；historical PASS 不能代替 effective validity。
4. 18 个 OPEN blocker 不删除、不降级，只重新分配到最早能产生其关闭证据的实验步骤。
5. 7B 是 primary implementation；14B、DPO、Qwen2.5-Audio 和其他 contingency 工作保持 `DEFERRED_NON_BLOCKING`。
6. 旧 P0–P5 计划保留为历史审计材料，但不再作为当前执行入口。
7. 本次重排不修改 rank、loss weight、warmup、S5、split、Gate threshold、Event visibility、A1–A13、ANSWER 长度、PMEmo mapping 或 A13 objective。

## 2. 新执行主线

```text
Step 0  正式执行基础与数据预检
   ↓
Step 1  MERT 真实帧率验证
   ↓
Step 2  手工特征、伪标签与不可变缓存
   ↓
Step 3  Sensor Heads 独立预验证
   ↓
Step 4  No-LLM 四视图原型与 RG-04
   ↓
Step 5  ChatTS + Qwen 7B：接入、Stage 1、H3 handoff 与 Stage-2 入口资格
   ↓
Step 6  CoT 合成与审核 → 正式 Stage 2 / 完整训练 → A1–A13 → 外部评估与汇总
   └─ 14B 放大验证：资源允许时的独立非阻塞扩展
```

这条主线必须成为新 Master plan 的唯一推荐执行顺序。技术基础仍可在不破坏依赖关系的情况下局部并行，但任何实验 Gate 不得越级。

### 2.1 Step 5 / Step 6 的依赖消歧

《方案 E》§8.3 的 Step 5 表达“接入 ChatTS+Qwen（7B 先行），Stage 1→Stage 2”，Step 6 表达“CoT 合成管线→完整训练”。后续冻结的 `TEB-FBC-3R+++`、Scheme B+ 和 `COTC-SN21-DSDP-3R+` 已进一步规定：

- Stage 2 synthetic-supervised song 在开跑前必须拥有 phase-matched immutable approved CoT/ANSWER artifact；
- `S2_EARLY` 使用 pseudo Event，`S2_LATE` 使用 frozen Stage-1 Sensor-snapshot Event；
- 因而 late CoT artifact 只能在对应 Stage-1 Sensor snapshot 与 H3+ handoff 形成后物化；
- 缺失、错误 phase 或错误 provenance 的 artifact 是 integrity failure，不能用 zero eligibility、占位数据或临时在线生成绕过。

所以本计划对源步骤作如下执行化解释，不新增研究语义：Step 5 完成 7B 接入、正式 Stage 1、H3+ handoff、Stage-2 compiled topology 与入口资格；Step 6 先生成并审核 early/late CoT artifact，再创建新的 Stage-2 run 并执行正式 Stage 2。Step 5 中的箭头表示“Stage 1 已具备合法进入 Stage 2 的 handoff”，不表示在 CoT artifact 之前提前完成 Stage-2 optimization。

## 2.2 六步实验合同总表

| 源步骤 | 必要输入 | 本步骤必须实际完成 | 不可变输出 | 放行条件 |
|---|---|---|---|---|
| 1. MERT 帧率→下游维度 | Step 0 data/upstream identity | 真实 forward、帧率测量、2 Hz 映射、全链 shape 反推 | `DownstreamDimensionBinding` + RG-01 evidence | frame/time/shape 全部一致 |
| 2. 四视图离线提取与缓存 | Step 1 dimension checksum | MERT deep/Mel/MFCC/chroma、四类 pseudo-label、至少 3 首抽样对齐 | four-view cache、pseudo-label bundle、alignment evidence | RG-01/02 dependency 与 cache integrity 有效 |
| 3. Sensor Heads 预验证 | Step 2 cache/pseudo-label identity | standalone 训练、train-only normalization、统一 class/mask/segment、梯度验证 | standalone Sensor bundle + Gate evidence | 所有适用 Sensor criterion 有效 PASS |
| 4. No-LLM 原型 | Step 1–3 当前有效证据 | 四视图+Markov+Fusion+direct VA；CRNN 对齐比较 | No-LLM checkpoint/evidence、RG-04/05/06 evidence | No-LLM Gate PASS |
| 5. ChatTS+Qwen 7B 接入 | Step 4 PASS、pinned Qwen/tokenizer | loader/双头集成、Stage 1、H3 handoff、Stage-2 entry qualification | Qwen manifest、Stage-1 terminal、H3 package、entry qualification | 可合法物化 phase-matched CoT 并进入 Step 6 |
| 6. CoT→完整训练→消融 | Step 5 handoff/Sensor snapshot | CoT 合成与 RG-07/09、正式 Stage 2、final inference、A1–A13、外部评估 | CoT/Stage-2/final-event/experiment/aggregation bundles | applicable Gates + COMPLETE_S5 + effective validity |

后续实施计划必须把这张表展开成测试驱动任务，不能再用“一段总述”替代任一步骤。

## 3. Step 0 — 正式执行基础与数据预检

### 3.1 目标

在任何音频特征提取或模型运行前，建立 machine-readable Foundation、stable identity、DEAM primary population、冻结 split、annotation audit、run lifecycle 和 formal/debug pre-flight。

### 3.2 核心交付物

- Python project/repository baseline；仅在实施阶段取得明确授权后执行 `git init`。
- `semantic_config_hash`、`resolved_config_hash` 和 canonical JSON。
- `RunSpec`、`RunManifest`、artifact/evidence manifest 和 append-only validity registry。
- `VariantCapabilityProfile`、stage/capability catalog 和 formal CLI firewall。
- DEAM 1,744 primary excerpt manifest。
- 唯一歌曲级 80/10/10 split verification。
- DEAM annotation、label domain、mask 和 train-only normalization provenance。
- Step 0 qualification bundle。

### 3.3 归属 OPEN blocker

- `MACHINE_READABLE_MANIFEST_CONFIG_BINDING`
- `CLEAN_GIT_PROVENANCE_PREFLIGHT`
- `OQ-DATA-DEAM-TARGET-ARTIFACT-BINDING`
- `DEAM_ANNOTATION_AUDIT`

### 3.4 退出门槛

只有在 formal pre-flight 能对 dirty workspace、semantic override、split mismatch、unregistered data 和 invalid dependency 执行 fail closed 后，才能进入 Step 1。M0 仍为 `READY_TO_CLOSE`，不得由计划自动改为 PASS。

当前权威文档只命名了 `SGA-M0M20-EV-3R+`，没有逐项列出 M0–M20 的 21 条 machine-transcribable stage row。Step 0 必须先生成 source-coverage report；如仍缺 predecessor/Gate/entry/exit authority，stage catalog 发布以 `IMPLEMENTATION_BLOCKED_BY_FROZEN_CONTRACT: SGA-M0M20-EV-3R+` 停止，实施者不得推断或补造映射。

## 4. Step 1 — MERT 真实帧率验证

### 4.1 目标

使用已 pin 的 MERT artifact 对经过身份绑定的真实音频执行只读 forward，测量真实 output frame rate、层输出形状和有效 coverage；随后把实测量与冻结常量共同编译成全局 `DownstreamDimensionBinding`，反推或核验所有下游张量维度，并据此冻结 MERT→2 Hz 的 executable alignment binding。Step 1 不只是“打印帧率”，而是整个后续模型维度图的唯一入口。

### 4.2 核心任务

- 发现并 pin MERT upstream manifest、revision、file hash 和 processor/config identity。
- 对注册音频 fixture 运行 MERT，记录 input duration、sample count、hidden frame count 和实际 frame rate。
- 验证第 5/6 层输出与门控聚合入口。
- 编译 MERT frame 到 canonical 2 Hz TimeGrid 的边界/池化映射。
- 生成 RG-01 evidence；任何实测值不得被旧方案中的 50 Hz 假设覆盖。
- 将实测结果绑定为 TimesNet period/downsampling 设计的唯一 executable input。
- 生成不可变 `manifests/dimensions/mert-downstream-dimensions-v1.json`，至少记录：
  - source duration、source sample rate、MERT input sample count；
  - MERT revision、processor identity、selected layer、hidden size；
  - output frame count、actual frame rate、effective stride、frame-center boundary；
  - TimesNet input length、period/downsampling/pooling 参数及其推导来源；
  - canonical `T = 90`、2 Hz window boundary 和 MERT-frame→TimeGrid index map；
  - `X_deep [T,768]`、`E_deep [T,256]`；
  - `X_mel [T,128]`、`X_mfcc [T,40]`、`X_chroma [T,12]` 与各 `E_v [T,256]`；
  - `γ_v/γ̃_v [T,N]`、state-consistency bias `[T,T]`；
  - fusion concat `[T,1024]`、`F_fused [T,256]`；
  - Sensor output、Qwen time-series token、VA output `[T,2]` 和 full ANSWER `T=90` 的 shape contract。
- 上述字段必须逐项注明 `MEASURED`、`DERIVED` 或 `FROZEN_CONSTANT` 及其 source contract。不能把方案文档中的示意 shape 当成实测 evidence。
- 后续 Step 2–6 的 config/schema/model build 必须引用该 dimension manifest checksum；禁止重新独立计算或静默覆盖维度。

### 4.3 退出门槛

- 所有 shape/rate 为 finite 且可复现。
- 2 Hz 映射完整覆盖有效输入，不把 padding 当作 silence。
- RG-01 具备 immutable evidence bundle。
- `DownstreamDimensionBinding` 能从 MERT 实测量和冻结常量完整解析到最终 `T=90` VA/ANSWER output，且所有 shape assertion 通过。
- 任意下游 dimension mismatch 必须以 `DOWNSTREAM_DIMENSION_BINDING_MISMATCH` fail closed，并指出产生 mismatch 的最短依赖路径。
- 未确认 frame rate、未终结 dimension manifest 或 checksum 不匹配时，不得创建 feature cache、模型 checkpoint 或 formal run。

## 5. Step 2 — 手工特征、伪标签与不可变缓存

### 5.1 目标

按冻结定义离线提取并缓存四视图（MERT deep、Mel、MFCC、chroma）与四类 Sensor pseudo-label，统一映射到 2 Hz TimeGrid，形成 content-addressed、versioned、不可覆盖的 feature cache。

### 5.2 必要输入

- Step 0 的 stable `dataset_id/song_id/sample_id`、primary split、source-audio checksum 和 train-only target-normalization provenance；
- Step 1 `DownstreamDimensionBinding` 的 identity/checksum；
- 冻结的 RG-02 feature definition、TimeGrid、padding 与 bin-membership contract。

### 5.3 核心任务

- Mel：冻结 STFT/mel 参数、dB conversion 和 2 Hz binning。
- MFCC：冻结系数集合、c0 语义和 2 Hz binning。
- Chroma：冻结 CQT 参数、逐帧 L1 normalization 和 2 Hz binning。
- Deep：使用 Step 1 pin 的 MERT/layer/gating/alignment contract 对冻结 population 做离线抽取；保存原始 MERT frame coverage 与映射后的 2 Hz deep view，禁止另行估算帧率或改变 pooling。
- Pseudo-label：RMS、spectral centroid、mode strength、key segment label。
- 四视图与 waveform 的抽样对齐可视化；可视化是 qualification evidence，不改变算法。
- Cache key 覆盖 dataset/sample identity、source hash、feature definition、implementation version 和 TimeGrid binding。
- Final cache content/version immutable；修改 definition 必须生成新 version。
- 建立 `InputCoverageMask`、`TargetValidityMask`、Sensor/Event validity 的 executable binding。

抽样可视化不得只画 feature heatmap。每首抽检歌曲必须在同一时间轴上显示 waveform、2 Hz bin boundaries、Mel/MFCC/chroma aggregation 结果和四类 pseudo-label segment/boundary，并能从图中回查 `song_id`、source checksum、feature-cache checksum 与绘图配置。至少 3 首注册歌曲的选择规则必须在查看图像前冻结，不能只挑“看起来对齐”的歌曲。

### 5.4 不可变输出

- `FourViewFeatureCacheManifest`：MERT deep/Mel/MFCC/chroma 逐 tensor identity、shape、dtype、finite status、content checksum、feature definition/version、TimeGrid/dimension dependency；
- `PseudoLabelBundleManifest`：RMS/brightness/mode/key 的 rule version、train-only statistics、class/segment/mask identity 与 checksum；
- `HandcraftedAlignmentEvidenceBundle`：冻结抽样 frame、至少 3 首可视化、逐 bin numerical audit 和 machine-readable verdict；
- cache version 封存记录：相同 version 的内容不可重写，任何 definition/content 改变必须生成新 version。

### 5.5 归属 OPEN blocker

- `EVALUATION_INPUT_COVERAGE_EXECUTABLE_BINDING`

### 5.6 失败边界与退出门槛

- RG-02 feature-binning evidence 有效。
- 所有 cached tensor 的 shape、dtype、finite、time boundary 和 checksum 通过 schema。
- Chroma L1 normalization、窗口归属和 padding policy 可机器判定。
- 任一歌曲的 bin coverage gap/overlap、dimension checksum mismatch、非 finite 值、source/cache checksum 漂移或 pseudo-label provenance 缺失均 fail closed；不得跳过坏样本后继续生成“完整”cache。
- 未登记或被修改的 feature cache 不能进入 Step 3。

## 6. Step 3 — Sensor Heads 独立预验证

### 6.1 目标

在不连接主干 Markov/Fusion/Qwen 的情况下，独立训练和验证 Sensor Heads，证明 pseudo-label、mask、loss 和输出 bundle 可用于后续事件符号化。

### 6.2 必要输入

- Step 2 封存的 handcrafted feature cache、pseudo-label bundle、train-only normalization statistics 与统一 class/mask/segment contract；
- 对应 `DownstreamDimensionBinding` checksum；
- 冻结 primary train/validation membership，test 不参与训练、调粒度、选模或 Gate 判定。

### 6.3 核心任务

- 构建只消费 `E_mel`、`E_mfcc`、`E_chroma` 的 standalone Sensor fixture。
- 验证 energy、brightness、mode-strength regression 和 key classification。
- Normalization statistic 只来自 train membership。
- 统一 class/mask/segment contract、zero-frequency smoothing 和 CE checkpoint selection。
- 输出完整 learning curve、selected checkpoint、metric、pseudo-label/version provenance 和 Sensor bundle checksum。
- 不允许 Sensor 从 `F_fused` 或 Markov output 派生。
- 验证 Sensor loss 能反向更新对应 handcrafted encoder。

独立预验证必须为每个 head 保存完整 train/validation learning curve、eligible denominator、selected checkpoint rule 和逐 seed 结果。Mode 是冻结的连续 `[-1,1]` regression、key 是 12-way tonic CE；不能为迎合源方案旧表中的简写阈值而把当前 head semantics 改回另一种任务。

### 6.4 不可变输出

- `StandaloneSensorRunManifest` 与 terminal training records；
- `StandaloneSensorBundle`：对应 handcrafted encoder + Sensor Heads state、pseudo-label/class/mask/segment/normalization identity、selected checkpoint checksum；
- `SensorPrequalificationEvidenceBundle`：各 head 原始与聚合指标、学习曲线、CE selection、梯度到对应 encoder 的证据、machine-readable criterion results；
- 供后续 EventSequence 引用的 Sensor bundle identity/hash；此时只授权后续生成器引用，不把 standalone 预测静默当作 final inference source。

### 6.5 失败边界

原方案中的“指标不好就调整伪标签粒度”不得在 formal observation 后自动执行。若冻结 criterion 失败，应生成 immutable FAIL evidence；任何 pseudo-label rule 修改必须先走 `DECISIONS.md`、研究规范同步和 version bump。

### 6.6 退出门槛

- 所有适用 Sensor criterion 有机器可读 PASS/FAIL。
- Sensor bundle identity 可被 EventSequence provenance 引用。
- 独立预验证 PASS 后才允许进入 No-LLM 主干集成。
- 任一适用 head 未达 criterion、encoder 未收到所需梯度、test 泄漏、mask/segment 不一致或 Sensor bundle checksum 不完整时均不得进入 Step 4。

## 7. Step 4 — No-LLM 四视图原型与 RG-04

### 7.1 目标

实现不加载 Qwen 的完整声学主线：MERT/TimesNet + 三手工视图 Encoder + shared-A Markov bridge + state-biased fusion + direct VA head，并以冻结 CRNN reference 执行 RG-04。

### 7.2 必要输入

- Step 1 的 MERT/time/dimension binding；
- Step 2 的 feature cache、mask 与 pseudo-label identities；
- Step 3 当前有效的 Sensor prequalification evidence 与 Sensor bundle；
- 在观察 No-LLM 结果前已冻结的 CRNN reference、primary validation population、S5 和 metric/selection protocol。

### 7.3 核心任务

- 实现 deep、Mel、MFCC、chroma 四视图 encoder。
- 实现 per-view prototype、shared `A/π`、Markov recursion 和 state-collapse diagnostics。
- 实现 `β_mel`、`β_mfcc`、`β_chroma` state bias 与 anchored cross-view fusion。
- Sensor branch 在原始 handcrafted `E_v` 处分叉；View Dropout 只作用于进入 Markov/Fusion 的 handcrafted path；deep view 永不 dropout。
- 实现 direct No-LLM VA head；不得加载 Qwen hidden backbone 或 LM head。
- 冻结 CRNN reference、相同 primary validation split、label domain、mask、metric implementation、selection protocol 和 S5。
- 执行 RG-04：逐维 `ΔV >= -0.02 AND ΔA >= -0.02`，禁止使用 V/A 平均值掩盖失败。
- 同步执行 RG-05 state collapse 和 RG-06 beta stability evidence。

### 7.4 不可变输出

- `NoLLMModelBundle`：exact topology、dimension/config identities、terminal checkpoints 与 checksums；
- `CRNNReferenceBundle`：冻结 baseline provenance，不允许看到结果后重定义；
- RG-04 逐维 `CCC_CRNN`、`CCC_noLLM`、`DeltaV/DeltaA` 与 `REACHED/NEAR` 状态；
- RG-05 完整 state occupancy/transition evidence 与 RG-06 三条完整 beta learning curve；
- Sensor/View-Dropout gradient-topology evidence，证明 Sensor supervision 未被 `[VIEW-DROP]` 污染。

### 7.5 失败边界与退出门槛

No-LLM Gate PASS 后才允许 LLM integration。No-LLM head 是 Gate/baseline head，不是正式 Full LLM 模型的永久 ensemble branch。

任一维 RG-04 失败、RG-05/06 适用 Gate 非 PASS、CRNN protocol mismatch、No-LLM 意外加载 Qwen、或 formal run 使用 test 选模均 fail closed。失败只能保留证据并修复实现/依赖；不能观察结果后改阈值、平均 V/A 掩盖失败或重选 baseline。

## 8. Step 5 — ChatTS + Qwen 7B 接入、Stage 1 与 Stage-2 入口资格

### 8.1 目标

在 No-LLM 主线通过后，绑定 exact Qwen snapshot/tokenizer，组装 Full LLM 双头 7B 模型，完成 7B executable integration、正式 Stage 1、immutable H3+ handoff 和 Stage-2 entry qualification。正式 Stage-2 optimization 不在 phase-matched CoT artifact 获批前启动。

### 8.2 必要输入

- Step 4 当前有效的 No-LLM Gate predecessor 与 acoustic-module checkpoint/provenance；
- exact Qwen upstream snapshot、tokenizer、chat template、license 与本地可解析 artifact；
- Step 0 的 VariantCapabilityProfile、TrainingStageDefinition、formal/debug firewall、S5 lineage；
- Step 1–3 的 dimension/cache/Sensor identities。

### 8.3 核心任务

- Pin Qwen upstream snapshot、tokenizer、chat template、license 和 hashes。
- 组装 Full LLM 双头拓扑并完成 loader、projection、forward/backward、gradient ownership 与 numerical qualification。
- Stage 1：无 LoRA；按冻结 parameter/mode/loss profile 和 S5 lineage 训练适用外部模块，形成 terminal Stage-1 Sensor snapshot。
- H3+ immutable typed checkpoint handoff：每个 Stage-1 terminal run 生成独立 handoff package；不得把 terminal run 重新打开成 Stage 2。
- 编译 Stage-2 topology：冻结 base，按 `LQ7-R16A32-ZD-NF4-3R+` 和 `QKB-SPMS-3R++` 加载 QLoRA；这里只完成 load/resident/optimizer/loss/capability/artifact dependency 检查，不用缺失的正式 CoT 占位开跑。
- Full LLM 双头：hidden states→VA head；LM head→CoT/ANSWER；parser→`Y_answer`；`Y_va` 与 `Y_answer`→`L_consist`。
- 对 PB-1+、CS-3+、CRR、durable TrainingState、exact-once sampler、loss accumulation、optimizer/scheduler 和 BF16/FP32 island 完成静态编译与适用的 executable qualification；真正的 5-epoch PB/CS 运行证据在 Step 6 形成。
- Tokenizer/ANSWER budget 和 full `T=90` RG-10-v2 qualification。
- RG-08 分阶段完成：Step 5 负责可在无正式 CoT population 下合法完成的 P1/P2 Stage-1、P1 Stage-2 load/resident 及 A10 applicable role；需要 exact Stage-2 artifact/完整 accumulation/generation 的 P3/P4 在 Step 6 artifact 获批后完成。本机证据不能替代 canonical RTX 4090 authority。

### 8.4 不可变输出

- `QwenUpstreamModelManifest` 与 executable loader qualification；
- `FullLLMTopologyQualificationBundle`：双头、parser、projection、parameter/mode/gradient、numeric 与 memory role evidence；
- 每个适用 S5 lineage 的 terminal `Stage1RunManifest`、Stage-1 Sensor snapshot 与 checksum；
- `H3StageHandoffPackage`：model state、typed schema、source run、semantic identity、target Stage-2 expectations 与 checksum；
- `Stage2EntryQualificationBundle`：compiled topology/capabilities、artifact dependency list、已完成与待 Step 6 完成的 RG-08 roles；
- `ANSWERBudgetAndParserQualificationBundle`；qualification fixture 与 formal artifact 必须具有不同 identity，前者不得进入 paper evidence。

### 8.5 归属 OPEN blocker

- `PINNED_QWEN_UPSTREAM_MANIFEST`
- `QWEN_EXECUTABLE_LOADER_QUALIFICATION`
- `A10_PROJECTION_EQUIVALENCE_QUALIFICATION`
- `TOKENIZER_ANSWER_BUDGET_QUALIFICATION`
- `CRR_EXECUTABLE_QUALIFICATION`
- `GRADIENT_FLOW_QUALIFICATION`
- `NUMERICAL_POLICY_QUALIFICATION`
- `RG08_CANONICAL_HARDWARE_EVIDENCE`

### 8.6 失败边界与退出门槛

- Stage-1 topology/run 与 Stage-2 compiled topology、trainability、loss activation、handoff 和 replay policy 均符合冻结合同。
- Final model 不默认 ensemble No-LLM output。
- 双头 qualification `PredictionBundle` 保存两路 raw result、parsed result 和 consistency evidence。
- Step 5 所属的 RG-08 roles 当前有效；P3/P4 等 artifact-dependent roles 被明确标为 `PENDING_STEP_6_ARTIFACT`，不能伪造 PASS，也不构成越级启动 Stage 2 的许可。
- H3 package、Stage-1 Sensor snapshot、Qwen/tokenizer/loader identity 和 Stage-2 artifact requirements 全部可由 Step 6 解析。

如果缺少 phase-matched approved CoT/ANSWER artifact，系统必须以 `STAGE2_BLOCKED_BY_COT_ARTIFACT` 停在入口；不得使用空 CoT、临时在线 teacher、复制另一 phase/variant artifact、qualification fixture 或 zero eligibility 启动 formal Stage 2。

## 9. Step 6 — CoT 合成、正式 Stage 2、消融、外部评估与汇总

### 9.1 目标

基于 Step 5 的 Stage-1 Sensor snapshot 与 H3 package，先物化并批准 phase/variant-matched CoT artifact，再创建新的 Stage-2 run 完成 7B primary training；随后执行 A1–A13 消融、long58/PMEmo evaluation、RG-09 人工审查和 paper-ready aggregation。

### 9.2 必要输入

- Step 5 的 current-effective Stage-1 terminal runs、Sensor snapshots、H3 packages、pinned Qwen/tokenizer 与 Stage-2 entry qualification；
- Step 2–4 的 feature/Sensor/No-LLM dependencies；
- `COTC-SN21-DSDP-3R+`、Scheme B+、PB-1+、CS-3+、FI-A+ 和 A1–A13 frozen contracts；
- 对每个 variant/seed/phase 明确的 Event producer、consumer、visibility、artifact allowlist 与 predecessor graph。

### 9.3 核心任务

- `COTC-SN21-DSDP-3R+` deterministic coverage partition。
- 先生成 `S2_EARLY` pseudo-Event 与 `S2_LATE` frozen Stage-1 Sensor-snapshot EventSequence；每条保存 event source、Sensor bundle identity/hash、symbolizer version、segment boundaries 与 population provenance。
- 再生成 phase/variant-matched CoT/ANSWER artifact；A2 event-free；A5 shadow-only；A6 paired A1 replay；A8 child-scoped；synthetic-supervised songs 必须全覆盖且 native-DEAM VA-only songs 不得伪造 teacher CoT。
- 对实际物化 artifact 执行 RG-07 token-length qualification、RG-09 固定 100 条人工审核和 immutable CoT manifest；artifact 未获得 applicable approval 前不得进入正式 Stage 2。
- A5 model-visible firewall qualification。
- A6 Event replay bundle binding。
- 在 exact approved artifact 上完成剩余 RG-08 P3/P4 等 applicable required roles；不满足本机资源时保留 evidence/handoff，由新 external run 衔接。
- 每个 Stage 2 使用新的 run_id，从 H3 package 进入；epochs 1–3 使用 early artifact，epoch 3 commit 后 epochs 4–5 使用 late artifact，phase switch 不重置 optimizer/scheduler/LoRA/RNG。
- 只 designation logical-epoch-5 terminal checkpoint；完成 FI-A+ final Event materialization、VA inference、autoregressive generation、RG-10-v2 与完整 PredictionBundle。
- 7B A1 canonical base 的完整 S5 formal execution；每个 seed 的 Stage 1/CoT/Stage 2 lineage 必须闭合。
- A1–A13 使用 canonical base + semantic diff + dependency closure；需要独立 Stage-1 snapshot/CoT artifact 的 variant 必须执行自己的依赖链，不能借 A1 evidence；每个 final evidence 使用统一 S5。
- long58 sliding-window secondary evaluation。
- PMEmo 2019 artifact/hash/domain audit与 A1-S5 zero-shot evaluation。
- RG-09 双独立人工 reviewer 按冻结规则 adjudication；Codex/LLM 不得替代人工 authority。
- Formal aggregation 必须检查 exact S5、active Gate definition、effective predecessor chain 和 invalidation registry。

### 9.4 不可变输出

- per phase/variant/seed `EventSequenceManifest` 与 `CoTAnswerArtifactManifest`；
- RG-07、RG-09、RG-10 和 remaining RG-08 evidence bundles；
- per seed `Stage2RunManifest`、terminal checkpoint、Final Event/PredictionBundle 与 artifact checksums；
- `ExperimentEvidenceMatrix`：A1–A13 semantic diff、dependency closure、applicable Gate/evidence、S5 completeness；
- long58/PMEmo evaluation runs 与 `PaperAggregationManifest`，所有 source_run_id/effective validity 可追踪。

### 9.5 归属 OPEN blocker

- `A5_MODEL_VISIBLE_INPUT_FIREWALL_QUALIFICATION`
- `A6_EVENT_REPLAY_BUNDLE_BINDING`
- `COT_ARTIFACT_MATERIALIZATION`
- `PMEMO_ARTIFACT_HASH_DOMAIN_AUDIT`
- `RG09_HUMAN_REVIEW_EXECUTION`

### 9.6 失败边界与退出门槛

- 任一 required CoT/Event artifact 缺失、未批准、wrong phase/variant/seed/provenance、被 invalidated，均禁止对应 Stage-2 entry；
- PB switch、terminal selection、final Event source、S5 completeness、Gate applicability 或 dependency validity 不符时，run/evidence 不得进入 formal aggregation；
- human review 未完成时 RG-09 保持 non-PASS，不允许 Codex 补签；canonical RTX4090 evidence 未完成时保持 execution blocker，不能由本机 debug PASS 替代；
- A1 primary 完整证据与所有声明所需 A-variant/evaluation current-effective 后，才生成 paper-ready aggregation；14B 不计入此退出条件。

### 9.7 14B 边界

14B 放大验证只能在 7B primary evidence 完成且资源可用后，以独立 catalog identity、run 和 Gate applicability 执行。它不是 Step 6 的退出条件，不得阻塞 Foundation 或 7B paper evidence。

## 10. 18 个 OPEN Blocker 完整映射

| Blocker | 新归属 |
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
| `RG08_CANONICAL_HARDWARE_EVIDENCE` | Step 5（P1/P2 与 entry roles）+ Step 6（artifact-dependent P3/P4/final roles） |
| `A5_MODEL_VISIBLE_INPUT_FIREWALL_QUALIFICATION` | Step 6 |
| `A6_EVENT_REPLAY_BUNDLE_BINDING` | Step 6 |
| `COT_ARTIFACT_MATERIALIZATION` | Step 6 |
| `PMEMO_ARTIFACT_HASH_DOMAIN_AUDIT` | Step 6 |
| `RG09_HUMAN_REVIEW_EXECUTION` | Step 6 |

Step 1、Step 3 和 Step 4 仍受 RG-01、Sensor Gate、RG-04、RG-05、RG-06 等冻结 Gate 约束；这些 Gate 的设计已经冻结，因此即使不在 18 个 OPEN blocker register 中，也必须产生相应 executable evidence。

## 11. 《方案 E》附录 B 工程验证清单覆盖

附录 B 的“动手前按序完成”解释为：10 项验证按依赖顺序分布在 Step 1–6，并在各自正式依赖首次使用前完成；它不表示必须在尚未构建 Sensor、No-LLM、CoT 和 parser 时，于 Step 1 一次性运行全部 10 项。Step 1 的特殊责任是完成第 1 项，并产生后续 2–10 项共同依赖的全局维度合同。

| 附录 B 项目 | 新计划归属 | 当前冻结解释 |
|---|---|---|
| 1. MERT 真实输出帧率打印验证 | Step 1 | 实测帧率并生成 `DownstreamDimensionBinding`，反推/核验全链维度；未通过不得进入 Step 2 |
| 2. 手工特征分箱对齐抽检 | Step 2 | 至少 3 首注册歌曲的 waveform/四视图 2 Hz alignment evidence，服从 RG-02 |
| 3. Sensor Heads 独立预训练 | Step 3 | 使用当前冻结 Sensor Gate、train-only normalization、统一 class/mask/segment contract 和 CE selection；不得观察失败后静默改变伪标签规则 |
| 4. No-LLM 原型精度摸底 | Step 4 | 使用冻结 RG-04：逐维 `ΔV >= -0.02 AND ΔA >= -0.02`，而不是模糊的“达到或接近” |
| 5. Markov 状态使用率监控 | Step 4 | 使用已冻结 RG-05 operational definition 和完整 state-usage evidence；不得仅沿用旧文档的单一示意阈值 |
| 6. `β_v` 学习曲线检查 | Step 4 | 使用 RG-06：最后 3 个正常 validation checkpoint 全部 `β>0` 且逐视图 `CV<=0.10`；`β_chroma>β_mel` 仅为 Directional Hypothesis，不影响 PASS |
| 7. Event text token length | Step 5/6 | Step 5 绑定 tokenizer/budget contract；Step 6 对真实 early/late artifact 执行 RG-07，不使用未 pin tokenizer 的估算 token 数 |
| 8. 4090 memory test | Step 5/6 | Step 5 完成合法的 load/Stage-1/entry roles，Step 6 在批准 artifact 上完成 Stage-2/generation roles；全部 applicable required roles 的 canonical max `<22*2^30` byte；本机非 4090 结果只作 local/debug evidence |
| 9. CoT synthesis quality review | Step 6 | 使用 RG-09 的 immutable 100-record package、两名独立真实人工 reviewer 和适用 adjudication；Codex/LLM 不得代替人工 |
| 10. `L_consist` ANSWER parser test | Step 5/6 | Step 5 用明确标识的 qualification fixture 验证 RG-10-v2 full `T=90` parser；Step 6 对正式 generation population 形成实际 RG-10-v2 Gate evidence与 consistency evidence，fixture 不进入 paper result |

如附录 B 的早期示意阈值与后来冻结 Gate 定义不同，以 `DECISIONS.md`、`RESEARCH_SPEC.md` 和 current Gate catalog 中的后续冻结定义为准；这属于既有 authority precedence，不是本次重排新增科研决定。

## 12. 新计划文件结构

当前执行入口将重写为：

- `docs/superpowers/plans/2026-08-12-sc-mv-dmer-execution-master-plan.md`
- `docs/superpowers/plans/2026-08-12-sc-mv-dmer-step-0-foundation-preflight-implementation-plan.md`
- `docs/superpowers/plans/2026-08-12-sc-mv-dmer-step-1-mert-frame-rate-implementation-plan.md`
- `docs/superpowers/plans/2026-08-12-sc-mv-dmer-step-2-handcrafted-feature-cache-implementation-plan.md`
- `docs/superpowers/plans/2026-08-12-sc-mv-dmer-step-3-sensor-prequalification-implementation-plan.md`
- `docs/superpowers/plans/2026-08-12-sc-mv-dmer-step-4-no-llm-prototype-implementation-plan.md`
- `docs/superpowers/plans/2026-08-12-sc-mv-dmer-step-5-qwen-7b-stage1-handoff-implementation-plan.md`
- `docs/superpowers/plans/2026-08-12-sc-mv-dmer-step-6-cot-stage2-ablation-evaluation-implementation-plan.md`

现有 blocker-oriented P0–P5 计划在重写实施计划时移动到：

`docs/superpowers/plans/archive/2026-08-12-blocker-oriented/`

归档文件保留原内容，不作为 current execution entry，也不参与新 Master 的推荐顺序。

## 13. 各子计划的写作合同

每份 Step 计划必须遵守 `superpowers:writing-plans`：

- 中文 Goal、Architecture、Tech Stack 和 Global Constraints；
- 精确 file path 与单一职责；
- 相邻任务共享的精确 interface signature；
- 对所有可测试软件先写失败测试；
- 精确 command、expected failure、minimal implementation 和 pass verification；
- 真实 data/model/hardware 未知项写为 `PRE-FLIGHT DISCOVERY TASK`；
- 遇到冻结合同缺失或冲突时输出 `IMPLEMENTATION_BLOCKED_BY_FROZEN_CONTRACT: <contract_id>`；
- 每个任务给出未来 commit boundary，但本规划阶段不执行 commit。

## 14. 设计闭包判定

- 原方案 §8.3 六步顺序：完整覆盖。
- Step 2–6 input/task/output/Gate/failure boundary：全部显式化。
- 附录 B 10 项工程验证：10/10 明确归属；第 1 项包含全局 `DownstreamDimensionBinding`。
- 18 个 OPEN blocker：18/18 唯一归属。
- No-LLM 与 A10：明确分离。
- 7B primary 与 14B optional：明确分离。
- Sensor gradient/event provenance/View Dropout topology：保持冻结。
- Research Gate/effective validity：保持冻结。
- Stage 2 不早于 phase-matched approved CoT artifact：依赖倒置已消除。
- 新增科研 Decision：0。
- Foundation semantic contradiction：0。

**设计结论：** `REVISION_2_APPROVED_FOR_WRITING_PLANS`。对应中文 Master 与 Step 0–6 实施计划已写入 `docs/superpowers/plans/`；本次仅持久化计划，不实施代码、不训练、不执行 git 操作。
