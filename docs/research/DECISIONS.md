# SC-MV-DMER Decision Log

本文件记录已经明确批准、会影响研究正确性或工程可复现性的长期决定。除非通过新的、可追溯的决定条目正式取代，所有状态为 `Accepted` 的决定均为项目约束。

## DEC-0001：冻结唯一的歌曲级 Primary Split

- 日期：2026-08-11
- 状态：Accepted
- 决策类型：User-approved Research Governance Decision
- 适用范围：DEAM 主实验及其所有正式 baseline、主模型、消融和 multi-seed 实验

### 背景

研究方案规定使用歌曲级 `80/10/10` 的 train/validation/test 划分，并以 5 个随机种子报告均值与标准差，但没有明确 5 个种子是否允许改变数据集成员。若不同种子重新划分数据，训练随机性与数据划分差异会被混入同一组统计，削弱结果的可解释性和可复现性。

### 决定

1. 项目只生成并冻结一份歌曲级 `80/10/10` primary split manifest。
2. 正式实验的 5 个随机种子只控制模型初始化、采样、shuffle、Dropout 等训练随机性。
3. 5 个随机种子不得改变 train、validation 或 test 的数据集成员。
4. 正式实验开始后，不得因实验指标、预期结论或模型表现重新生成 primary split。
5. 所有 baseline、主实验、A1-A13 消融和正式 multi-seed 统计必须引用同一份 primary split manifest。
6. 如需评估数据划分敏感性，必须建立独立的 `split-robustness` 实验；其结果不得与主实验的 5-seed 统计混合。

### 变更控制

发现数据损坏、重复样本、身份映射错误或歌曲级泄漏时，不得静默修改 manifest。必须先保存问题证据，再创建新的决定条目说明原因、受影响实验、修订方式和需要作废或重跑的结果。

## DEC-0002：Manifest-first 分阶段单仓库

- 日期：2026-08-11
- 状态：Accepted
- 决策类型：User-approved Engineering Architecture Decision
- 适用范围：M0-M20 全部阶段及所有正式或诊断性实验

### 决定

1. 项目采用 Manifest-first 分阶段单仓库作为总体工程方向。
2. Git 仓库与外部数据根目录分离。源码不得硬编码绝对数据路径；所有数据、特征缓存、检查点和大型产物必须通过可配置的 data root 及受控路径解析规则定位。
3. 每次实验必须拥有唯一 `run_id`、最终解析配置（resolved config）和可追踪的运行清单。
4. 运行清单采用两阶段生命周期：
   - 运行开始时生成 `run_spec`；
   - 运行成功或失败结束时生成最终 `run_manifest`。
5. `run_manifest` 至少记录：Git commit 与 dirty state、数据 fingerprint、split fingerprint、feature cache 版本、seed、最终解析配置及其 hash、运行环境、启动命令、指标、checkpoint checksum 与 artifact checksum。
6. 已结束的 run 是不可变记录，不允许覆盖修改。任何修正、补跑或配置变化都必须创建新的 `run_id`。
7. 当前阶段不引入 Hydra、DVC、MLflow 等完整实验平台。配置、metrics、manifest 与 artifacts 接口必须保持机器可读、平台无关，并允许未来通过适配器接入 MLflow 等平台。
8. M0-M20 的所有阶段继续受 Research Gates 控制；run 成功结束不等同于对应 Research Gate 自动通过。

### 后续文档约束

本决定必须进入后续 `ARCHITECTURE.md` 和 Foundation Design Spec，并在其中细化路径解析、配置解析、manifest schema、不可变性、失败终态、checksum 及适配器边界。

## DEC-0003：双重配置身份与受控配置解析

- 日期：2026-08-11
- 状态：Accepted
- 决策类型：User-approved Reproducibility Architecture Decision
- 适用范围：所有 formal/debug run、A1-A13 消融、baseline、multi-seed、cross-dataset 与 scaling 实验

### 决定

1. 配置输入采用分层 YAML，最终运行保存严格校验后的 resolved config 及规范化 JSON 快照。
2. 每个 run 至少计算两个 SHA-256：
   - `semantic_config_hash`：标识论文意义上的具体实验实例；包含研究语义以及该实例的 seed，不包含机器、路径、时间戳或纯 runtime 参数。
   - `resolved_config_hash`：覆盖最终完整 resolved config，用于复现具体执行。
3. 同一研究实验只更换 runtime profile 时，`semantic_config_hash` 必须保持不变；`resolved_config_hash` 可以变化。
4. `RESEARCH_SPEC.md` 是研究语义的人类可读 Source of Truth；`configs/research/defaults.yaml` 仅是机器可执行映射。冻结研究语义的变更必须依次经过 `DECISIONS.md`、`RESEARCH_SPEC.md`、config/schema 更新和 version bump。
5. run mode 分为 `formal` 与 `debug`。formal CLI 只允许覆盖 runtime allowlist；研究变量必须由版本化 experiment overlay 表达。debug 可显式允许研究语义 CLI override，但不得进入正式 paper result aggregation。
6. hash 基于确定性的 canonical JSON，而非原始 YAML 文本。canonicalization 必须固定 key 顺序、UTF-8、bool/null、数值与 enum 表示，并排除不属于相应身份的字段。
7. `run_spec` 必须保存完整配置 provenance chain、schema version、CLI overrides、resolved config、两个配置 hash，以及相对主模型的 `semantic_diff.json`。
8. schema 校验必须覆盖 unknown key、类型、必填字段和跨字段研究不变量；具体模块依赖只在后续 A1-A13 设计获得批准后加入，不得提前猜测。
9. 配置版本向前演进。旧 run 保留创建时的 `config_schema_version`、`research_spec_version` 与原始 snapshot；迁移必须显式执行且不得覆盖历史记录。

## DEC-0004：仓库、数据身份与派生产物不可变性

- 日期：2026-08-11
- 状态：Accepted
- 决策类型：User-approved Data and Artifact Governance Decision
- 适用范围：数据集、上游模型、四视图特征、sensor/event/CoT 派生数据及所有 run

### 决定

1. Git 仓库保存源码、配置、schema、manifest、研究文档和受控汇总；大型数据与 run 产物保存在可配置的外部 data root。
2. 数据区严格分为 `raw`、`interim`、`processed`、`features`、`pseudo_labels`、`upstream_models`、`runs` 和 `tmp`。`raw` 不允许原地修改。
3. 每个外部预训练模型必须由 upstream model manifest 登记精确来源、revision、文件 checksum、许可与本地验证状态。formal run 禁止引用未登记或可漂移的 `latest` 模型。
4. sensor labels、symbolic events 与 CoT targets 使用相互独立的版本和 manifest，并保存父级 fingerprint、生成配置/schema、生成代码版本及内容 checksum。上游变化必须产生新的下游版本，不得原地覆盖。
5. 项目维护稳定的 `dataset_id`、`song_id`、`sample_id` 身份契约。身份不得依赖本机路径或可变文件名；同一歌曲的所有片段共享 `song_id`，split 以 `song_id` 为分组边界。
6. feature cache 在 finalize 后内容与版本均不可变。任何输入数据、上游模型、特征定义、提取代码或 schema 变化都必须创建新 cache version；formal run 不得引用 `latest` 或混合多个未声明版本。
7. 未完成的派生数据或 cache 只能存在于 staging 状态。只有完整性校验、shape/dtype/time-grid 审计和 checksum 通过后才能原子化登记为 finalized version。

## DEC-0005：冻结模块边界与接口流程

- 日期：2026-08-12
- 状态：Accepted
- 决策类型：User-approved Model and Interface Architecture Decision
- 适用范围：四视图 Encoder、Sensor、View Dropout、Markov、Fusion、Event、No-LLM Gate、Full LLM 及 PredictionBundle

### 决定

1. Sensor Heads 直接消费未经 View Dropout 修改的 `E_mel`、`E_mfcc`、`E_chroma`；`L_sensor` 必须反向更新对应手工视图 Encoder。Sensor 不得从 `F_fused` 或 Markov 输出派生。
2. View Dropout 位于手工视图 Encoder 输出分叉后的 Markov/Fusion 主路径，只影响进入 Markov/Fusion 的手工视图；Sensor branch 使用原始 `E_v`。Deep view 永不 dropout。
3. EventSequence 必须记录 `event_source`、SensorBundle identity/hash、symbolizer rule version、segment boundaries 和源 sensor evidence；禁止事件来源隐式变化。
4. `no_llm_head` 只用于阶段 Gate 和 baseline，不是 Full LLM 模型的永久预测支路。No-LLM Gate PASS 前禁止 LLM integration；Full LLM 默认不得 ensemble 或回退到 no-LLM 输出。
5. Full LLM 显式包含两条输出路径：Qwen hidden states → VA Regression Head → `Y_va`；Qwen → LM Head → CoT/ANSWER → ANSWER Parser → `Y_answer`。`Y_va` 与有效 `Y_answer` 进入 `L_consist`。
6. PredictionBundle 必须保存两路原始输出、ANSWER 原文、parser 状态/错误、解析结果、有效 mask 和一致性证据。
7. 其余模块所有权、数据契约、Markov/Fusion 边界和 fail-fast 规则按 Foundation Design 第 3 节冻结。

### 变更控制

第 3 节状态为 `Approved / Frozen`。后续任何模块边界、梯度路径、Dropout 拓扑、事件 provenance、No-LLM 生命周期或 Full LLM 双头路径变化，必须先通过新的 `DECISIONS.md` 条目显式批准，不得仅通过配置或源码静默改变。

## DEC-0006：冻结 Run/Gate 生命周期与依赖感知有效性

- 日期：2026-08-12
- 状态：Accepted
- 决策类型：User-approved Lifecycle and Research Governance Decision
- 适用范围：所有 run、Gate Definition、Gate Evaluation Attempt、stage dependency、invalidation/supersession、formal aggregation 与 paper/report 产物

### 决定

1. Run lifecycle 与 Gate lifecycle 保持为独立状态机；run `SUCCEEDED` 不自动表示 Gate PASS。
2. `SUCCEEDED`、`FAILED`、`INTERRUPTED` 均为不可逆终态。final manifest 后整个 run 逻辑封存；续训、跨机器恢复和 post-hoc evaluation 必须创建关联的新 run。
3. formal PRE-FLIGHT 强制研究相关 Git workspace clean；dirty worktree 只能中止 formal run 或降为 debug。
4. stale RUNNING 按版本化 heartbeat/lease policy 判定，recovery finalization 必须保留完整 provenance。
5. Gate Definition 与 Gate Evaluation Attempt 分离。相同 criteria 可有多个不可变 attempts；只有 criteria/threshold/evidence/authority 改变才增加 Gate version。
6. criterion 明确 automatic/manual authority。Codex 可以计算和推荐，但不得生成或冒充 human approval。
7. Gate Record 引用 evidence commit；PASS 后 stage-close commit 提交 Gate Record；下一阶段记录 predecessor stage-close commit，避免提交自引用。
8. invalidation/supersession 使用 append-only record，不修改任何历史 run、Gate Attempt 或 commit；formal aggregation 必须排除当前无效证据。
9. `historical_verdict` 与动态 `effective_verdict` 分离。当前有效性至少包括 `VALID`、`INVALIDATED`、`STALE_DEPENDENCY`。
10. effective validity 沿显式 provenance/dependency graph 传播，并报告 invalidated source、直接受影响 attempt、下游 stages/runs、传播原因及 replacement evidence。
11. 下一阶段 entry、formal aggregation、paper/report generation 只使用 effective validity。Gate version supersession 同样通过 resolver 确定 active definition；历史 version/verdict 永久保留但不得误作当前 predecessor。

### 变更控制

Foundation Design 第 4 节状态为 `Approved / Frozen`。任何终态、恢复、Gate authority、commit chain、invalidation 或 effective-validity 语义修改，都必须通过新的 Decision 条目显式批准。

## DEC-0007：冻结 RG-03 的 MSE 收敛操作性定义

- 日期：2026-08-12
- 状态：Accepted
- 决策类型：User-approved Research Gate Criterion
- 适用范围：RG-03 Sensor Independent Validation 中的能量与亮度回归头

### 决定

1. RG-03 使用固定、无泄漏的 Sensor train/validation split；test 不参与 normalization 拟合、选模、early stopping、threshold 调整或 Gate verdict。
2. constant baseline 只能由 Sensor-train targets 拟合，并在冻结 validation split 上与模型比较。
3. 能量和亮度两个 target 的 model/baseline validation MSE 必须有限，且分别满足 `MSE_model <= MSE_train_only_constant - tolerance`。
4. `tolerance = max(1e-8, 1e-6 * abs(MSE_train_only_constant))`。该容差只处理浮点等价，不增加未经研究方案支持的效果量门槛。
5. model 与 baseline 使用相同 membership、normalization 和 metric implementation；训练按预登记 early-stopping 规则结束，并保存完整曲线与最佳 checkpoint。
6. normalization 参数只在 Sensor-train 上拟合。evidence 保存 target/pseudo-label version、transform、fit-split fingerprint、参数、schema/code version 和 metric 空间。
7. 调式强度 `CCC > 0.7` 保持研究方案原始阈值。本决定不为调性分类头凭空增加硬阈值。

### 变更控制

任何 baseline 数据范围、比较空间、容差、test 使用边界或 normalization provenance 变化，都必须创建新的 Decision 并增加对应 Gate version。

## DEC-0008：预注册 RG-04 的逐维 CCC 接近界限

- 日期：2026-08-12
- 状态：Accepted
- 决策类型：User-approved Research Gate Engineering Decision
- 适用范围：RG-04 No-LLM 主模型与冻结 CRNN reference 的 validation 比较

### 决定

1. `0.02` 是项目预注册的逐维 CCC 接近界限，不是原研究方案或文献给出的阈值。
2. 定义 `Delta_V = CCC_noLLM,V - CCC_CRNN,V`、`Delta_A = CCC_noLLM,A - CCC_CRNN,A`。
3. RG-04 PASS 当且仅当 `Delta_V >= -0.02 AND Delta_A >= -0.02`。
4. PASS 且两项 Delta 均不小于 0 时标记 `REACHED`；否则标记 `NEAR`。
5. 禁止用 V/A 平均 CCC 或单维超额表现掩盖另一维未达标。
6. CRNN 与 No-LLM 使用完全一致的 primary validation split、标签域、mask、metric implementation、选模协议和冻结 seed set；test 不参与 Gate 判定。
7. CRNN reference、threshold、seed set 和比较协议必须在 Gate 前冻结，结果观察后不得重定义或挑选 baseline。
8. 多 seed 时，Gate CCC 使用预注册逐维算术均值，同时保留全部 per-seed 数值、离散度和 run identity。

### 变更控制

任何阈值、Delta 方向、维度聚合、reference baseline、seed set 或比较协议变化都需要新 Decision 和 Gate version；历史 attempts 保留原始定义。

## DEC-0009：冻结 RG-03 调性分类头训练有效性标准

- 日期：2026-08-12
- 状态：Accepted
- 决策类型：User-approved Research Gate Engineering Decision
- 适用范围：RG-03 调性分类 Sensor Head

### 决定

1. 不凭空设定 accuracy/macro-F1 硬阈值；以冻结 validation split 上的 CE 相对 train-only class-prior baseline 判断训练有效性。
2. prior 只使用 Sensor-train 的有效 5 秒 segments，采用 Laplace/add-one smoothing：`p_c=(n_c+1)/(N+C)`，避免零频类别产生零概率。
3. model/prior validation CE 必须有限，并满足 `CE_model <= CE_train_only_prior - max(1e-8, 1e-6*abs(CE_train_only_prior))`。
4. model 与 baseline 使用相同版本化 class vocabulary/order、target encoding、5 秒 segment boundaries、valid/missing-label mask 和 CE reduction。
5. checkpoint 选择和 early stopping 只依据预登记 validation CE；test、accuracy、macro-F1 和人工结果挑选不参与选模或 Gate verdict。
6. evidence 保存 smoothing、train counts/prior、contract identity、accuracy、macro-F1、混淆矩阵和类别支持度；后四项仅作诊断。

### 变更控制

prior 数据范围、smoothing、class/mask/segment contract、CE reduction 或选模规则的任何变化，都必须创建新 Decision 并增加 RG-03 Gate version。

## DEC-0010：冻结 RG-06 Beta Learning Stability

- 日期：2026-08-12
- 状态：Accepted
- 决策类型：User-approved Research Gate Engineering Decision
- 适用范围：`beta_mel`、`beta_mfcc`、`beta_chroma` 的学习稳定性与方向性假设

### 决定

1. 每个正常完成的 validation checkpoint 保存模型实际使用的 effective beta；所有记录 finite，并保留完整 curve。
2. 至少需要训练时间顺序上 3 个连续、正常完成的 validation checkpoints；不足时 criterion result 为 `INSUFFICIENT_EVIDENCE`，Gate 不得 PASS。
3. last-3 固定为训练时间顺序上最后三个正常完成的 validation checkpoints，禁止事后挑选、围绕 best checkpoint 选择、静默删除异常点或改变窗口。
4. 对每个 beta，last-3 必须全部严格大于 0；计算 `mean_v=mean(last3)`、`std_v=population_std(last3,ddof=0)`、`CV_v=std_v/abs(mean_v)`，要求 `CV_v <= 0.10`。
5. `0.10` 是预注册 Engineering Decision，不得在观察 formal result 后修改。
6. `PASS = mel_pass AND mfcc_pass AND chroma_pass`；禁止跨视图平均掩盖失败。
7. evidence 保存完整 curves、checkpoint index/epoch/step、last-3、mean/std/CV、判定、selected checkpoint、双配置 hash、source run IDs 和 checksum。
8. `beta_parameterization` 及 effective-value 变换必须进入 provenance。结构强制 positivity 时不得把正值描述为训练发现；参数化变化需要新的 Decision。
9. `beta_chroma > beta_mel` 是 Pre-registered Directional Hypothesis，不是 PASS 条件。单独报告 `SUPPORTED|NOT_SUPPORTED`，不得为迎合预期重选 checkpoint、调参或改 Gate。

### 变更控制

记录频率、last-3 定义、positive condition、CV 公式/阈值、参数化或 Directional Hypothesis 地位的变化，都需要新 Decision 和 RG-06 Gate version。

## DEC-0011：冻结 RG-05 Markov State Collapse / Occupancy Gate

- 日期：2026-08-12
- 状态：Accepted
- 决策类型：User-approved Research Gate Engineering Decision
- 适用范围：四视图 8-state occupancy、Shared-A diagnostics 与 collapse 判定

### Research-plan-derived requirements

- 四个视图各有 8 个状态；
- 每个状态最低使用率严格 `>5%`；
- 监控状态使用率直方图与 `A` row entropy；
- 任一视图单状态占比 `>60%` 原本为报警条件。

### Engineering Decisions

1. formal Gate 基于冻结 primary validation split、有效 2Hz mask frames 和预登记 selected checkpoint；test 不参与。
2. deterministic evaluation 要求 `model.eval()`、View Dropout 关闭、四视图全部在线，mask/time-grid contract 与正式 validation 一致。
3. 状态使用率用 post-Markov `gamma_tilde` hard occupancy 操作化；argmax 平局选择最小 state index。
4. `hard_occupancy[v,n]=count_hard[v,n]/N_valid` 在整个 validation split 有效帧上直接聚合。
5. 四视图 × 8 状态共 32 cells 全部严格 `>0.05`；每视图 maximum 必须 `<=0.60`。把 `>60%` 报警升级为 formal non-PASS 是预注册 Engineering Decision。
6. `PASS = AND over all views and all required occupancy criteria`，禁止跨视图或状态平均。
7. raw `gamma` 与 post-Markov `gamma_tilde` 的 soft mass、hard occupancy、histogram 必须保存；仅 gamma_tilde hard occupancy 决定 Gate。
8. 完整 `A`、row entropy 和 `L_state` 为 required diagnostics；当前不增加 row-entropy threshold。
9. Gate 使用预登记 selected checkpoint，禁止按 occupancy post-hoc 重选。
10. RG-05 只解释单 run 内 occupancy。不同 seed 的 state index 不得在无 state-alignment protocol 时视为同一语义。

### Required provenance

evidence 保存 checkpoint identity/checksum/selection、split/TimeGrid/mask、`N_valid`、全部 occupancy/soft-mass statistics、`A`/entropy/`L_state`、tie-break version、evaluation-mode evidence、双配置 hash、source run 和 artifact checksum。

### 变更控制

occupancy probability、argmax/tie-break、聚合范围、5%/60% 判定、checkpoint policy、diagnostic/gating 地位或 state-alignment 语义变化，都需要新 Decision 和 RG-05 Gate version。

## DEC-0012：冻结 RG-07 EventSequence Token-Length Gate

- 日期：2026-08-12
- 状态：Accepted
- 决策类型：User-approved Research Gate Engineering Decision
- 适用范围：所有会被 formal LLM stage 实际注入的 EventSequence artifacts

### Research-plan-derived requirements

- Event text 必须硬控长度；
- 7B `<=180 tokens`；
- 14B/通用 `<=200 tokens`；
- 禁止逐帧 90-frame 文本膨胀 prompt。

### Engineering Decisions

1. Gate 按 `target_llm_profile + EventSequence version + event_source` 判定；同一 artifact 可 PASS_14B/FAIL_7B。
2. 使用对应 upstream manifest 的精确 tokenizer revision/checksum；revision 变化必须重评。
3. 计数为 `len(tokenizer.encode(serialized_event_block, add_special_tokens=False))`，truncation/padding/BOS/EOS 均关闭或排除。
4. canonical serialization 使用 NFC、UTF-8、LF、时间升序、固定标题/标点/空格和固定 trailing-newline 规则；禁止 trailing whitespace。
5. 计数对象是实际注入的完整 EventSequence block；保存 `event_text_sha256` 并在 Prompt assembly 核验 measured text 与 injected text 相同。
6. 全量检查 formal pipeline 所有 eligible train/validation/test records；使用 per-record maximum，不以平均/分位数/抽样替代。
7. `pseudo_label` 与 `sensor_prediction` source/version 独立判定；stage 使用多个 source 时必须全部 PASS。
8. 禁止 runtime silent truncation。超限必须 non-PASS、保留旧 artifact、显式改规则/config、version bump、生成新 artifact 并重评。

### PASS

- 7B: `max_event_token_count <=180`；
- 14B: `max_event_token_count <=200`；
- `violation_count==0`；
- tokenizer/serializer 已冻结、全量完成、hash contract 一致、无 truncation。

### 变更控制

tokenizer、block boundary、canonicalization、profile limit、population、source coverage、hash contract 或 truncation policy 变化，都需要新 Decision 和 RG-07 Gate version；历史 evidence 不得复用。

## DEC-0013：冻结 RG-08 LLM Memory Feasibility

- 日期：2026-08-12
- 状态：Accepted
- 决策类型：User-approved Research Gate Engineering Decision
- 适用范围：7B local/reference memory feasibility 与未来独立 14B scaling Gate

### Research-plan-derived requirements

- Qwen2.5-7B-Instruct、QLoRA 4-bit、gradient checkpointing；
- micro batch 4、accumulation 4、generation upper bound 300；
- RTX4090 24GB canonical reference；
- measured peak `<22GB`。

### Engineering Decisions

1. `<22GB` 操作化为严格 `measured_peak_bytes < 22*2^30`，即 `<22 GiB`。
2. 同一 Gate Definition 下 local RTX5060Ti 16GB 与 canonical RTX4090 profile 使用独立 Evaluation Attempts。Local PASS=`LOCAL_FEASIBLE`，不得替代 reference PASS；无 4090 时 reference=`BLOCKED_REFERENCE_HARDWARE`。
3. reference probe 不得缩小冻结 runtime；local 只允许修改 allowlist runtime fields，语义变化必须新 identity/Decision。
4. probe 包含 P1 full load、P2 Stage1 full optimizer step、P3 Stage2 两个完整 accumulation cycles（含 optimizer-state init/steady state）和 P4 正式 autoregressive generation。
5. 使用最大正式 sequence/input 与完整实际 training topology；artifact 最大长度变化必须重评。
6. 同时记录 PyTorch allocated/reserved 与 NVML process/device memory；`measured_peak=max(torch_reserved,nvml_process_peak)`。缺必要 source=`INSUFFICIENT_EVIDENCE`。
7. sub-probes 优先 fresh process；共用时记录 reset/cache/residency/boundary，禁止清 cache 造低峰值。
8. actual runtime state 必须验证 checkpointing、4-bit、LoRA、optimizer/state dtype、precision、backend、use_cache、batch/accumulation/sequence。
9. Local 完整无 OOM/fallback 且 evidence 完整才 `LOCAL_FEASIBLE`；失败为 `BLOCKED_LOCAL_COMPUTE`，保存证据并生成 portable handoff。
10. 外部恢复创建新 run，保留适用 semantic identity 与 continuation provenance，不重开终态 run。
11. RTX4090 reference 的 P1–P4 每项 peak 均 `<22 GiB`；`RG08_peak=max(required_subprobe_peaks)`。
12. 14B 必须独立 hardware/config/sequence/probe/Gate；7B PASS 不推断 14B。

### 变更控制

单位、hardware profile、reference runtime、sub-probes、sequence envelope、measurement source/peak 定义、Local/reference verdict 或 14B independence 的变化，需要新 Decision 和 RG-08 Gate version。

## DEC-0014：冻结 RG-09 CoT Synthesis Quality Gate

- 日期：2026-08-12
- 状态：Accepted
- 决策类型：User-approved Research Gate Engineering Decision
- 适用范围：7B/14B finalized synthetic CoT training artifacts

### Research-plan-derived requirements

- automatic filtering 后人工抽检 100 条；
- qualified `>=85/100`；
- 格式/ANSWER-CCC 自动过滤；
- 7B/14B CoT 配置与 synthetic:real ratio 不同。

### Engineering Decisions

1. Gate 只评 finalized `synthetic_cot_records`，不从 mixed synthetic+real artifact 抽样；test 排除。
2. 按 `target_profile + synthetic artifact version` 独立 Gate，7B/14B 不混合。
3. population<100=`INSUFFICIENT_EVIDENCE`；否则按 cot_record_id 无放回抽 exactly 100 unique records。
4. sampling frame/hash、algorithm、seed、strata、IDs/order、artifact/profile 在 human 查看前冻结；结果差不得换样。
5. 两名真实 human reviewers 独立首轮且互不可见判断；Codex/LLM 不得充当 reviewer 或 adjudicator。
6. 冻结 C1 Format、C2 Event Citation、C3 VA/ANSWER Consistency、C4 No Fabricated Evidence、C5 Usability binary rubric；`qualified=C1 AND C2 AND C3 AND C4 AND C5`。
7. failure 保存 structured reason codes；分歧由真实 human adjudicator 处理，A/B 原始判断 append-only 保留。
8. `qualified_count>=85` 才 PASS；84 FAIL，不四舍五入，不替换失败样本或修改分母。
9. reviewer identity/blinding/order/session 保存；agreement/kappa/criterion agreement 和可选 CI 仅诊断，不改变 Gate。
10. human 未完成=`PENDING_MANUAL_REVIEW`；sampling/provenance invalidation 服从 effective validity。

### 变更控制

population、profile、sampling、rubric、reviewer/adjudication authority、qualified AND logic、85/100 threshold、replacement 或 agreement gating 地位变化，需要新 Decision 和 RG-09 Gate version。

## DEC-0015：冻结 RG-10 ANSWER → VA Parser Stability Gate

- 日期：2026-08-12
- 状态：Accepted
- 决策类型：User-approved Research Gate Engineering Decision
- 适用范围：Qwen2.5-7B/14B formal primary-validation generation artifacts

### Research-plan-derived requirements

- ANSWER parser pipeline 必须测试；
- parsing success rate `>=98%`。

### Engineering Decisions

1. Gate identity 至少为 `target_llm_profile + answer_schema_version + parser_version + generation_protocol_version/hash`；7B/14B 独立，attempt 绑定实际 run/checkpoint/config/prompt/tokenizer identity。
2. 使用冻结 primary validation full population，不抽样，test 排除，不得 cherry-pick subset。
3. infrastructure/execution failure 导致 generation population 不完整时不塞入 denominator，attempt 为 incomplete/BLOCKED/non-PASS；completed generation 的 empty/malformed/truncated/invalid ANSWER 进入 denominator 并计 failure。
4. 冻结 successful parse S1–S8：唯一结构、contract-derived count、V/A 齐全、确定 numeric grammar、finite、既有 VA range、结构完整、mask alignment；禁止 silent trim/pad/clipping/repair。
5. 标准 45s/2Hz 通常是 90 pairs，但 expected count 从 Sample/TimeGrid/mask contract 推导，不得隐藏硬编码。
6. 重复合法 VA 数值不构成错误；只有重复结构项或重复 explicit time position 才 FAIL。
7. `parser_normalization_version` 只允许预注册 deterministic representation normalization；禁止 semantic repair、GT reconstruction 或 LLM repair。
8. completed output parse failure 后禁止 retry、换 seed/decoding、局部重跑、人工修复、replacement、deletion 或 denominator change。
9. 精确判定 `parser_success_rate=N_success/N_eligible` 且 `100*N_success >= 98*N_eligible`；禁止显示值四舍五入判 PASS。
10. generation protocol 必须冻结；raw generation text 在 parsing 前 immutable，保存 SHA-256 与完整 generation provenance。
11. 每个 failure 保存 raw output 与 structured multi-reason codes；presence/schema/numeric/time-grid/range diagnostics 为 non-gating。
12. parser/generation implementation bug 必须 invalidation 原 attempt，按需 bump version 并创建新 generation/evidence/Gate attempt，禁止覆盖历史。
13. RG-10 不隐式定义 parse failure 在训练中如何进入 `L_consist`；留待 Loss/Training Protocol 冻结。

### 变更控制

阈值、population/split、profile independence、S1–S8、infrastructure/parser failure boundary、normalization/repair、integer comparison、retry/replacement、raw immutability 或 training-loss scope 的改变，需要新 Decision 和 RG-10 Gate version。

## DEC-0016：冻结 RG-01 MERT Actual Frame-Rate & Time-Axis Contract

- 日期：2026-08-12
- 状态：Accepted
- 决策类型：User-approved Research Gate Engineering Decision
- 适用范围：MERT deep-view preprocessing、feature extraction、TimesNet/2Hz alignment 与 feature cache

### Research-plan-derived requirements

- MERT input `24kHz`；标准 DEAM `45s`、`1,080,000 samples`；
- MERT research layer 5/6、hidden dimension design `768`；
- 必须先真实 forward 验证 frame rate/shape，再反推 TimesNet 与下游 dimensions；
- downstream primary TimeGrid 为 2Hz/90 frames。

### Engineering Decisions

1. Gate identity 绑定 pinned upstream model/weights、processor、resampler、layer-selection 与 MERT time-axis contract versions。
2. layer 5/6 需冻结 hidden-state collection semantics、numbering convention、actual indices 与 block identity；indices 必须实测，不凭记忆填写。
3. 分离 production preprocessing 与 model temporal contracts；45s preprocessing 必须确定地产生 `1,080,000@24kHz`，无隐式 decoder/MERT length correction。
4. 至少 P1 exact deterministic waveform 与 P2 registered real DEAM probe 完成真实 forward；config/README/mock 不可替代。后续 cache 仍需逐记录 integrity validation。
5. Layer 5/6 T/D 一致、D=768、outputs finite；不匹配不得先 projection 再 PASS。
6. `observed_frame_count/45s` 仅 diagnostic；真实 timestamps 从 pinned feature-extractor stride/receptive-field/padding geometry 推导，expected count 必须与 observed forward 完全相等。
7. frame timestamps/assignment 优先使用 integer sample/rational arithmetic，不依赖 float boundary coincidence。
8. 2Hz target 使用 90 个 0.5s half-open bins；每 frame exactly one bin，assigned total等于 observed、unassigned=0、duplicate=0，全部 90 bins non-empty。
9. 禁止 hidden trim/pad/drop/clamp、复制/fill/interpolation 或 adaptive pooling；每 bin 使用冻结 mean over assigned valid frames，T=90 由 contract 推导。
10. invalid source/frame/bin 由显式 TimeGrid/Mask 表达；不得后处理静默制造有效 mask。
11. 重复/跨机器要求 structural/time-axis identity；不要求不同 backend hidden activations bitwise identical。
12. downstream raw T、TimesNet periods、reduction、T=90、masks/cache shapes 引用 `MERTTimeAxisContract`；禁止硬编码 50Hz/2250。
13. upstream/processor/resampler/crop-pad/layer/geometry/binning 变化必须 new version、invalidation/supersession、new RG-01 attempt 与受影响 cache 重验。

### 变更控制

input rate/length、layer/dimension identity、probe set、temporal geometry、timestamp/binning/aggregation、exactly-one/non-empty约束、mask、determinism 或 version propagation 的改变，需要新 Decision 和 RG-01 Gate version。

## DEC-0017：冻结 RG-02 Handcrafted Feature Binning & Cross-View Alignment Gate

- 日期：2026-08-12
- 状态：Accepted
- 决策类型：User-approved Research Gate Engineering Decision
- 适用范围：Mel/MFCC/Chroma feature definitions、2Hz cache 与 four-view alignment evidence

### Research-plan-derived requirements

- Mel：22,050Hz、`n_fft=2048`、`hop=512`、128、power 2.0→dB、`[90,128]`；
- MFCC：共享 Mel STFT timing、40、`[90,40]`；
- Chroma CQT：`hop=512`、12、`bins_per_octave=36`、逐 raw-frame L1、`[90,12]`；
- window-center assignment、within-window mean、0.5s/2Hz/90 bins；
- 可视化 3 首歌检查 waveform 与四视图时间对齐。

### Engineering Decisions

1. RG-02 依赖当前有效 RG-01 PASS；上游失效通过 effective-validity dependency graph 传播。
2. Gate identity 绑定 decoder、22.05k resampler、extractors、timestamp/binning/TimeGrid versions；center/pad/CQT edge semantics 与 library defaults 显式冻结。
3. Mel/MFCC/Chroma 消费同一 22.05k waveform，并与 Deep 共享 stable IDs、segment、TimeGrid 与 mask；mismatch fail fast。
4. raw timestamps 使用 sample-domain integer/rational arithmetic；每 frame window-center exactly-one-bin，每 bin mean。
5. 所有 formal eligible train/validation/test records 自动检查；这不读取 target、不选模，因此不构成 test leakage。
6. 每 `record×view` assigned=raw、unassigned=0、duplicate=0、bin-sum=raw，全部 90 bins non-empty；禁止 trim/pad/clip/copy/fill/interpolation/adaptive pooling/reshape。
7. finalized shapes 必须为 Mel `[90,128]`、MFCC `[90,40]`、Chroma `[90,12]`，且 schema/dtype/finite/TimeGrid/mask 全合法。
8. Chroma 顺序为 per-raw-frame L1 后 bin mean；zero/near-zero rule、epsilon（如有）与 tolerance 在 formal attempt 前预注册版本化，不伪称原研究阈值。
9. 四视图只要求同一 `t` 对应同一真实时间窗；不要求数值、相关性或峰值一致。
10. 人工样本为查看前冻结的 3 unique train songs，确定性无放回；FAIL 后不得重抽或用 validation/test 替换。
11. visualization summaries 在查看前版本化，不能为制造曲线一致选择 projection 或使用 GT。
12. 真实 human reviewer 使用 M1 identity、M2 coverage、M3 bins、M4 applicable events、M5 no offset；三个 samples 全 AND。Codex 不得生成 human approval。
13. `RG02_PASS=AUTOMATIC_ALL_RECORDS_PASS AND MANUAL_3_SONGS_PASS`，任一侧不能替代另一侧。
14. 修复 extractor/preprocessing/binning 必须 bump version、生成新 immutable cache/new attempt，保留旧 evidence。

### 变更控制

predecessor、feature dimensions/parameters、timestamp/binning/accounting、Chroma normalization、automatic population、manual sampling/rubric/authority、cross-view meaning、PASS AND logic 或 cache/version propagation 的改变，需要新 Decision 和 RG-02 Gate version。

## DEC-0018：冻结 Section 6 Primary Experiment Matrix 语义闭包

- 日期：2026-08-12
- 状态：Accepted
- 决策类型：User-approved Research / Experiment Catalog Decision
- 适用范围：Section 6、A1–A13、formal primary ablation aggregation

### 决定

1. 接受 Section 6 Closure Audit 的 `CONDITIONAL PASS`；A1–A13 semantic coverage、ID uniqueness、base consistency、factor uniqueness、research-plan coverage、special conflict audit 与 primary matrix closed world 均为 PASS。
2. Section 6 semantic design 标记为 `APPROVED / FROZEN`，`semantic_closure=COMPLETE`。
3. 同时保持 `machine_readable_normalization=PENDING`、`executable_binding=PENDING_DEPENDENCIES`、`formal_ready=false`，三者不得与 semantic closure 混淆。
4. 后续未决 OQ 属于 executable-binding dependencies，不能据此重新打开 A1–A13 scientific definitions。发生真实矛盾时必须走 append-only Decision 与 version-change review。
5. A1–A13 是当前 `PRIMARY_EXPERIMENT_MATRIX_CLOSED_WORLD`；未经批准的新 A-number/A-variant 不能进入 formal paper aggregation。Debug/exploratory run 必须标记 debug 且不得聚合。

### 变更控制

变更 variant identity/version、base、primary factor、dependency closure、analysis anchor 或 closed-world 边界，需要新 Decision；已被 formal evidence 使用的 identity 不得覆盖修改。

## DEC-0019：冻结 15 个 executable identities、A8 grouping 与双轴 scope

- 日期：2026-08-12
- 状态：Accepted
- 决策类型：User-approved Experiment Identity Engineering Decision
- 适用范围：Experiment Catalog schema、S5 evidence、run PRE-FLIGHT

### 决定

1. 冻结 15 个 executable identities：`A1-v1`、`A2-v1`、`A3-v1`、`A4-v1`、`A5-v1`、`A6-v1`、`A7-v1`、`A8-MEL-v1`、`A8-MFCC-v1`、`A8-CHROMA-v1`、`A9-v1`、`A10-v1`、`A11-v1`、`A12-v1`、`A13-v1`，其完整 variant IDs 以 `EXPERIMENT_MATRIX.md` 为准。
2. `A8_SINGLE_HANDCRAFTED_VIEW` 是 `NON_EXECUTABLE_GROUP`：不产生 run、seed、checkpoint 或 paper row。三个 children 各自完成有效 `COMPLETE_S5` 后才可声明 A8 family complete。
3. Catalog 必须分别表达 `scope_tier` 与 `execution_obligation`。前者至少覆盖 PRIMARY/SECONDARY/OPTIONAL/CONTINGENCY；后者至少表达 REQUIRED/CONDITIONAL/NOT_TRIGGERED/OPTIONAL_NOT_COMMITTED/BLOCKED 或语义等价状态。
4. A12 是 PRIMARY、CONDITIONAL；trigger definition 未解析时等待 trigger，预注册 trigger 不满足时转为 NOT_TRIGGERED，但不删除 catalog entry。
5. scope/obligation 只能由冻结 Scope Hierarchy、Trigger Protocol 与 Resource Policy 解析，禁止根据实验结果调整。

### 变更控制

把 A8 parent 变为 executable、遗漏任一 A8 child、合并 scope 与 obligation、或以结果驱动改变 obligation，均需新 Decision 和 catalog version review。

## DEC-0020：冻结 A1–A13 科学因素与关键非等价边界

- 日期：2026-08-12
- 状态：Accepted
- 决策类型：User-approved Research Semantics Decision
- 适用范围：A1–A13 implementation binding、Metrics、paper interpretation

### 决定

1. A2 与 A8 不重复：A2 是 complete handcrafted subsystem removal；A8 是一个完整 handcrafted subsystem 在 Deep-only 之上的 conditional contribution。
2. A3 与 A11 不重复：A3 改变 transition-matrix sharing；A11 增加 explicit posterior consensus regularization。
3. A4 与 A13 不重复：A4 在完整 A1 内移除 predictive state bias；A13 是 independent feature-concat baseline。
4. A5 与 A10 不重复：A5 移除 prompt-side Event injection；A10 移除 generative CoT/ANSWER subsystem。
5. A6 与 A13 不重复：A6 是 full A1 mechanism 加 controlled A1 Event Replay；A13 不含 Event、Markov 或 LLM。
6. Event 三分语义冻结：A2 无 handcrafted/Sensor/Event 且使用 event-free target；A5 保留 Sensor/Event artifacts、Event 不可见并保留 A1 event-derived targets；A6 无 local Sensor、使用 paired A1 Event Replay 并保留 A1 targets。
7. Markov 四分语义冻结：A3 shared A→per-view A；A4 只移除 predictive state bias；A11 增加 KL posterior consensus；A12 增加 HSIC representation decorrelation。
8. A10 是 Qwen-based VA-only LLM；No-LLM 是 `F_fused→no_llm_head` 且没有 ChatTS/Qwen/LoRA。二者不得复用 identity、checkpoint、RG-04 verdict 或 aggregation row。
9. A13 的 base 为 NONE、comparison anchor 为 A1，只通过 component contract refs 共享研究接口；永久标记 `KANG_HERREMANS_INSPIRED` 与 `NOT_EXACT_REPRODUCTION`。

### 变更控制

任何合并上述实验、改变事件来源/可见性、扩大或缩小 Markov intervention、混淆 A10/No-LLM、或把 A13 称为精确复现的行为，均构成 frozen research semantics 变更，必须新 Decision。

## DEC-0021：冻结 S5 全局证据规则与 Blocking Dependency Register 治理

- 日期：2026-08-12
- 状态：Accepted
- 决策类型：User-approved Reproducibility / Dependency Governance Decision
- 适用范围：所有 trainable formal variants、Experiment Catalog、formal readiness

### 决定

1. 所有 trainable formal variants 最终使用同一冻结 S5，并要求 `COMPLETE_S5`；A8 children 分别满足，A12 若触发也满足。staged execution 不改变最终证据要求。
2. 本决定当时将具体 seed integers/order/version/RNG domains 保留给 `OQ-SEEDSET-S5-VALUES`；该历史待决项后由 DEC-0023 的 `SS-3+`/`RNG-3+` 解决，不再是 OPEN。
3. 建立 `docs/research/EXPERIMENT_MATRIX.md` 作为 human-readable Experiment Catalog Source of Truth，并建立统一 Blocking Dependency Register。
4. 每个 dependency 至少记录 ID、owner section、affected variants、blocking stage、description、status 与 resolution reference。未解析依赖必须有 required-before 边界，不得散落为无 owner 的自然语言。
5. A10–A12 等自然语言不变量足够完成 semantic freeze，但 formal PRE-FLIGHT validator 前必须显式展开 machine-readable `invariant_fields`，禁止只写 `everything_else_same_as_A1`。
6. 本决定当时未授权 PMEmo、14B、14B CoT-DPO、Qwen2.5-Audio 或 static MER contingency formal run；其中 PMEmo 后由 DEC-0024 的 `PME-ZST19-A1S5-3R++` 授权为 A1-S5 required zero-shot evaluation，其他项仍 `DEFERRED_NON_BLOCKING`。

### 变更控制

S5 evidence completeness、seed-set identity、dependency owner/required-before、machine-readable invariant expansion 或 formal authorization boundary 的改变，需要新 Decision；依赖 resolution 可以在其获批 owner section 中追加 `resolution_ref`，不得覆盖历史记录。

## DEC-0022：冻结 Foundation Batch 1 Training Execution Boundary

- 日期：2026-08-12
- 状态：Accepted / Frozen
- 决策类型：User-approved Foundation Engineering Decision
- 当前 authority：`TEB-FBC-3R+++`
- 包含：`VCP-CCEW-3R++`、`A10-RG08-ASAND-FVA-DP-3R+++`

### 决定

1. 冻结 versioned Stage Contract Graph、`EventSourceCurriculum + CoT Artifact Boundary (Scheme B+)`、`FI-A+`、`PB-1+`、`CS-3+`、`H3+`、`SPP-3R+`、`MMP-GT-3R+` 与 `LSAM-3R+`。Stage 1→2 新建 run，H3+ transfer，optimizer rebuild、scheduler reinitialize、新 Stage-2 RNG namespace；Stage 2 epochs 1–3 early/4–5 late，epoch 3 commit 后切换，且不重置 Stage-2 状态。
2. Early 用 pseudo Event，late 用 frozen Stage-1 Sensor-snapshot Event；live Stage-2 Sensor 继续训练。Final A1-like inference 使用 terminal Stage-2 own Sensor；A5 local shadow/model-invisible，A6 paired same-S5 A1 replay/no local Sensor，A10 local model-visible Event-conditioned VA，A2/A13 N/A。
3. `VCP-CCEW-3R++` 编译 module existence、event visibility/source、loss、prompt/target、Gate、stage、parameter/mode/gradient 和 evidence；formal runtime/CLI 不得改变其研究语义。
4. A10 保留 Qwen hidden backbone、local model-visible Event 与 VA head，但没有 LM generation/LM head output、CoT/ANSWER、Parser、`Y_answer`、`L_CoT` 或 `L_consist`；RG-08 applicable roles 为 P1/P2 Stage 1、P1/P3 Stage 2、FVA final，P4 N/A。PASS 使用 applicable-role max `<22*2^30`，本机证据不替代 canonical RTX4090。
5. terminal run/checkpoint 不回写；跨机器 continuation 创建新 run 并保留 typed handoff、phase、RNG 与 artifact provenance。

### 分类与变更控制

上述 stage/capability/execution binding 为 `PRE_REGISTERED_ENGINEERING_DECISION`。改变 final event producer、phase boundary、checkpoint selection、handoff schema、capability closure 或 A10 topology/applicability 必须新 Decision/version；loader、projection、hardware 与机器绑定只影响 executable qualification，不重开语义。

## DEC-0023：冻结 Foundation Batch 2 Loss / Optimization / Reproducibility

- 日期：2026-08-12
- 状态：Accepted / Frozen
- 决策类型：User-approved Foundation Engineering Decision
- 当前 authority：`LOR-FBC-3R+++`
- 包含：`COTC-SN21-DSDP-3R+`、`A13-NLBTP-VAS-3R+`

### 决定

1. 冻结 `DLC-P1-MESM-3R+`、`DLC-P2-TFES-3R+`、`DLC-P3-ARSA-3R+`、`DLC-P4-NEHF-3R+`、修订版 `DLC-P5-FHST-3R+`、`DLC-P6-DAAI-3R+`、`DLC-P7-KLV-CN-3R+`、`HLAC-HDSA-3R+`，以及 Stage-1/Stage-2 的精确 objective/active-set；PB boundary 不改变 active set、weights、uncertainty registry 或 optimizer membership。
2. active loss 必须由 capability/stage 编译并保存 numerator/denominator/eligibility；零 eligibility 不产生伪 numeric loss，也不更新 Kendall task uncertainty。
3. 冻结 `COTC-SN21-DSDP-3R+` 为 Synthetic-CoT / Native-DEAM 2:1 Deterministic Song Partition：`N_syn=floor(2N/3+0.5)`，SHA-256 song ranking，S5-independent、phase-invariant。Native DEAM 为合法 VA-only coverage、`L_CoT` zero-by-manifest，但可按能力参与 `L_consist`；不得称为 real/human/ground-truth CoT。Human CoT 与 difficulty weighting deferred non-blocking。
4. 冻结 A13 independent No-LLM five-epoch baseline：`L_A13=L_VA+0.05L_smooth`，固定权重/no Kendall，OSNP/OSNS Stage-1 numeric profile、TSC、epoch-5 terminal only；A13 不是 A1 reuse 或精确外部复现。
5. 冻结 `TSC-GPESA-3R+`、`RNG-3+`、`SS-3+`、`CRR-HRSC-3R++`、`OSNP-SLAW-3R++`、`OSNS-DB10-3R+`。S5 identity 为 `SC-MV-DMER-FORMAL-S5/v1`，顺序为 `[52826381, 128866372, 1616929435, 1871035633, 1460830465]`。
6. 冻结 `LQ7-R16A32-ZD-NF4-3R+` 与 `QKB-SPMS-3R++`。completed parser failure 为 zero consistency eligibility + retained evidence；infrastructure failure 为 FAIL/INTERRUPT。全局 zero eligibility 时相关 conditional parameter 必须 `grad=None`，禁止用 zero tensor 让 AdamW momentum/decay 改参。

### 分类与变更控制

研究方案明确的 loss family、2:1 mixture 与 7B QLoRA intent 为 `RESEARCH_PLAN_DERIVED`；精确 reduction、eligibility、RNG、replay、numeric 和 loading 规则为 `PRE_REGISTERED_ENGINEERING_DECISION/INTERPRETATION`。任何语义修改需新 Decision；CRR、loader、gradient 与 numerical runtime 验证仍是执行 blocker。

## DEC-0024：冻结 Foundation Batch 3 Metrics / Evaluation / Global Stage

- 日期：2026-08-12
- 状态：Accepted / Frozen
- 决策类型：User-approved Foundation Research/Engineering Decision
- 当前 authority：`MEGPC-FBC-3R++++`
- supersedes：`MEGPC-FBC-3R+++`

### 决定

1. 冻结 `MEP-CCC2-S5-3R+`：CCC 使用 FP64 pooled population moments，MSE 使用 pooled valid-frame squared-error；V/A 分开；per-seed、S5 arithmetic mean、sample std (`ddof=1`) 与 `COMPLETE_S5` 必须保留；禁止 best-seed。
2. 冻结 `GEN-DTB-VA90-3R++`。研究方案中的 300-token statement 被 `PRE_REGISTERED_ENGINEERING_CORRECTION` 解释为 `THINK explanatory payload <=300 pinned-tokenizer tokens`，不是 total generation budget。`T=canonical TimeGrid length`，标准 DEAM `T=90`；每个 V/A 各 exact T 个 4-decimal signed values。`max_new_tokens=B_think+B_answer+B_wrapper`，其中后两者必须 tokenizer-qualified；失败则 formal execution blocked。
3. 冻结 `EvaluationInputCoverageContract`：`InputCoverageMask` 与 `TargetValidityMask` 分离；padded/uncovered 位置的 model input、target、Sensor/Event validity 均为 false；padding 不是真实 silence。
4. 冻结 `ECA-ATOM-PREC-3R++`：machine metric 为 `EventCitationPrecision`；EventAtom stable ID；formal syntax exact `<CITE id="E03"/>`；exact registry lookup，禁止 free-form NLP denominator。无 recognized citation 为 `UNDEFINED_NO_CLAIMS`。A5 shadow、A6 paired replay、A2/A10/A13 N/A。
5. 冻结 `A12-HGST-Z0-3R++`：paired exact S5，`G_d=mean_s(CCC_A1,s,d-CCC_A2,s,d)`，`tau=1e-12`；both `>tau` 才充分，任一 `<=tau` 触发；missing/invalid/stale 为 BLOCKED/STALE，不解释为 trigger outcome。
6. 冻结 `DEAM-L58-SW45-3R++` 与 `PME-ZST19-A1S5-3R++`。前者 A1-only/no-retrain、45s/22.5s、coverage+target masks；后者采用 `DISTRIBUTED_CHORUS_CLIP_LOCAL_TIME`、A1 S5 zero-shot、manifest/domain audit，禁止把 full-song audio 当必需输入。
7. 冻结 `FPP-VAH-MIN-3R+` 的 projector/No-LLM/Qwen7B VA-head 精确结构与 `dropout=0`；`tanh` 是 forward activation，不是 post-hoc clipping。
8. 冻结 `SGA-M0M20-EV-3R+` 与 `PER-EVAND-S5-3R++`；stage entry/paper aggregation 使用 effective validity 与 exact S5。14B、DPO、Qwen2.5-Audio、optional/contingency 均 `DEFERRED_NON_BLOCKING`，不阻塞 7B primary Foundation Closure。

### 研究方案偏差与 claim firewall

1. `FINAL_GENERATION_HUMAN_EXPLANATION_EVAL` 与 `MARKOV_HUMAN_STRUCTURE_ALIGNMENT` 记录为 `DEFERRED_NON_BLOCKING_SOURCE_PLAN_DEVIATION`；RG-09 只证明 synthetic CoT artifact quality，不是 final model explanation human evaluation。
2. 没有人类结构对齐 evidence 时，不得宣称 shared A learned true musical structure 或与 human chorus boundaries 对齐；可按实际 diagnostics 描述 non-degenerate dynamics、occupancy 和 interpretable transition matrices。
3. 没有 final-generation human evidence 时，不得宣称 final explanations human-validated。

### 变更控制

`MEGPC-FBC-3R+++`、`ECA-ATOM-PREC-3R+`、`A12-HGST-Z0-3R+`、`DEAM-L58-SW45-3R+`、`PME-ZST19-A1S5-3R+`、`GEN-DTB-VA90-3R+` 仅可作为 `SUPERSEDED` 历史引用。改变 metric aggregation、budget/ANSWER identity、citation syntax、trigger、secondary scope/timeline、head architecture、stage applicability 或 claim firewall 必须新 Decision/version。

## DEC-0025：记录 Foundation Closure 审计状态与人类批准 provenance 边界

- 日期：2026-08-12
- 状态：Accepted
- 决策类型：Closure Governance Decision

### 决定

1. Batch 1/2 已由用户在先前对话批准，修订后的 Batch 3 已由用户在本轮批准；`HUMAN_APPROVAL_SOURCE=external conversational approval pending local provenance transcription`。
2. 三批文档持久化和纯文档 self-review 未发现 unresolved semantic contradiction，因此 `FOUNDATION_RESEARCH_SEMANTICS=APPROVED/FROZEN`、`FOUNDATION_DESIGN_VERDICT=CONDITIONAL_PASS_EXECUTION_BLOCKED`、`FORMAL_EXECUTION_READY=NO`。
3. 由于没有单独的 machine-verifiable human approval artifact/transcript checksum，不伪造其存在；M0 标为 `READY_TO_CLOSE` 而不是 PASS。
4. 余项只可作为 data、machine、executable 或 provenance blocker；不得把已冻结设计 OQ 继续标为 OPEN，也不得以执行 blocker 重新开启已冻结语义。

## DEC-0026：Foundation Authority Synchronization — Four Persistence Corrections

- 日期：2026-08-12
- 状态：Accepted / Frozen Synchronization
- 决策类型：`NO_NEW_SCIENTIFIC_DECISION` / Documentation and Authority Synchronization
- 适用范围：DLC-P5、CRR、RG-10、RG-08 P1 的 current documentation authority

### 同步决定

1. `DLC-P5-FHST-3R+` 的唯一 ownership 是 Flat-Head Sum Sensor Supervision：`L_sensor=L_rms+L_brightness+L_mode+L_key`，Key 为 12-way tonic classification，Mode 为 continuous `[-1,1]` regression，`lambda_sensor=0.1`。它不拥有 `L_smooth`、`L_HSIC` 或 A11/A12 scientific semantics；前者属于 DLC-P1/A13，后者属于 A12+DLC-P6/HLAC。
2. `CRR-HRSC-3R++` 的 `CheckpointReplayCapsule` 是 per-checkpoint-invocation、step-local stochastic replay object，只治理 activation-checkpoint original/recompute 的 RNG entry/exit、shadow replay、call trace 和 capsule identity。durable model/optimizer/scheduler/sampler/logical-epoch/PB/RNG/GradScaler/artifact state 属于整体 TrainingState checkpoint。mid-step failure 从最后 durable LogicalOptimizationStep boundary 重放，不恢复 partial capsule。
3. Batch-3 `GEN-DTB-VA90-3R++` supersedes DEC-0015 中 mask-dependent RG-10 expected-length interpretation。Current Gate Definition 提升为 `RG-10-v2`：`T=canonical output TimeGrid length`，且 `parsed_V_count==T AND parsed_A_count==T`；TargetValidityMask 不缩短 ANSWER，只治理 applicable target-based loss/metric eligibility。DEC-0015 原文作为 historical `RG-10-v1` provenance 保留；本决定不创建、不声称任何 Gate Evaluation Attempt。
4. RG-08 P1 的 global semantic 仍为 Load/Resident Topology Probe，但每个 P1 必须由 `VariantCapabilityProfile + TrainingStageDefinition + Context Profile` 编译 exact topology。canonical A1-like P1[STAGE1] 的 LoRA 为 `NOT_APPLICABLE_BY_STAGE`；P1[STAGE2] 包含适用 Stage-2 LoRA。A10 继续 no lm_head/no generation，并使用自己的 P1/FVA profile。

### 变更边界

本决定只修复文档 persistence/authority synchronization defects，不重新打开 Foundation scientific semantics，不改变 RG-10 的 `>=98%` threshold、population、repair/retry policy 或证据有效性规则。未来修改上述 current authority 必须走正常 append-only Decision/Gate version/effective-validity 流程。

## DEC-0027：首次冻结 DEAM Primary Song Split

- 日期：2026-08-25
- 状态：Accepted / Frozen
- 决策类型：User-authorized Data Governance Decision
- 适用范围：`E2_DEAM_PRIMARY_SPLIT_READY`、DEAM primary 及其后续正式 baseline、主模型、消融和 multi-seed runs

### 决定

1. 用户授权首次冻结 `primary-song-80-10-10-v1`，输入必须是已登记的 `DEAM-PRIMARY-v1` dataset manifest，且 dataset manifest internal hash 与 file checksum 原样绑定。
2. split contract 固定为 `DEAM-PRIMARY-SONG-SPLIT/v1`。对每条 1,744 条 PRIMARY record 生成 canonical JSON：`dataset_id`、`song_id`、`split_contract_id`；使用固定 key 排序、UTF-8、compact JSON 和 SHA-256。
3. 以 `(rank_digest ASC, song_id ASC)` 排序，`rank_index` 从 0 开始。固定配额为 train `floor(0.8*1744)=1395`、validation `floor(0.1*1744)=174`、test `175`。
4. ranking 和 membership 不得读取 annotation、target、model result、seed、路径或文件名；不得使用随机数、sklearn/random split 或任何结果驱动选择。
5. manifest 必须保存每条 `song_id`、`sample_id`、`logical_song_key`、split、rank digest/index、canonicalization rule、dataset manifest provenance 和自身 checksum。三 split 必须歌曲互斥并完整覆盖 1,744 条 PRIMARY。
6. 这是唯一首份冻结 split。后续执行只能验证 registered manifest/hash；任何成员或配额变更必须新建可追踪 Decision/version，不得覆盖或生成第二份 primary split。

### 变更控制

本决定只冻结用户明确授权的 deterministic membership procedure，不改变数据内容、annotation semantics、RG-01 或后续特征/训练研究语义。任何 split identity、contract、quota 或 ranking rule 修改必须通过新的 append-only Decision 并使依赖它的 readiness/run evidence 失效。
