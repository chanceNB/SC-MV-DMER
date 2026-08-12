# SC-MV-DMER Research Specification

> 状态：Foundation 三批研究语义已获用户批准并冻结；当前 Closure Audit 为 `CONDITIONAL_PASS_EXECUTION_BLOCKED`。`FORMAL_EXECUTION_READY=NO`；M0 为 `READY_TO_CLOSE`，等待可机器核验的人类批准 provenance transcription。本文档是当前完整的人类可读研究语义 Source of Truth。

## 1. Source of Truth

`research_plan.docx` 是原始研究设计基准。经用户明确批准并记录在 `DECISIONS.md` 中的后续决定，是对未定义工程细节的受控冻结；不得用未记录的会话记忆替代仓库文档。

## 2. Frozen Data-Split Policy

### 2.1 Primary split

- 主实验采用唯一一份歌曲级 `80/10/10` train/validation/test split manifest。
- primary split manifest 一经正式冻结，所有正式 baseline、主模型、A1-A13 消融和 multi-seed 实验必须复用其成员关系。
- 正式实验开始后，不得根据实验结果重新生成或筛选 primary split。

### 2.2 Five-seed semantics

- 5 个随机种子只控制模型初始化、训练采样、shuffle、Dropout 及同类训练随机性。
- 随机种子不得改变 train、validation 或 test 的样本成员。
- 5-seed 均值与标准差只表示固定数据划分下的训练随机性，不表示数据划分敏感性。

### 2.3 Split-robustness boundary

- 数据划分敏感性必须通过独立命名、独立配置、独立产物目录和独立报告的 `split-robustness` 实验评估。
- `split-robustness` 结果不得并入、替代或混淆 primary split 上的正式 5-seed 统计。

### 2.4 Integrity rule

若发现数据损坏、重复、样本身份错误或歌曲级泄漏，必须按照 `DECISIONS.md` 的变更控制流程处理。任何 primary split 变更都必须可追溯，并明确标记受影响且需要作废或重跑的实验。

## 3. Frozen Research-Engineering Governance

### 3.1 Repository and data-root boundary

- 项目采用 Manifest-first 分阶段单仓库。
- Git 仓库与外部数据根目录分离；本机默认的大文件根目录为 `D:\SC-MV-DMER-data`。
- 源码不得硬编码绝对数据路径。路径必须由可配置的 data root 和受控的相对路径规则解析。
- 数据、feature cache、checkpoint 及大型运行产物不得作为普通源码文件提交到 Git；仓库保存其机器可读元数据、fingerprint、checksum 和引用关系。

### 3.2 Run identity and required provenance

每个实验必须具有唯一 `run_id`、resolved config 和可追踪的 run manifest。最终 manifest 至少记录：

- Git commit 与 dirty state；
- 数据与 primary split fingerprint；
- feature cache 版本；
- seed；
- resolved config 及其 hash；
- 运行环境与启动命令；
- 指标；
- checkpoint 和 artifact checksum。

### 3.3 Two-phase run lifecycle

1. 运行开始时生成 `run_spec`，固化计划执行的身份、输入、配置和环境。
2. 运行成功或失败结束时生成最终 `run_manifest`，记录终态、指标、产物或失败证据。
3. 已结束的 run 不允许覆盖修改；任何修正、补跑或配置变化必须创建新 run。

### 3.4 Platform-independent interfaces

- 当前阶段不引入 Hydra、DVC、MLflow 等完整实验平台。
- 配置、metrics、manifest 与 artifact 索引必须使用机器可读、平台无关的接口。
- 未来增加 MLflow 等平台时，应通过适配器消费既有记录，不得改变核心运行语义或使平台数据库成为唯一 Source of Truth。

### 3.5 Research Gate authority

M0-M20 的所有阶段继续受 Research Gates 控制。进程退出成功、run 完成或指标文件存在，都不自动表示 Gate PASS；只有满足对应验收标准并保留规定证据后，才能进入下一阶段。

## 4. Frozen Configuration Governance

### 4.1 Human and executable Source of Truth

- `docs/research/RESEARCH_SPEC.md` 是研究语义的人类可读 Source of Truth。
- `configs/research/defaults.yaml` 是已批准研究语义的机器可执行映射，不构成独立的第二套研究定义。
- 冻结研究语义的修改必须按以下顺序进行：`DECISIONS.md` → `RESEARCH_SPEC.md` → config/schema 更新 → version bump。
- 禁止直接修改 YAML 使研究语义静默变化。

### 4.2 Layered configuration

配置按研究默认值、dataset/feature、model、experiment overlay、runtime profile 和显式 CLI override 分层解析。所有层合并后必须经过严格 schema 与研究不变量校验，并保存完整 resolved config。A1-A13 等研究变量必须通过版本化 experiment overlay 表达，不得通过手改源码表达。

### 4.3 Dual configuration identity

每个 run 至少具有以下两个配置身份：

- `semantic_config_hash`：覆盖会改变研究结论的配置，包括 dataset version、split manifest/hash、feature definition、model architecture、Markov/sensor/event setting、loss definition、evaluation protocol、experiment overlay，以及具体 multi-seed run 的 seed。
- `resolved_config_hash`：覆盖最终完整 resolved config，用于复现这一次具体运行。

`semantic_config_hash` 不得包含 batch size、gradient accumulation、num_workers、device、本地路径、checkpointing、machine name、GPU type、output directory 或 timestamp。同一研究实验更换 runtime profile 后，semantic hash 必须不变，resolved hash 可以变化。

### 4.4 Formal and debug override policy

- `formal`：CLI 只允许覆盖明确列入 allowlist 的 runtime 字段，例如 device、batch size、gradient accumulation、num_workers、precision 和 checkpointing。禁止通过 CLI 修改 split、feature semantics、model architecture、`shared_A`、`sensor.enabled`、`events.enabled`、loss definition 或 evaluation protocol。
- `debug`：可以在显式授权范围内使用研究语义 CLI override，但必须标记 `run_mode=debug`，且不得自动进入正式 paper result aggregation。

### 4.5 Deterministic canonicalization

配置 hash 必须由 canonical JSON 计算 SHA-256。canonicalization 至少满足：固定 key 排序与 UTF-8 编码；统一 bool/null、数值和 enum 表示；排除时间戳等随机字段；Windows/Linux 路径差异和本地绝对数据路径不得污染 semantic identity。禁止直接对原始 YAML 文本计算实验身份 hash。

### 4.6 Configuration provenance and semantic diff

`run_spec` 必须保存 research defaults、dataset config、feature config、model config、experiment overlay、runtime profile 各来源文件及其 hash，并保存 CLI overrides、`config_schema_version`、`research_spec_version`、resolved config 和双重配置 hash。

每个实验还必须保存 `semantic_diff.json`，以机器可读方式记录其相对正式主模型的研究语义差异。A1-A13 的审计不应依赖人工比较整份 resolved config。

### 4.7 Schema and invariant validation

schema 必须拒绝 unknown key、type error、missing field 以及跨字段/语义冲突。formal run 必须验证冻结 split hash、2Hz 主时间基准、runtime 与 research 字段边界、experiment overlay 修改权限、已批准的模块依赖，以及数据/feature cache version 的登记状态。模块依赖的具体规则只在对应设计节获得批准后冻结。

### 4.8 Forward-only evolution

至少维护 `config_schema_version` 与 `research_spec_version`。旧 run 永久保留创建时的 version 与原始 snapshot；新 schema 不得静默重新解释旧配置。迁移必须显式执行，并同时保留迁移前后记录。

## 5. Frozen Data, Identity, and Cache Governance

### 5.1 Storage zones

Git 仓库与外部 data root 分离。本机默认 data root 为 `D:\SC-MV-DMER-data`，但源码和 semantic identity 不依赖该绝对路径。外部根目录按 `raw`、`interim`、`processed`、`features`、`pseudo_labels`、`upstream_models`、`runs` 和 `tmp` 分区。

`raw` 是只读证据层；任何清洗、重采样、切片或身份规范化都必须生成新的 interim/processed 数据版本。primary split manifest 保存在 Git 中，只引用稳定身份，不引用本机绝对路径。

### 5.2 Upstream model manifest

MERT、Qwen 及其他外部预训练模型必须先登记 upstream model manifest。manifest 至少记录稳定 `upstream_model_id`、提供方/仓库、精确 revision、来源、许可信息、预期文件、文件 checksum 和本地验证结果。formal run 只能引用已登记的 manifest identity；禁止以可漂移的模型名、未固定 revision 或 `latest` 作为正式输入。

### 5.3 Stable identity contract

- `dataset_id` 标识登记的数据集版本，而不是本地目录。
- `song_id` 标识音乐作品/录音的稳定歌曲级实体；同一歌曲派生的全部片段必须共享该 ID。
- `sample_id` 标识具体可训练/评估样本，并确定性关联 `dataset_id`、`song_id`、片段边界及源记录。

三类 ID 不得依赖盘符、绝对路径或可变文件名；已发布 ID 不得复用。若源数据没有可靠 ID，必须在 ingestion 时生成并冻结 source-to-stable-ID mapping。身份纠错必须通过可追踪的修订/别名记录完成，不得静默重编号。所有 split 按 `song_id` 分组，禁止同一歌曲的样本跨 split 泄漏。

### 5.4 Derived sensor, event, and CoT versions

sensor labels、symbolic events 与 CoT targets 是三个独立的派生数据层，各自具有 version、schema 和 manifest。每个版本必须保存父级 dataset/feature/sensor/event fingerprint、生成配置 hash、生成代码版本、schema version、记录数量和内容 checksum，形成可追踪 lineage：

`dataset/features → sensor labels → symbolic events → CoT targets`。

任一上游输入或生成规则变化都必须创建新的下游版本；禁止原地覆盖或让旧 CoT 数据静默引用新事件规则。

### 5.5 Feature cache immutability

feature cache 必须绑定稳定 sample identity、dataset fingerprint、upstream model manifest（适用时）、feature definition、提取配置、提取代码版本、schema、输出 shape/dtype/time grid 和内容 checksum。cache 与 split 解耦，同一稳定样本的同一特征版本可被所有合法 split/seed 复用。

cache 构建期间使用 staging 状态；只有完整性、shape/dtype、2Hz 对齐和 checksum 校验通过后才可 finalized。finalized cache 不得修改、补写或覆盖；任何语义或内容变化必须创建新 version。formal run 必须引用精确 cache version，禁止引用 `latest` 或未登记缓存。

## 6. Frozen Module and Interface Semantics

### 6.1 Sensor and View Dropout topology

Sensor Heads 直接消费原始 `E_mel`、`E_mfcc`、`E_chroma`，且 `L_sensor` 必须反向更新对应手工视图 Encoder。每个手工视图在 Encoder 输出处分叉：Sensor branch 不经过 View Dropout；Markov/Fusion branch 在训练时接受 View Dropout。Deep view 永不 dropout。Sensor 不得从 `F_fused`、Markov state 或其派生输出获取输入。

### 6.2 Markov, Fusion, and Event boundaries

四视图拥有独立原型并共享 `A` 与 `pi`。Fusion 以 Deep view 为 Query anchor，以手工视图为 Key/Value，并使用 Markov consistency bias 和可学习 `beta_v`。禁止退化为普通 early concatenation。

EventSequence 由确定性 symbolizer 生成，必须记录 `event_source`、SensorBundle identity/hash、symbolizer rule version、segment boundaries 和源 evidence。pseudo label 与 sensor prediction 之间的来源切换必须由版本化配置和 provenance 显式表达。

### 6.3 No-LLM and Full LLM boundary

`no_llm_head` 是阶段 Gate/baseline head，不属于 Full LLM 的永久输出路径。只有 No-LLM Gate PASS 后才能集成 ChatTS-style adapter 与 Qwen。Full LLM 默认不得 ensemble、平均或回退到 no-LLM 输出。

Full LLM 使用显式双头：Qwen hidden states → VA Regression Head → `Y_va`；Qwen → LM Head → CoT/ANSWER → ANSWER Parser → `Y_answer`；二者通过 `L_consist` 约束。PredictionBundle 保存双路原始输出、解析状态/错误、解析结果、mask 与一致性证据。

### 6.4 Change control

上述模块与接口语义已冻结。后续任何修改必须先在 `DECISIONS.md` 中形成新的显式决定，再同步更新 Research Spec、Design Spec、config/schema 和版本号。

## 7. Frozen Run, Gate, and Effective-Validity Semantics

### 7.1 Immutable run lifecycle

run 在 PRE-FLIGHT 后原子写入 `run_spec`，随后进入 RUNNING，并最终产生 `SUCCEEDED`、`FAILED` 或 `INTERRUPTED` final manifest。三种终态不可逆；final manifest 后 run 逻辑封存，已有 metrics、checkpoints 和 artifacts 不得修改。

跨机器恢复、续训和 post-hoc evaluation 创建新 run，并分别通过 `parent_run_id`、`continuation_of`、`resume_from_checkpoint` 或 `source_run_id` 建立关系。formal PRE-FLIGHT 要求研究相关 Git workspace clean。stale recovery 服从版本化 heartbeat/lease policy 并记录 provenance。

### 7.2 Gate definitions and evaluation attempts

Gate Definition 保存 versioned criteria、authority、required evidence 和 predecessor。Gate Evaluation Attempt 保存 `gate_evaluation_id`、`gate_version`、attempt、`evidence_bundle_hash`、逐项结果和 verdict。同一 Gate version 允许多个不可变 attempts；criteria 或 authority 变化才增加 version。

Automatic/manual authority 必须逐 criterion 声明。Codex 可计算和推荐 verdict，但不能代替人工审批。Gate Evaluation Record 引用 evidence commit；stage-close commit 在 PASS 后提交该记录；下一阶段引用 predecessor stage-close commit。

### 7.3 Append-only invalidation and supersession

后续发现 run、artifact、evidence 或 Gate Attempt 无效时，只能追加 invalidation/supersession record，不得修改历史 terminal record、attempt 或 commit。paper/report aggregation 必须排除当前被 invalidated 或按 policy supersede 的证据。

### 7.4 Dependency-aware effective validity

Gate Attempt 的 `historical_verdict` 永久保存原始 PASS/FAIL/BLOCKED；`effective_verdict` 由当前 invalidation/supersession registry、active Gate Definition 与 predecessor dependency graph 动态计算。

当前有效性至少包含：`VALID`、`INVALIDATED`、`STALE_DEPENDENCY`。`STALE_DEPENDENCY` 表示实体自身未被判定执行失败，但其研究结论依赖的上游证据当前失效；它不修改历史事实，需要重评、重跑或有效人工影响确认。

下一阶段 entry、formal aggregation 和 paper/report generation 必须使用 effective validity。传播沿明确依赖图进行，并输出 invalidated source、直接受影响 Gate Attempt、下游 stages/runs、传播原因及 replacement evidence。Gate version supersession 通过同一 resolver 确定 active definition；旧 version 和旧 PASS 保留为历史事实，但不得作为未来阶段当前有效 predecessor。

## 8. Frozen Research Gate Criteria

### 8.1 RG-03 Sensor Independent Validation

调式强度回归在冻结 Sensor validation split 上必须达到 `CCC > 0.7`。

能量头和亮度头的 MSE 收敛定义为：在同一冻结 validation membership、同一 target normalization 和同一 metric implementation 下，model 与 train-only constant baseline 的 MSE 均为有限值，且每个 target 满足：

`MSE_model <= MSE_train_only_constant - max(1e-8, 1e-6 * abs(MSE_train_only_constant))`。

constant baseline 与 normalization 参数只能由 Sensor-train targets 拟合。test 不参与拟合、选模、early stopping、threshold 调整或 Gate verdict。训练必须按预登记 early-stopping 规则结束，并保存完整曲线、停止原因、最佳 checkpoint 与全部原始 Gate 数值。

normalization provenance 必须记录 target、pseudo-label version/hash、transform、fit-split fingerprint、拟合参数、实现/schema version 和 metric 空间。

调性分类头不设置未经研究方案支持的 accuracy/macro-F1 硬阈值。其 class-prior baseline 只使用 Sensor-train 的有效 5 秒 segments，并采用 Laplace/add-one smoothing：`p_c=(n_c+1)/(N+C)`。在完全一致的版本化 class vocabulary/order、target encoding、segment boundaries、valid/missing-label mask 和 CE reduction 下，model/prior validation CE 必须有限，且：

`CE_model <= CE_train_only_prior - max(1e-8, 1e-6 * abs(CE_train_only_prior))`。

checkpoint 选择和 early stopping 只依据预登记 validation CE；test、accuracy、macro-F1 和人工挑选不参与选模或 Gate verdict。accuracy、macro-F1、混淆矩阵、类别支持度、train counts/prior 和 smoothing 参数必须保存为诊断 evidence。

### 8.2 RG-04 No-LLM Main Model versus CRNN

逐维 CCC 接近界限 `0.02` 是预注册 Engineering Decision，不来源于原研究方案或文献。

定义 `Delta_V = CCC_noLLM,V - CCC_CRNN,V`、`Delta_A = CCC_noLLM,A - CCC_CRNN,A`。RG-04 PASS 当且仅当：

`Delta_V >= -0.02 AND Delta_A >= -0.02`。

若两项 Delta 均不小于 0，PASS 标签为 `REACHED`；否则为 `NEAR`。禁止以 V/A 平均 CCC 掩盖某一维未达标。

CRNN 与 No-LLM 必须使用相同 primary validation split、标签域、mask、metric implementation、选模协议和冻结 seed set；test 不参与 Gate。CRNN reference、threshold 和协议在结果观察前冻结。多 seed 时使用逐维算术均值进行 Gate 比较，并保留全部 per-seed 原始 evidence。

### 8.3 RG-06 Beta Learning Stability

`beta_mel`、`beta_mfcc`、`beta_chroma` 必须在每个正常完成的 validation checkpoint 保存模型实际使用的 effective value，并保留完整 curve；全部记录必须 finite。

Gate 至少需要训练时间顺序上 3 个连续正常 validation checkpoints。不足时 criterion result 为 `INSUFFICIENT_EVIDENCE`，不得 PASS。严格取最后三个正常完成 checkpoint，禁止事后挑选、围绕 best checkpoint 取样、静默删除异常点或改变窗口。

对每个 `v`，last-3 必须全部 `beta_v > 0`，并计算：

- `mean_v = mean(last3)`；
- `std_v = population_std(last3, ddof=0)`；
- `CV_v = std_v / abs(mean_v)`；
- 要求 `CV_v <= 0.10`。

`0.10` 是预注册 Engineering Decision。总体 `PASS = mel_pass AND mfcc_pass AND chroma_pass`，不得跨视图平均。

evidence 保存三条 curves、checkpoint index/epoch/step、last-3、mean/std/CV、positivity/stability、selected checkpoint、`beta_parameterization`、双配置 hash、source run IDs 和 checksum。若 parameterization 结构性保证 positivity，报告必须明确该事实，不能声称是训练发现；参数化变化必须先经 Decision。

`beta_chroma > beta_mel` 冻结为 Pre-registered Directional Hypothesis，不是 Gate PASS 条件。单独输出 `directional_hypothesis_chroma_gt_mel = SUPPORTED | NOT_SUPPORTED`；不支持时如实报告并进入 Discussion，不得重选 checkpoint、调参或修改 Gate。

### 8.4 RG-05 Markov State Collapse / Occupancy

原研究方案规定四视图各 8 states、每个状态最低使用率严格 `>5%`、监控状态使用率与 `A` row entropy，并在任一视图单状态占比 `>60%` 时报警。使用 post-Markov `gamma_tilde` hard occupancy、确定性 tie-break、把 `>60%` 升级为 formal non-PASS 以及 selected-checkpoint evaluation 均是预注册 Engineering Decisions。

RG-05 仅在冻结 primary validation split 的有效 2Hz mask frames 上，使用预登记 selected checkpoint 和 deterministic evaluation mode 计算。`model.eval()`、View Dropout 关闭、四视图全部在线；test 不参与。禁止按 occupancy post-hoc 重选 checkpoint。

对 `v in {deep,mel,mfcc,chroma}`、`n in {0,...,7}`：

- `hard_state(v,s,t)=argmax_n gamma_tilde[v,s,t,n]`，平局选择最小 state index；
- `N_valid=sum mask[s,t]`；
- `count_hard[v,n]=sum mask[s,t]*I(hard_state(v,s,t)==n)`；
- `hard_occupancy[v,n]=count_hard[v,n]/N_valid`。

32 个 cells 全部必须 `hard_occupancy[v,n] > 0.05`，且每个视图必须 `max_n hard_occupancy[v,n] <= 0.60`。总体 `PASS = AND over all views and all required occupancy criteria`；禁止平均掩盖失败。

raw `gamma` 与 post-Markov `gamma_tilde` 都保存 per-state soft mass、hard occupancy 和 histogram；`soft_mass[v,n]=sum(mask*probability[v,:,n])/N_valid` 只作诊断。完整 `A`、逐行及 mean/min/max row entropy、对应 checkpoint 和 `L_state` 也是 required diagnostics，但当前无 row-entropy 硬阈值。

evidence 保存 selected checkpoint、split/TimeGrid/mask、`N_valid`、全部 counts/occupancies/soft mass、histograms、`A`/entropy/`L_state`、tie-break rule/version、evaluation-mode evidence、双配置 hash、source run 和 checksum。

RG-05 只判断单 run 内状态健康使用。不同 seed 的 state index 具有 permutation symmetry；无额外 state-alignment protocol 时不得作同语义比较。

### 8.5 RG-07 EventSequence Token-Length

研究方案规定 7B `<=180 tokens`、14B/通用 `<=200 tokens`，并要求避免逐帧事件文本膨胀 prompt。RG-07 按 `target_llm_profile + EventSequence version + event_source` 判定；同一 artifact 可通过 14B 而不通过 7B。

使用 target profile 对应 upstream manifest 的精确 tokenizer revision/checksum。计数：

`event_token_count=len(tokenizer.encode(serialized_event_block, add_special_tokens=False))`。

truncation/padding/BOS/EOS 均关闭或不计。serialized block 包含固定标题、segments、separators、换行、标点和实际空格，不含外围 instruction、心理框架、time-series tokens、block 外 markers 或模型 special tokens。

canonical serialization 冻结为 NFC、UTF-8、LF、segment 时间升序、固定标题/标点/空格、禁止 trailing whitespace，并固定 trailing-newline 规则。保存 `event_text_sha256`，Prompt assembly 核验 measured text 与 injected text 完全一致。

全量检查 formal pipeline 所有 eligible train/validation/test records，不抽样。分别覆盖实际配置使用的 `pseudo_label` 与 `sensor_prediction` source/version。定义 `max_event_token_count=max_s event_token_count[s]`；average/median/P95/P99 只作诊断。

7B 要求 maximum `<=180`，14B 要求 `<=200`，且 `violation_count==0`。禁止 silent truncation；超限必须保留旧 artifact，显式修改并 version bump，生成新 artifact 后重评。

evidence 保存 target/tokenizer identity、event/source/SensorBundle/symbolizer/serializer identity、canonical/counting protocol、完整统计、全部 violations/maximum sample IDs、split membership、machine-readable verdict 和 checksum。stage 使用多个 EventSequence source/version 时：`PASS_stage=AND over all configured source/version results`。

### 8.6 RG-08 LLM Memory Feasibility

RG-08 使用 hardware-profile-specific Evaluation Attempts。Local RTX5060Ti 16GB 只判断 `LOCAL_FEASIBLE`；canonical RTX4090 24GB 是研究方案 reference。无 4090 时 reference 为 `BLOCKED_REFERENCE_HARDWARE`，Local PASS 不替代 reference PASS。

研究方案的 `<22GB` 预注册操作化为严格 `measured_peak_bytes < 22*2^30`（`<22 GiB`）。reference runtime 保持 Qwen2.5-7B、QLoRA 4-bit、gradient checkpointing、micro batch 4、accumulation 4；具体 resident stack 必须按 `VariantCapabilityProfile + TrainingStageDefinition + Context Profile` 编译，generation bound 服从 `GEN-DTB-VA90-3R++`。LoRA 只在 Stage 2/capability 适用时存在，canonical A1-like Stage 1 不得提前创建 LoRA。

RG-08 applicable role set 由 variant/stage/context capability 编译；possible roles 包括：P1 stage/context-specific load/resident topology probe，其中 P1[STAGE1] 加载 exact Stage-1 compiled topology 且 LoRA N/A，P1[STAGE2] 加载 exact Stage-2 topology 和适用 LoRA；P2 执行 Stage1 `forward-loss-backward-optimizer-zero_grad`；P3 执行 Stage2 两个完整 accumulation cycles，覆盖 optimizer-state lazy init 与 steady state；P4 仅在 generation capability applicable 时使用正式 Prompt/Event/90 positions/`max_new_tokens=B_think+B_answer+B_wrapper` autoregressive generation，其中 `B_think=300` 只约束 THINK payload。所有 applicable roles 使用最大正式 input 和对应 context 的实际 topology。

同时保存 PyTorch allocated/reserved peak 与 NVML process/device peak、total/start-free/background usage。`measured_peak=max(torch_peak_reserved,observed_nvml_process_peak)`；必要 source 缺失为 `INSUFFICIENT_EVIDENCE`。sub-probes 优先 fresh process，禁止清 cache 制造低 peak；actual checkpointing、4-bit、LoRA、optimizer、dtype、precision、backend、use_cache、batch/accumulation/sequence 必须验证。

Local profile 的所有 applicable required RG-08 roles 无 OOM/allocator failure/silent fallback 且 evidence 完整才 `LOCAL_FEASIBLE`；否则 `BLOCKED_LOCAL_COMPUTE`，保存 OOM/peak/config/hardware/run/partial checksum 并生成 handoff。外部运行使用新 run 与 continuation provenance。

RTX4090 reference 要求 all applicable required RG-08 roles 无 OOM、适用的完整 accumulation/optimizer init/最大 sequence 或 generation 已覆盖、measurement 完整，且每个 applicable role 的 `measured_peak_bytes <22*2^30`。`RG08_peak=max(applicable_required_role_peaks)`，要求 `<22 GiB`。A10 的 applicable set 仍为 P1/P2 Stage1、P1/P3 Stage2、FVA final，且 P4=N/A。

7B 结果不得推断 14B。14B scaling 使用独立 hardware profile、resolved config、sequence constraints、probe 和 Gate Evaluation。evidence 保存完整 hardware/software/runtime/sequence/memory/provenance identity 与 checksum。

### 8.7 RG-09 CoT Synthesis Quality

研究方案要求 automatic filtering 后人工抽检 100 条且 qualified `>=85/100`。RG-09 只从 finalized、有效、未 invalidated、训练用途的 `synthetic_cot_records` sampling；不得用 real records 稀释。test 排除。按 `target_profile + synthetic artifact version` 分开评估 7B/14B。

population<100 为 `INSUFFICIENT_EVIDENCE`。否则按 `cot_record_id` 无放回抽 exactly 100 unique records。sampling frame/hash、algorithm/version、seed、strata、selected IDs/order、artifact/profile 在 human 查看前冻结，禁止因结果换样；bug 需 invalidation 整 attempt 后新建。

两名真实 human reviewers 独立首轮且互不可见判断。相同 review package 至少含 CoT/ANSWER raw、parsed VA、EventSequence/provenance、允许的 GT VA reference、teacher audio summary（适用）和 profile/format，并保存 package hash。Codex 不得代替 reviewer/adjudicator。

每条使用冻结 binary rubric：C1 Format Integrity、C2 Event Citation Correctness、C3 VA/ANSWER Consistency、C4 No Fabricated Acoustic Evidence、C5 Explanation Usability。`qualified_r[i]=C1 AND C2 AND C3 AND C4 AND C5`；失败保存 reason codes。

任何 qualified/关键 criterion 分歧由真实 human adjudicator 处理，产生 final `adjudicated_qualified`；A/B 原始结果、notes、disagreement、adjudicator/result/reason append-only 保留。

100 条完成后 `qualified_count=sum adjudicated_qualified[i]`；只有 `qualified_count>=85` PASS，84 FAIL，不四舍五入、不换失败样本、不改分母。human 未完成=`PENDING_MANUAL_REVIEW`。

保存 reviewer identity、order/session/blinding、逐条 C1–C5、failure/adjudication、aggregate reasons 和 agreement diagnostics。agreement/kappa/criterion agreement、可选 CI 是 non-gating diagnostics。evidence 同时保存 artifact/sampling/rubric/package/run/config/commit/schema/checksum provenance。

### 8.8 RG-10-v2 ANSWER → VA Parser Stability

`RG-10-v2` 是 current Gate Definition；DEC-0015 的旧 mask-dependent count interpretation 作为 historical `RG-10-v1` 保留并由 DEC-0026 supersede。此次只同步 Gate authority，没有创建或伪造 evaluation attempt。

研究方案要求 `parser_success_rate>=0.98`。RG-10 按 `target_llm_profile + answer_schema_version + parser_version + generation_protocol_version/hash` 独立评估；Qwen2.5-7B/14B 不共享 PASS。每个 attempt 绑定 source generation run、checkpoint checksum、config hashes、prompt、tokenizer/upstream identity。

使用冻结 primary validation generation population 全量评估，不抽样且 test 排除。generation infrastructure/execution failure 导致 population 未完整形成时不进入 parser denominator，attempt 为 incomplete/BLOCKED/non-PASS；已正常完成 generation 并形成 raw text 的 empty/missing/malformed/truncated/invalid ANSWER 则必须进入 `N_eligible` 并计 unsuccessful。

`parser_success` 要求 S1 唯一合法 ANSWER；S2 令 `T=canonical output TimeGrid length`，`parsed_V_count==T AND parsed_A_count==T`，标准 DEAM 45s 的 `T=90`，不得用 `sum(TargetValidityMask)` 或 valid-mask-derived length；S3 canonical TimeGrid 每个位置 V/A 齐全；S4 冻结 grammar 可确定解析；S5 全部 finite；S6 满足既有 VA domain 且不 clipping/rescale；S7 结构无 missing/extra/重复显式位置/错序；S8 Prediction/ANSWER 绑定正确 Sample/TimeGrid/TargetValidityMask metadata，下游 target-based evaluation 严格使用 mask，但 mask 不缩短 ANSWER。合法的相邻/不同时间点重复 VA 数值不算错误，只禁止重复结构或显式时间位置。

`parser_normalization_version` 只能包含预注册 deterministic representation normalization；禁止补数、删/补 pairs、重排、clipping、GT-guided 或 LLM semantic repair。Parser 只读取 raw text、schema 与 expected TimeGrid/mask metadata，GT 不得用于 reconstruction。规则修改必须 bump version 并新建 attempt。

completed generation 不得因 parse failure retry、换 seed/decoding、局部重跑、人工修复、替换、删除或改 denominator。bug 需 invalidation 原 attempt，按需 bump parser/generation version，创建新 evidence/new attempt。

`parser_success_rate=N_success/N_eligible`，PASS 精确判定 `100*N_success >= 98*N_eligible`，不按显示百分比四舍五入。generation protocol 冻结 decoding、sampling/seed、max_new_tokens/stop、tokenizer、Prompt Builder 和 answer-format instruction。

逐失败 record 保存 immutable raw output/hash 与结构化 multi-reason taxonomy；保存 ANSWER/schema/numeric/time-grid/range diagnostics，但均 non-gating。evidence 覆盖 identity、full population、逐 record parse/count/range/finite/mask/failure、aggregate exact fraction、run/config/commit/manifests/schema/checksum provenance。

RG-10 不决定 parser failure 在训练中如何进入 `L_consist`；该 loss 行为必须在后续 Loss/Training Protocol 单独冻结。

### 8.9 RG-01 MERT Actual Frame-Rate & Time-Axis Contract

研究方案冻结 MERT input `24,000Hz`、标准片段 `45s`/`1,080,000 samples`、hidden dim `768`、研究第 5/6 层，以及“真实 forward 先验证，再反推 TimesNet/2Hz/90-frame downstream dimensions”。约 50Hz/约 2250 frames 不是可硬编码模型真值。

Gate identity 为 upstream model manifest、processor/preprocessor、resampler、layer-selection 与 `MERTTimeAxisContract` versions。layer 5/6 必须记录 returned-hidden-state semantics、index convention、actual tensor indices 与 block identity；具体 indices 从 pinned implementation 实测，禁止凭记忆填写或发生 silent off-by-one。

Production preprocessing 确定地产生 `1,080,000@24kHz`，保存 source audio/hash/rate/count、resampler、crop/pad、channel 与 dtype。P1 使用固定 exact-length waveform；P2 使用至少一条 registered real DEAM 45s sample。二者均执行真实 MERT forward；config/README/mock 不替代 evidence。Layer 5/6 T/D 相同、D=768、outputs finite；revision 不符时不得 projection 后宣称 PASS。

`observed_fps=observed_frame_count/45` 仅为 diagnostic。真实时间轴从 pinned feature extractor kernels/strides/padding/dilation/reductions 推导 effective stride、receptive field、first support、frame centers 与 expected count，并严格要求 `expected_frame_count_from_contract==observed_real_forward_frame_count`。timestamps 优先用 integer sample/rational representation。

2Hz TimeGrid 为 90 个 0.5s 半开 bins `[0,0.5)...[44.5,45.0)`。用 versioned sample-domain deterministic arithmetic 分配 frames；每个有效 frame exactly one bin，`sum(bin_frame_counts)==observed_frame_count`、unassigned=0、duplicate=0，且全部 90 bins 非空。禁止 silent trim/pad/drop/clamp、frame copy/fill/interpolation 或 adaptive pooling隐藏问题。每 bin 对所属 valid frames 使用冻结 mean reduction，`T=90` 从 contract 推导。

TimeGrid/Mask 显式表达 invalid source/frame/bin；标准完整 formal record 若协议要求全有效则 `valid_bin_count=90`。重复 probe 与受支持跨机器环境必须保持 sample/frame count、shapes、layer/timestamp/bin mapping 和 bin counts；不要求不同硬件 activation bitwise identical，只要求 finite 与 structural/time contract 一致。

任何 model weights/revision、processor/resampler、crop/pad、layer selection、temporal geometry、timestamp/binning 变化必须新 contract/cache version、effective invalidation/supersession、新 RG-01 attempt 和受影响 cache 重验。evidence 保存 upstream/preprocessing/layer/forward/geometry/90-bin/probe/config/commit/hardware/checksum provenance。

### 8.10 RG-02 Handcrafted Feature Binning & Cross-View Alignment

研究方案冻结 Mel 22.05kHz/`n_fft=2048`/`hop=512`/128/power2→dB→`[90,128]`，MFCC 共享 STFT timing/40→`[90,40]`，Chroma CQT `hop=512`/12/`bins_per_octave=36`/逐 raw-frame L1→`[90,12]`；三者 window-center assignment、within-bin mean 到 2Hz/90 bins，并人工可视化 3 首歌检查 waveform 与四视图对齐。

RG-02 依赖当前有效 RG-01 PASS 及其 sample/segment/TimeGrid/mask/Deep timestamp contract；保存 predecessor evaluation/evidence/stage-close provenance，上游失效按 dependency-aware effective validity 传播。

Gate identity 绑定 decoder、22.05k resampler、Mel/MFCC/Chroma configs、handcrafted timestamp/binning 与 TimeGrid versions。显式冻结 library versions、center/pad/CQT alignment/edge behavior、dtype/channel 和 timestamp rules。三个手工视图必须消费同一 22.05k waveform，并与 Deep 共享 dataset/song/sample、45s segment、TimeGrid 与 mask。

raw-frame timestamps 从 sample rate/hop/centering/padding/support 用 integer/rational arithmetic 推导。每 frame 按中心恰好进入一个 RG-01 0.5s bin；每 bin mean。每 `record×view` 要求 assigned=raw eligible、unassigned=0、duplicate=0、bin counts sum=raw eligible，且全部 90 bins 非空。禁止 silent trim/pad/clip、复制/fill/interpolation/adaptive pooling 或 reshape 制造 90。

自动检查覆盖 train/validation/test 全部 formal eligible records，不读取 target 或选模。所有 Mel `[90,128]`、MFCC `[90,40]`、Chroma `[90,12]` 必须 schema/dtype/finite/TimeGrid/mask 合法；任一 record 失败即 automatic FAIL。MFCC c0 policy 写 manifest。

Chroma 顺序为 raw→per-frame L1→bin assignment→bin mean。L1 calculation、epsilon（如有）、zero/near-zero behavior 和 tolerance 必须在 attempt 前预注册并版本化，不冒充研究原要求；非零 frame L1 norm 在 tolerance 内为 1，zero/near-zero 不得伪造 tonal distribution，保存 counts/violations。

Cross-view alignment 只要求相同 `t` 是同一 real-audio 0.5s window，不要求 curves、correlation 或 peaks 一致。manual sampling 在查看前冻结 3 个 unique train songs、population/algorithm/seed/IDs/order，无放回；不得换失败样本或使用 validation/test。

versioned review package 共享 0–45s axis，显示 waveform、boundaries、四视图预登记 summaries、frame/bin counts、coverage/mask。真实 human reviewer 判 M1 identity、M2 start/end、M3 bin consistency、M4 applicable event timing、M5 no systematic offset；M1/M2/M3/M5 必须 PASS，M4 applicable 时 PASS，N/A 需原因。三个 samples 全 AND。Codex 不得代替 human approval。

`RG02_PASS=AUTOMATIC_ALL_RECORDS_PASS AND MANUAL_3_SONGS_PASS`。evidence 保存 predecessor、identity/preprocessing/extractors、逐 record accounting/shapes/checksums、Chroma normalization、cross-view equality、manual sample/package/reviewer/M1–M5、run/config/commit/cache/schema/checksum provenance。修复需新 definition/cache/version 与新 attempt，旧 artifact 不覆盖。

## 9. Frozen Primary Experiment Matrix

### 9.1 Semantic closure and authority

Section 6 Experiment / Ablation Matrix 已完成研究语义闭包并标记 `APPROVED / FROZEN`。A1–A13 构成当前 `PRIMARY_EXPERIMENT_MATRIX_CLOSED_WORLD`。该状态只表示科学问题、实验身份、base 关系和必要语义闭包已冻结；`machine_readable_normalization=PENDING`、`executable_binding=PENDING_DEPENDENCIES`、`formal_ready=false`。

人类可读 Experiment Catalog Source of Truth 为 `docs/research/EXPERIMENT_MATRIX.md`。后续协议只能绑定训练、loss、metrics、prompt、artifact、Gate applicability、Stage 和 reproducibility，不能因依赖未解析而重新定义 A1–A13。真实冲突必须通过 `DECISIONS.md` 和 version-change review。

### 9.2 Frozen identities

15 个 executable variants 为：

1. `A1_FULL_7B / A1-v1`
2. `A2_DEEP_ONLY_COT / A2-v1`
3. `A3_INDEPENDENT_A / A3-v1`
4. `A4_STATE_BIAS_DISABLED / A4-v1`
5. `A5_EVENT_INJECTION_DISABLED / A5-v1`
6. `A6_SENSOR_HEADS_DISABLED / A6-v1`
7. `A7_VIEW_DROPOUT_DISABLED / A7-v1`
8. `A8_DEEP_PLUS_MEL / A8-MEL-v1`
9. `A8_DEEP_PLUS_MFCC / A8-MFCC-v1`
10. `A8_DEEP_PLUS_CHROMA / A8-CHROMA-v1`
11. `A9_TIMESNET_DISABLED / A9-v1`
12. `A10_VA_ONLY_LLM / A10-v1`
13. `A11_CONSENSUS_ENABLED / A11-v1`
14. `A12_HSIC_DECORRELATION_ENABLED / A12-v1`
15. `A13_FEATURE_CONCAT_BASELINE / A13-v1`

这些 identities 唯一、版本化，并在被 formal evidence 使用后保持不可覆盖。`A8_SINGLE_HANDCRAFTED_VIEW` 仅为 non-executable group；不产生 run、seed 或 checkpoint，三个 children 各自 `COMPLETE_S5` 后方可声明 A8 family complete。

### 9.3 Scope, obligation, and S5

Experiment Catalog 必须把 `scope_tier` 与 `execution_obligation` 分为两个字段。scope 至少表达 PRIMARY/SECONDARY/OPTIONAL/CONTINGENCY；obligation 至少表达 REQUIRED/CONDITIONAL/NOT_TRIGGERED/OPTIONAL_NOT_COMMITTED/BLOCKED 或语义等价状态。二者由冻结 Scope Hierarchy、Trigger Protocol 与 Resource Policy 决定，禁止按实验结果改变。

A12 为 PRIMARY、CONDITIONAL；其 trigger 服从 `A12-HGST-Z0-3R++`，若预注册 trigger 不满足则标记 NOT_TRIGGERED，但不从 catalog 删除。所有 trainable formal variants 使用相同冻结 S5 并以 `COMPLETE_S5` 作为最终 evidence；A8 三个 children 分别满足，A12 若触发也满足。S5 冻结为 `SC-MV-DMER-FORMAL-S5/v1`，有序整数为 `[52826381, 128866372, 1616929435, 1871035633, 1460830465]`，选择与 RNG domain 分别服从 `SS-3+`、`RNG-3+`。

### 9.4 Scientific factors

- A1：完整 7B canonical model。
- A2：完整 handcrafted subsystem removal，保留 Deep 与 event-free CoT。
- A3：共享 transition matrix 改为 per-view independent A，π 仍共享。
- A4：移除 predictive state bias，但保留 Markov definitions/diagnostics。
- A5：只移除 prompt-side EventSequence injection；Sensor/Event artifacts 和 A1 event-derived targets 保留。
- A6：移除 local Sensor Heads/`L_sensor`，保留 paired immutable A1 Event Replay。
- A7：移除 View Dropout，并与 A1 接受相同 non-gating missing-branch stress suite。
- A8：三个 child 分别测量 Mel/MFCC/Chroma 完整 subsystem 在 Deep-only 之上的 conditional total contribution。
- A9：移除 TimesNet，以严格 RG-01 timestamp/bin mean 接入相同 Deep post-bin path。
- A10：保留 Event-conditioned Qwen/LoRA 与 hidden-state VA head，移除生成式 CoT/ANSWER subsystem；它不是 No-LLM。
- A11：启用 strict forward-KL posterior consensus。
- A12：按预注册 trigger 条件启用 RBF-HSIC representation decorrelation。
- A13：base NONE 的 independent four-view concat + No-LLM VA head baseline。

A2/A8、A3/A11、A4/A13、A5/A10、A6/A13 均研究不同因素，不得合并解释。

### 9.5 Event, Markov, and comparison boundaries

Event 语义严格区分：A2 无 handcrafted/Sensor/Event 且使用 event-free target；A5 保留 Sensor/Event artifacts、Event 不可见并保留 A1 event-derived target；A6 无 local Sensor、使用 paired A1 Event Replay 并保留 A1 target。

Markov 语义严格区分：A3 改变 A sharing；A4 只移除 predictive state bias；A11 增加 posterior consensus；A12 增加 representation decorrelation。

A10 与 No-LLM 不得复用 identity、checkpoint、RG-04 verdict 或 aggregation row。A13 的 `base_experiment_variant_id=NONE`、`primary_comparison_anchor=A1_FULL_7B`，只通过 component contract refs 共享 dataset/feature/Encoder/TimeGrid/evaluation 语义。A13 永久标记 `KANG_HERREMANS_INSPIRED`、`NOT_EXACT_REPRODUCTION`。

### 9.6 Closed-world boundary

当前 trainable primary 闭世界仅覆盖 A1–A13 matrix。`PME-ZST19-A1S5-3R++` 已单独授权 A1-S5 required zero-shot PMEmo evaluation，但不增加 trainable A-variant。14B scaling、14B CoT-DPO、Qwen2.5-Audio 与 static MER contingency 为 `DEFERRED_NON_BLOCKING`，在获得独立 versioned catalog entry 前均为 `FORMAL_RUN_NOT_YET_AUTHORIZED`。

所有 executable blockers 统一登记在 `EXPERIMENT_MATRIX.md` 的 Blocking Dependency Register。它们可以阻断 PRE-FLIGHT、Stage、Gate、artifact、aggregation 或 formal readiness，但不得静默重开已冻结 scientific semantics。

## 10. Frozen Foundation Batch Authorities

### 10.1 Training execution boundary

Batch 1 authority is `TEB-FBC-3R+++`, including `VCP-CCEW-3R++` and `A10-RG08-ASAND-FVA-DP-3R+++`. It binds the versioned Stage Contract Graph, `EventSourceCurriculum + CoT Artifact Boundary (Scheme B+)`, final inference `FI-A+`, logical epoch 3-of-5 phase boundary `PB-1+`, complete-curriculum terminal checkpoint selection `CS-3+`, immutable typed Stage 1→2 handoff `H3+`, capability-compiled parameter policy `SPP-3R+`, module/gradient policy `MMP-GT-3R+`, and stage loss activation matrix `LSAM-3R+`.

Stage 1→2 creates a new TrainingStageRun/run_id, transfers model state through H3+, rebuilds optimizer, reinitializes scheduler, and starts a new Stage-2 RNG namespace under the same S5 lineage. Stage 2 logical epochs 1–3 are early and 4–5 late; the switch fires only after epoch 3 is committed and does not reset run, optimizer, scheduler, LoRA, parameters, or Stage-2 RNG. Early uses pseudo Events; late uses frozen Stage-1 Sensor-snapshot Events while live Stage-2 Sensor continues training where applicable.

Final A1 inference uses the terminal Stage-2 checkpoint's own Sensor-derived events from real input coverage. FinalEventDisposition is: A1/A3/A4/A7/A8/A9/A11/A12 local model-visible; A5 local shadow-only; A6 paired same-S5 A1 replay model-visible with no local Sensor/producer; A10 local model-visible for its Event-conditioned VA path; A2/A13 N/A. Terminal runs are immutable; continuation uses a new run and exact provenance. Only logical-epoch-5 terminal checkpoints are designated; no validation/loss/parser/sigma ranking or earlier fallback.

A VariantCapabilityProfile expresses only scientific existence (`PRESENT`/`NOT_APPLICABLE_BY_DESIGN`); a CompiledExecutionCapabilityManifest resolves context activation (`ACTIVE`/`INACTIVE`/`NOT_APPLICABLE`); ExecutionReadiness independently resolves READY/dependency-blocked/Gate-blocked/stale/invalid. The compiler explicitly covers module, Sensor, Event producer/consumer/visibility, prompt, Qwen, LM-head, parser, generation, cache, stage/phase/inference, parameter/handoff, RG-08/Gate/artifact and forbidden roles. Formal runtime/CLI cannot mutate it.

A10 has no LM generation/parser/CoT path but consumes local model-visible Event for its VA path. RG-08 retains P1 load/resident, P2 Stage-1 training, P3 Stage-2 training, P4 generation; A10 adds FVA final-VA load/forward. Its required roles are P1/P2 Stage 1, P1/P3 Stage 2 and FVA; P4 is N/A and cannot contribute a zero/pass/fail or borrow A1 evidence. Canonical PASS requires every applicable role effectively valid/no OOM/no fallback and max peak `<22*2^30`; local 5060Ti evidence cannot replace RTX4090 authority.

### 10.2 Loss, optimization, and reproducibility

Batch 2 authority is `LOR-FBC-3R+++`, including `COTC-SN21-DSDP-3R+` and `A13-NLBTP-VAS-3R+`. The detailed loss authorities are `DLC-P1-MESM-3R+`, `DLC-P2-TFES-3R+`, `DLC-P3-ARSA-3R+`, `DLC-P4-NEHF-3R+`, revised `DLC-P5-FHST-3R+`, `DLC-P6-DAAI-3R+`, and `DLC-P7-KLV-CN-3R+`; the rejected P5 proposal is historical and not current authority.

Every active loss is capability/stage compiled and reports numerator, denominator, and eligibility. `HLAC-HDSA-3R+` preserves denominator-safe hierarchical reduction across frames, views, songs, microbatches, and optimizer steps. Stage 1 is `L_VA+0.01L_state+0.10L_sensor+I_A11*0.005L_consensus+I_A12*0.005L_HSIC`, without Kendall parameters. Stage 2 is `sum_{i∈C}[0.5exp(-s_i)L_i+0.5s_i]+0.05L_smooth+0.01L_state+0.10L_sensor+I_A11*0.005L_consensus+I_A12*0.005L_HSIC`, with A1-like `C={VA,CoT,consist}` and A10 `C={VA}`. Phase change cannot mutate active sets/weights/uncertainty/optimizer membership.

Zero eligibility does not create a numeric loss or update Kendall log-variance. If the contract requires no update, conditional parameters have `grad=None`, not a zero tensor that allows AdamW momentum/decay mutation. Completed parse failure means zero consistency eligibility plus retained evidence; generation/runtime infrastructure failure is FAIL/INTERRUPT. No failure is ground-truth repaired, retried, or removed from required diagnostics.

`COTC-SN21-DSDP-3R+` means Synthetic-CoT / Native-DEAM 2:1 Deterministic Song Partition. For `N` unique StageTrainingPopulationManifest songs, `N_syn=floor(2N/3+0.5)` and `N_native=N-N_syn`; a domain-separated SHA-256 ranking over contract/dataset/split/population/song identity freezes the S5-independent membership without reading targets, difficulty, loss, statistics, or predictions. Roles remain invariant across Stage-2 phases.

Synthetic songs require the phase-matched immutable approved CoT/ANSWER artifact; missing/wrong artifact is integrity failure. Native-DEAM songs legally have no teacher target and `L_CoT` zero eligibility, but retain all other applicable objectives and can use online `L_consist`. They are not real/human/ground-truth CoT; the current independently-human CoT population is empty. This binds A1/A2/A3/A4/A5/A6/A7/A8/A9/A11/triggered A12; A10/A13 are N/A. Human CoT and difficulty weighting are deferred non-blocking extensions.

`A13-NLBTP-VAS-3R+` governs one independent five-logical-epoch baseline stage with exact `L_A13=L_VA+0.05L_smooth`, fixed weights and no Kendall. It uses the same split/songs/TimeGrid/targets/S5/TSC, canonical 16-song logical optimization step and OSNP/OSNS Stage-1 AdamW numeric profile (`lr=1e-4`, warmup 10% committed-update ceiling, betas 0.9/0.999, eps 1e-8, WD 0.01/0, max grad norm 1.0, AMSGrad false). Only epoch-5 terminal is designated; validation/test do not rank. A13 has no Markov/Sensor/Event/ViewDrop/ChatTS/Qwen/LoRA/LM/CoT/Parser/generation and is neither A1 reuse nor an exact external reproduction.

`DLC-P5-FHST-3R+` exclusively governs Flat-Head Sum Sensor Supervision: `L_sensor=L_rms+L_brightness+L_mode+L_key`, 12-way tonic key classification, continuous `[-1,1]` mode regression, and `lambda_sensor=0.1`. It applies to A1/A3/A4/A5/A7/A8 children/A9/A10/A11/triggered A12 and is N/A for A2/A6/A13. It does not own `L_smooth` (DLC-P1/A13 authority), `L_HSIC` (A12+DLC-P6/HLAC), or A11/A12 semantics.

`TSC-GPESA-3R+` defines global-permutation exact-once logical epochs. `RNG-3+` defines namespaced RNG domains; `SS-3+` freezes `SC-MV-DMER-FORMAL-S5/v1 = [52826381, 128866372, 1616929435, 1871035633, 1460830465]`. `CRR-HRSC-3R++` governs only per-checkpoint-invocation stochastic replay for activation-checkpoint original/recompute equivalence: invocation/role identity, component RNG entry/exit refs or hashes, shadow-generator replay, call trace, execution profile and capsule checksum. It is not durable TrainingState.

Same-stage continuation restores the durable TrainingState containing applicable model/optimizer/scheduler, committed-update counter, sampler cursor, live RNG states, Stage/logical epoch/PB phase, artifact/provenance refs and GradScaler. Mid-step failure discards partial step-local capsules and TSC replays from the last durable LogicalOptimizationStep boundary; CRR governs stochastic recomputation inside that replay. `OSNP-SLAW-3R++` and `OSNS-DB10-3R+` govern optimizer/scheduler/numerical behavior. 7B uses `LQ7-R16A32-ZD-NF4-3R+` and `QKB-SPMS-3R++`; actual replay, loader, gradient and numeric behavior require executable qualification before formal use.

## 11. Frozen Metrics, Evaluation, and Global Stage Contracts

### 11.1 Metrics and S5 evidence

Batch 3 authority is `MEGPC-FBC-3R++++`. Under `MEP-CCC2-S5-3R+`, CCC uses FP64 pooled valid-frame population moments and MSE uses pooled valid-frame squared-error sums. V/A are separate. Per-seed values, S5 arithmetic mean, and sample standard deviation (`ddof=1`) are required; no best seed. Final variant evidence requires exact `COMPLETE_S5`, and paired deltas use identical seeds.

`Y_va` is primary. `Y_answer`, parser coverage, raw ConsistencyMSE, EventCitationPrecision, Sensor/Markov/beta/gradient/stress/denominator diagnostics remain distinct streams and cannot replace or average into primary `Y_va` evidence. Test does not participate in checkpoint selection, threshold choice, A12 triggering, or Gate verdicts unless a Gate explicitly defines final locked test reporting after records are sealed.

### 11.2 Dual token budget and exact ANSWER

`GEN-DTB-VA90-3R++` is a `PRE_REGISTERED_ENGINEERING_CORRECTION`: the research-plan 300-token statement is interpreted as `THINK explanatory payload <=300 pinned-tokenizer tokens`, not total generated tokens.

`T` equals canonical output TimeGrid length, never target-valid count; standard DEAM 45s has `T=90`. V and A each output exactly T signed decimals with four decimal places in `[-1,1]`. Target textualization uses deterministic round-half-even. TargetValidityMask never shortens output; no trim, repair padding, interpolation, clipping, reordering, or result-dependent format change.

`B_think=300`; `B_answer` is the pinned-tokenizer-qualified exact ANSWER schema upper bound; `B_wrapper` is the pinned-tokenizer-qualified structural wrapper bound; `max_new_tokens=B_think+B_answer+B_wrapper`. No unspecified delimiter margin is permitted. Qualification failure blocks formal execution. This identity binds RG-09, DLC-P2, DLC-P3, RG-08 Stage-2/final-generation probes, and RG-10.

### 11.3 Evaluation input coverage

`EvaluationInputCoverageContract` distinguishes `InputCoverageMask` and `TargetValidityMask`. Every padded/uncovered position has applicable `MODEL_INPUT_VALID=false`, `TARGET_VALID=false`, and `SENSOR_EVENT_VALID=false`. Padding is not silence and cannot create Sensor/Event evidence. Secondary pipelines must enforce this contract in a machine-verifiable form before execution.

### 11.4 Event citation

`ECA-ATOM-PREC-3R++` freezes `EventCitationPrecision` (precision semantics). Citable events have stable zero-padded `EventAtomID` values in frozen serialization order; global identity is `(event_sequence_id, EventAtomID)`. The only formal citation syntax is `<CITE id="E03"/>`. Exact parser/registry lookup determines recognized and supported citations; free-form NLP is not denominator authority.

Evidence reports recognized, supported, unsupported, precision, song coverage, mean citations, malformed citations, and EventSequence/schema/parser identities. Zero recognized citations yields `UNDEFINED_NO_CLAIMS`. A5 uses its own model-invisible shadow Event artifact; A6 uses paired A1 replay; A2/A10/A13 are N/A. A serialization change requires exact RG-07 requalification without moving the threshold.

### 11.5 A12 trigger

`A12-HGST-Z0-3R++` defines `G_d=mean_s(CCC_A1,s,d-CCC_A2,s,d)` for `d∈{V,A}` over exact S5 and `tau=1e-12`. Gain is sufficient iff `G_V>tau AND G_A>tau`; A12 triggers iff either is `<=tau`. Missing, undefined, invalid, or stale input yields `BLOCKED`/`STALE_DEPENDENCY`, not a trigger outcome. Test remains locked until the TriggerRecord is immutable.

### 11.6 Secondary required evaluations

`DEAM-L58-SW45-3R++` is A1-only, no-retrain, secondary required. It uses 45s windows, 22.5s stride, 0.5s-grid starts, no hidden-state carry, arithmetic overlap means, full-song pooled CCC/MSE, and per-song diagnostics. Tail windows carry both masks; only input-covered and target-valid predictions enter metrics. Each window's EventSequence is generated by the evaluated A1 checkpoint's Sensor over real coverage.

`PME-ZST19-A1S5-3R++` is A1-S5 zero-shot secondary required. Timeline is `DISTRIBUTED_CHORUS_CLIP_LOCAL_TIME`: supplied chorus excerpt local `t=0`; full-song offset is provenance only. It uses updated-2019 artifacts, no adaptation/calibration/extra inputs, actual excerpt audio plus coverage mask. A hashed manifest determines annotation validity; first-15s deletion and raw target domain require audit. Only audited `[0,1]` permits `y=2*y_raw-1`; otherwise binding fails.

### 11.7 Frozen projector and heads

`FPP-VAH-MIN-3R+` freezes the fusion projector `Linear(D_in,256,bias=true)→GELU→LayerNorm(256,eps=1e-5)`, where A1/A13 use `D_in=1024` and A8 uses `D_in=512`; the No-LLM/A13 VA head `Linear(256,128)→GELU→Linear(128,2)→tanh`; and the Qwen7B VA head `Linear(3584,512)→GELU→Linear(512,2)→tanh`. Dropout is zero. Tanh is a forward activation, not post-hoc clipping.

### 11.8 Stage/Gate applicability, eligibility, and closure

`SGA-M0M20-EV-3R+` defines M0–M20 predecessors, Gate applicability, stage entry/exit, and effective-validity requirements. `PER-EVAND-S5-3R++` requires valid run identity, evidence lineage, applicable Gates, effective predecessor chain, exact S5 completeness, and no active invalidation. Historical PASS alone never authorizes entry or aggregation.

`PRIMARY_CLOSURE` covers required 7B primary evidence. 14B, DPO, Qwen2.5-Audio, optional and contingency experiments are `DEFERRED_NON_BLOCKING` and cannot block current Foundation closure. DEAM long58 and PMEmo remain required secondary A1 evaluations under their registered contracts.

## 12. Evaluation Scope Deviations and Claim Firewall

The `EvaluationScopeDeviationManifest` registers `FINAL_GENERATION_HUMAN_EXPLANATION_EVAL` and `MARKOV_HUMAN_STRUCTURE_ALIGNMENT` as `DEFERRED_NON_BLOCKING_SOURCE_PLAN_DEVIATION`. RG-09 human review qualifies synthetic CoT artifacts only; it is not final-generation human explanation evaluation.

Without human structure-alignment evidence, reports may claim only measured non-degenerate transition dynamics, occupancy structure, and interpretable transition matrices; they may not claim learned true musical structure or alignment with human chorus boundaries. Without final-generation human evaluation, reports may not claim final explanations were human-validated.

## 13. Foundation Closure Status

The frozen batch versions are `TEB-FBC-3R+++`, `LOR-FBC-3R+++`, and `MEGPC-FBC-3R++++`. No unresolved research-semantic contradiction was found. Current status is:

```text
FOUNDATION_RESEARCH_SEMANTICS = APPROVED / FROZEN
FOUNDATION_DESIGN_VERDICT     = CONDITIONAL_PASS_EXECUTION_BLOCKED
FORMAL_EXECUTION_READY        = NO
M0                            = READY_TO_CLOSE
```

The user's external conversational approvals are recorded in `DECISIONS.md`; M0 is not marked PASS because no separate machine-verifiable approval artifact/transcript checksum exists. Execution blockers are authoritative in `EXPERIMENT_MATRIX.md`; contract identities in `FOUNDATION_CONTRACT_INDEX.md`; audit evidence in `FOUNDATION_CLOSURE_AUDIT.md`.
