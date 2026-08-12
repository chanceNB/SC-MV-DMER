# SC-MV-DMER 归档 Support Plans 实验阶段顺序改写设计

**状态：** APPROVED INPUT DESIGN（用户附件直接授权改写）

**日期：** 2026-08-12

## 1. 目标

将 `docs/superpowers/plans/archive/2026-08-12-blocker-oriented/` 中仍以 P0–P5 package DAG 为顶层顺序的计划，改写为以 `E0→E1→E2→E3→E4→E5→E6` 为最高执行权威。P0–P5 仅保留为 support work-package tag；不得再决定实验阶段 entry。

## 2. 采用方案

考虑过三种方式：

1. 全部删除后重写：阶段最清楚，但会丢失已经正确的 TDD、接口和命令细节。
2. 只改 Master：改动最小，但子计划的 P1/P2/P3/P4/P5 aggregate Exit Gate 仍会错误阻塞主线。
3. **重写 Master + 定点修改五份时序敏感子计划（采用）**：保留底层任务，在每份文件加入 `experiment_stage_owner`、Entry Gate、Required-before、can-prepare-early 和分阶段 Exit Gate；拆除错误的全包 AND。

`execution-governance` 保持底层技术内容不变，作为 E0 support。修改文件为 Master、data-binding、qwen-binding、training-qualification、variant-artifacts、external-evidence。

## 3. 顶层阶段合同

- E0：最低 governance/bootstrap 与 DEAM primary registration/split identity；不等待 PMEmo、Qwen、A5/A6、CoT、RG-09。
- E1：MERT actual forward、24 kHz/45 s/1,080,000 samples、layers 5/6、timestamps、2 Hz/90 bins、dimensions、RG-01。
- E2：Mel/MFCC/Chroma offline cache、cross-view alignment、3-song human review、RG-02。
- E3：pseudo-label-only Sensor pre-validation、RG-03；不接 Markov/Fusion/Qwen。
- E4：No-LLM four-view/Markov/ViewDropout/Fusion/VA、RG-04 与 applicable RG-05/06。
- E5：Qwen 7B pipeline，从 pinned upstream 到 Stage 1、H3/Sensor snapshot、Event/RG-07、CoT partition/materialization、RG-09、Stage-2 RG-08、epochs 1–5、final event、RG-10、primary DEAM evaluation。
- E6：A1 COMPLETE_S5、required ablations、A12 trigger、long58/PMEmo secondary、optional non-blocking 14B。

## 4. 时序修正

- RG-08 按 role 分布在 E5 Stage-1 entry、Stage-2 entry 和 final generation boundary；A10 独立 profile 在 E6/A10。
- RG-09 位于 CoT artifact 和任何依赖该 artifact 的 Stage 2 之间；未 PASS 不解锁对应 Stage 2。
- A5 firewall 与 A6 replay 只阻塞 E6 对应 ablation；不阻塞 canonical A1 E1–E5。
- PMEmo audit 可提前准备，但只阻塞 E6 PMEmo evaluation。
- long58 只在 primary A1 evidence 后执行，不阻塞 E1–E5。
- 14B 位于 7B primary 之后，资源不足保持 deferred。

## 5. Blocker 字段

Master 中每个 OPEN blocker 必须拥有：

- `experiment_stage_owner`
- `blocks_primary_critical_path`
- `required_before_exact_step`
- `can_prepare_early`

Late/secondary blocker 不能通过 package-level aggregate Gate 反向阻塞早期主线。

## 6. 子计划修改边界

- data-binding：Tasks 1–3 属 E0/E1 support；Task 4 分别服务 E0/E1/E2 与 E6 coverage；Task 5 PMEmo 属 E6；Task 6 改为分阶段 evidence，不要求 PMEmo 才释放 E1。
- qwen-binding：Tasks 1–3/5 属 E5.1–E5.2；Task 4 A10 属 E6/A10；aggregate closure 分裂，A10 不阻塞 A1 E5。
- training-qualification：横向 support 按第一次实际使用前放置；不再要求整个 P3 一次性 PASS 才能进入所有训练阶段。
- variant-artifacts：CoT/Event Tasks 属 E5.5–E5.8；A5/A6 Tasks 属 E6 对应消融；aggregate closure 分裂。
- external-evidence：RG-08 Tasks 按 E5 role boundary 调度；RG-09 Tasks 位于 E5.8；不再整体 late-bound。

## 7. 不变项

不修改科研语义、Gate threshold、A1–A13、S5、split、loss、Event visibility、artifact immutability 或人工 authority。只修改计划顺序与依赖表达；不实现代码、不构建数据、不加载模型、不训练、不执行 Gate、不 `git init`。

## 8. 验收

- Master 第一层严格为 E0–E6，包含完整 critical path、parallel preparation 和每阶段的九项合同字段。
- 18 个 blocker 各出现一次且四个新字段齐全。
- RG-08/RG-09、A5/A6、PMEmo、long58、14B 的位置符合本设计。
- 五份子计划保留全部原有编号技术任务，同时消除 package-level 错误全局 AND。
- placeholder/stale-order scan 为零，归档目录仍为 7 份文件，且无实现目录/Git metadata 因本轮产生。
