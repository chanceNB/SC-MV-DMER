# 普通 DMER 全量数据为当前工作方向

2026-09-23 后续更新（DEC-0033）：用户已批准80/10/10和架构修订并要求实施。新划分已发布，1395/174/175，58首long_test；全2197窗口覆盖259997个有效标注点，每点恰好一次。已完成全1802首解码审计，确认201首容器登记时长与实际PCM时长存在差异，但没有监督点落在实际音频之外。新特征已按 `DECODED_PCM_CANONICAL_WINDOWS/v1` 独立重建，`deam-dmer-features-v2-decoded` 已完成并通过2,197个窗口的哈希绑定校验；正式声学训练仍需按阶段门槛推进。详细实际进度以 `DMER_REVISION_IMPLEMENTATION.md` 为准。下文为重建标签时的历史状态。

2026-09-23：用户明确要求暂停沿用 995 首版本推进正式实验，保留全部 1,744 段短音频及 58 首完整歌曲，按普通 DMER 口径重新处理。

`DEAM-DMER-FULL-v1` 已完成样本和标签重建。新 manifest 位于 `manifests/datasets/deam-dmer-full-v1.json`；审计位于 `reports/data/deam-dmer-full-v1-audit.json`。

`DEAM-PDMER-CLEAN-v3`、旧 feature/pseudo-label cache、旧 normalization 与 896/99 split 仅保留为历史，不用于下一次全量普通 DMER 正式实验。`configs/research/execution-hold.json` 与 RG-03 preflight 共同阻止继续执行当前 995 首训练入口。

正式实验尚未绑定完整 feature cache，未授权训练。保留所有音频的真实时长与全部原始音频引用，仅对 15 秒后的原始情感标注按有效行求均值。缺少 WorkerId 不再排除歌曲，短于 45 秒也不排除歌曲。

当前解码审计：`reports/data/deam-dmer-full-v1-decoded-audio-audit.json`，记录1802首、实际解码采样点与采样率，标签覆盖检查为0个越界点。协议仍用登记时长建立固定输入/输出网格；特征的 `audio_mask` 与有效覆盖按解码PCM计算。全2197窗口已完成绑定校验，其中51个窗口实际掩码更严格，但没有遮掉有效标签。

完整中文重建报告位于主工作目录的 `docs/research/DEAM_DMER_REBUILD_REPORT.md`。
