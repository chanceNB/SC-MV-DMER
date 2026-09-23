# SC-MV-DMER：普通 DMER 全量研究

当前实施目录：`E:/SC-MV-DMER/.worktrees/rg03-formal-clean`。请在这里运行新脚本，不从早期 bootstrap 目录启动旧995首训练。

- 当前研究方案与实际进度：[架构修订执行记录](docs/research/DMER_REVISION_IMPLEMENTATION.md)
- 机器配置：[dmer-v2.json](configs/research/dmer-v2.json)
- 数据：1744短片＋58长曲；1395/174/175歌曲划分；长曲独立补充测试。
- 时间轴：45秒90输入，15–44.5秒60输出，长曲步长30秒。
- 旧995首配置与结果保存为历史，旧RG03执行锁保持有效。

## 本机运行

当前经验证的MERT/PyTorch解释器为 `E:/SC-MV-DMER/.worktrees/e0-minimum-bootstrap/.venv/Scripts/python.exe`。这是运行时复用，不表示应回到旧代码目录工作；运行前设置 `PYTHONPATH=src`。

```powershell
$env:PYTHONPATH='src'
$py='E:/SC-MV-DMER/.worktrees/e0-minimum-bootstrap/.venv/Scripts/python.exe'
& $py scripts/materialize_dmer_protocol.py --help
& $py scripts/materialize_dmer_features.py --help
& $py scripts/run_dmer_acoustic.py --help
```

新全量数据根：`E:/数据集/_pdmer_compat/processed/deam-dmer-full-v1`。
新协议根：同级 `deam-dmer-full-v1-protocol-v1`。
新特征根：同级 `deam-dmer-features-v2-decoded`；解码音频审计：`reports/data/deam-dmer-full-v1-decoded-audio-audit.json`。特征输入支持以实际解码 PCM 为准，协议时间轴和标签网格保持不变。只有完整提取并通过绑定检查后才能训练声学模型。

```powershell
& $py scripts/materialize_dmer_features.py `
  --manifest manifests/datasets/deam-dmer-full-v1.json `
  --data-root 'E:/数据集/_pdmer_compat' `
  --output-root 'E:/数据集/_pdmer_compat/processed/deam-dmer-features-v2-decoded' `
  --decoded-audit reports/data/deam-dmer-full-v1-decoded-audio-audit.json
```

`run_dmer_acoustic.py` 默认 development，只使用train/validation，不评价test。开发结果不标记为正式论文结果。新实验目录只写一次，完整保存配置、归一化、曲线、最佳／末轮检查点和验证预测。

## 测试

```powershell
$env:SC_MV_DMER_HISTORICAL_ARTIFACT_ROOT='E:/SC-MV-DMER-immutable/rg03-v3-bundle/historical'
$env:SC_MV_DMER_ARTIFACT_ROOT='E:/SC-MV-DMER-immutable/rg03-v3-bundle'
& $py -m pytest -q
```

旧MERT需要transformers4.24。tiny Qwen2测试用隔离包目录 `.superpowers/runtime/qwen51`（transformers4.51.3），前置到PYTHONPATH；不要直接升级旧MERT运行环境。正式Qwen7B、DSAML专用特征、人工解释审核与外部GPU是独立验收项，不由小模型测试自动认定完成。
