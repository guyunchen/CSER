# CSER

CSER 是一个连续语音情感回归项目，当前主线任务是 IEMOCAP 的 Valence/Arousal/Dominance 三维回归，输入为 80 维 Log-Mel 序列特征。

## 实验协议

IEMOCAP 实验默认采用严格的 Speaker-Independent LOSO 协议，避免 Speaker Leakage：

- IEMOCAP 共 5 个 session，每个 session 含 1 男 1 女两位说话人。
- 每个 fold 使用 1 个 session 作为 test，1 个 session 作为 val，其余 3 个 session 作为 train。
- 默认验证集策略为 `cyclic_next`：`Ses01` 测试时 `Ses02` 验证，依此循环。
- 训练只用 val 做 early stopping、scheduler 和 best checkpoint 选择。
- test 只在该 fold 的 best validation checkpoint 上最终评估一次。
- 最终论文结果使用 5 个 test fold 的均值和标准差。

固定 `train_path/test_path` 的旧式 IEMOCAP 训练入口默认已禁用，除非显式设置 `data.allow_speaker_dependent: true` 用作诊断。

## 数据流程

原始 IEMOCAP 数据是这三个 parquet shard：

```text
dataset/IEMOCAP/train-00000-of-00003.parquet
dataset/IEMOCAP/train-00001-of-00003.parquet
dataset/IEMOCAP/train-00002-of-00003.parquet
```

旧的 `dataset/IEMOCAP/final_data/train_v3.parquet` 和 `test_v3.parquet` 来自非说话人无关 split，不再作为新实验默认输入。

先从三个原始 shard 提取 Log-Mel 特征：

```bash
python extractor.py
```

默认输出：

```text
dataset/IEMOCAP/session_data/ses_1.parquet
dataset/IEMOCAP/session_data/ses_2.parquet
dataset/IEMOCAP/session_data/ses_3.parquet
dataset/IEMOCAP/session_data/ses_4.parquet
dataset/IEMOCAP/session_data/ses_5.parquet
```

新生成的特征 parquet 会包含 `file`、`session`、`speaker`、`valence`、`arousal`、`dominance` 和 `logmel_80`。训练时会合并这 5 个 session 文件，再按 session 做 LOSO train/val/test 组合。

可选：生成轻量划分清单，方便人工审计每条 utterance 在 5 个 fold 中的 split：

```bash
python scripts/write_iemocap_loso_manifest.py
```

输出：

```text
dataset/IEMOCAP/loso_splits.csv
```

这个 manifest 不复制大特征，只记录 `feature_shard`、`feature_row`、`session`、`speaker`、`fold_index` 和 `split`。

## 快速运行

运行 Lite-GLSER 主模型的 5 折 LOSO：

```bash
python train_loso.py --only-ids A3
```

烟雾测试单个 fold：

```bash
python train_loso.py --only-ids A3 --folds 0 --epochs 1 --max-batches 1 --num-workers 0
```

运行核心模型对比：

```bash
python experiments/run_experiments.py --config experiments/configs/core_models.yaml --skip-followups
```

汇总结果：

```bash
python experiments/summarize_results.py
```

LOSO 汇总表会写到：

```text
experiments/output/summaries/summary_loso.csv
```

## 项目结构

```text
configs/                  单模型旧配置与通用配置
data_utils/
  iemocap_protocol.py     IEMOCAP session/speaker 解析、LOSO fold、泄露校验
  reader_ctln.py          IEMOCAP Dataset 与 collate_fn
  metrics.py              CCC/MAE 指标
  normalization.py        Log-Mel 归一化与 padding mask
experiments/
  configs/                批量实验配置
  run_experiments.py      展开实验配置并运行 worker
  train_worker.py         单个 fold 的训练 worker
  summarize_results.py    fold 结果与 LOSO mean/std 汇总
models/
  unified_ser.py          统一 SER 模型，支持 CFC/S4/DA-LS4/Original-LS4
modules/                  模型组件
losses/                   CCC 多任务损失
liquid-s4-main/           作者 Liquid-S4 源码，仅 A4 Original-LS4 baseline 使用
training/
  data.py                 根据协议构建 train/val/test loaders
  engine.py               训练、验证、最终测试循环
  logging_utils.py        日志工具
scripts/                  独立评估与 manifest 脚本
train_loso.py             推荐的 IEMOCAP LOSO 入口
extractor.py              Log-Mel 特征提取，保留 file/session/speaker 元数据
```

## 主要实验配置

`experiments/configs/core_models.yaml`：

- `A0`: CFC identity
- `A1`: S4 identity
- `A2`: DA-LS4 identity
- `A3`: Lite-GLSER 主模型
- `A4`: Original-LS4

`experiments/configs/ablations.yaml`：

- 移除序列核心、移除注意力、关闭 DA-LS4 dynamic、修改 `p_order`/`d_state`、关闭 feature augmentation、关闭 label refine、CCC-only loss 等。

## 输出约定

批量实验输出集中在 `experiments/output/`：

```text
checkpoints/       每个 fold 的 best validation checkpoint
logs/              每个 fold 的训练日志
results/           每个 fold 的 JSON 结果
summaries/         summary_all.csv 与 summary_loso.csv
generated_configs/ 实际运行的展开配置
```

`results/*.json` 中的 `mean_ccc` 是该 fold 的 test CCC；`val_mean_ccc` 是用于选模型的 validation CCC。论文报告应使用 `summary_loso.csv` 中的 `*_mean` 和 `*_std`。
