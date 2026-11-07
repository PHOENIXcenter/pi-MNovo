# π-MNovo 0.2.0：使用与代码导读

## 从哪一步开始？

输入是含前体 m/z、电荷和碎片峰的 MGF。先按样本选择一个统一检查点，显式传给 `--model-dir`：

| 样本 | v0.2.0 权重下载 |
| --- | --- |
| 单菌分离株 | [pi-MNovo-isolate.ckpt](https://github.com/PHOENIXcenter/pi-MNovo/releases/download/v0.2.0/pi-MNovo-isolate.ckpt) |
| 真实群落宏蛋白质组 | [pi-MNovo-metaproteome.ckpt](https://github.com/PHOENIXcenter/pi-MNovo/releases/download/v0.2.0/pi-MNovo-metaproteome.ckpt) |

完整推理需要 Linux x86-64、Python 3.10 和支持 BF16 的 NVIDIA GPU。原生 CTC 解码器位于 `wheels/`，是 Linux 轮子，不适用于 Windows 原生环境。请从仓库根目录执行：

```bash
conda env create -f environment.yml
conda activate pi-mnovo
python -m pip install -e . --no-deps
python -m pip check
pi-mnovo --help
```

以单菌谱图为例，设 `CHECKPOINT` 为本机文件的实际路径：

```bash
# 在权重下载目录运行；真实群落样本改用 ./pi-MNovo-metaproteome.ckpt
CHECKPOINT=./pi-MNovo-isolate.ckpt
pi-mnovo --input sample.mgf --model-dir "$CHECKPOINT" --output predictions.tsv --fdr off
```

`Score` 是最终序列的置信分数，不是 q 值。若开启 CCT 过滤，还需要与样本范围、模型和实现相匹配的已拟合 profile；仅更换检查点不能代替这一步。详细参数见 [FDR 工作流](../CCT_MODULE.md)。

## 阅读顺序 / Reading order

| 文件 | 中文说明 | English |
| --- | --- | --- |
| `MNovo/cli.py` | 从这里看输入、输出和预测入口。 | Command-line inputs and outputs. |
| `MNovo/input.py`、`MNovo/input_audit.py` | 输入谱图的读取与接收、拒绝计数。 | Spectrum ingestion and input accounting. |
| `MNovo/candidate_generation.py` | 检查精确质量约束候选如何加入候选池。 | Candidate merging for precursor-mass-constrained decoding. |
| `MNovo/ranker/`、`MNovo/selection.py` | 区分候选评分和最终选择。 | Candidate ranking and sequence selection. |
| `MNovo/sequence_score.py` | 对已经选定的序列评分。 | Confidence scoring for the selected sequence. |
| `MNovo/cct/cct_pipeline.py` | 可选的 PSM 及肽段级过滤，不负责选择预测序列。 | Optional post-prediction quality control. |

## 本地追溯

本仓库的源码可直接用于运行和扩展。若修改 CCT 特征定义或评分算法，原校准参数可能不再适用；应重新验证或校准后再报告 FDR。

**安装与依赖。** `environment.yml` 创建 Linux 环境，`requirements.txt` 固定推理依赖，`requirements-dev.txt` 加入测试和打包工具；`pyproject.toml` 的区间是包元数据，并非另一份经过验证的环境锁。原生轮子仅用于匹配的 Linux x86-64 / CPython 3.10 / PyTorch 环境。

**评估。** 对有 SEQ 标注的输入，使用 `pi-mnovo --model eval --input labelled.mgf --model-dir "$CHECKPOINT" --output predictions.tsv --metrics-output metrics.json`。单独核对输入审计：未被输入层接收的谱图和已接收但序列预测为空的记录不是同一类。

**训练。** 命令和配置示例见主 README 的 Evaluation and training 一节。训练的 `--checkpoint` 是骨干训练权重，不要把统一推理包误当成断点续训文件。本次整理不执行训练示例。

**核对权重。** 从下载清单取得对应文件的可信 SHA-256，在受支持环境运行 `python scripts/verify_release.py --checkpoint "$CHECKPOINT" --expected-sha256 "$SHA256"`。只加载来源可信的权重与索引；哈希匹配不能替代来源判断。

`frozen-rc2` 是原始源码快照。`git diff frozen-rc2 HEAD` 显示此后已提交的改动；`git diff` 显示尚未暂存的改动；`git log --oneline --all --decorate` 显示各版本。检查点留在仓库外，以 SHA-256 核实身份。
