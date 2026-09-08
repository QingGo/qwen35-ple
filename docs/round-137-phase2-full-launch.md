# Round 137：正式 Phase 2 全量矩阵启动记录

> 日期：2026-09-08  
> 远程：AutoDL RTX 4090 24GB（`connect.nmb1.seetacloud.com:19236`）  
> 状态：已启动，后台运行中

## 1. 远程资产准备

### 1.1 可写磁盘

远程实例的 `/autodl-pub/data` 是只读公共数据挂载，实际使用：

```text
/root/autodl-tmp/qwen35-ple      50GB 数据盘（代码、环境、模型、输出）
/dev/shm/qwen38-rows              60GB RAM 盘（真实 PLE Store-I 行表）
```

`/dev/shm` 中的行表在实例重启后会丢失，但正式实验期间足够。

### 1.2 Qwen3.5-0.8B

从 ModelScope 单文件下载：

```text
Qwen/Qwen3.5-0.8B-Base
-> /root/autodl-tmp/qwen35-ple/models/Qwen3.5-0.8B
```

实测 ModelScope 单文件下载约 10–17MB/s，优于 hf-mirror。

### 1.3 qwen38-rows

没有从本地上传 48GB，而是直接从 ModelScope 的 FP8 checkpoint 抽取：

```text
ModelScope: Qwen/Qwen3.8-Flash-Next-FP8
```

新增 `scripts/download_qwen38_fp8_rows.py`：

- 只下载包含 PLE n-gram embedding 的 33 个 checkpoint shard；
- 按 safetensors header 抽取 128 个 `ngram_embedding.shard_N.weight`；
- 输出为 EngramDB Store-I `shard_NNN.bin`；
- 与本地 `/Volumes/My Passport/qwen38-rows` 的 `shard_000.bin` / `shard_127.bin` 校验 sha256 一致；
- 总大小 51,200,245,760 bytes（128 × 400,001,920）。

抽取结果：

```text
/dev/shm/qwen38-rows/shard_000.bin ... shard_127.bin
```

### 1.4 Python 环境

```text
Python 3.12.3
torch 2.6.0+cu124
transformers 5.16.1
engramdb-python 0.2.12
```

路径：

```text
/root/autodl-tmp/qwen35-ple/venv/bin/python
```

## 2. 实验配置

```text
backbone : Qwen3.5-0.8B (frozen, float32)
PLE table: qwen38-rows Store-I live fetch
reader   : OfficialSourceQwenReader + 2-layer MLP bridge/out_proj
layer    : 8
scale    : 0.00019931793212890625
steps    : 500
seeds    : 0 1 2
modes    : real / control / no-reader
corpora  : PURE_WIKI PURE_FINEWEB PURE_STEM PURE_CODE FW_CODE FW_STEM
QA       : 150 items × 96 new tokens
QA batch : max 16 items, max 2048 padded tokens
```

## 3. 启动命令

```bash
cd /root/autodl-tmp/qwen35-ple/repo
export PYTHONPATH=/root/autodl-tmp/qwen35-ple/repo/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

bash scripts/run_phase1_matrix.sh \
  --pyenv /root/autodl-tmp/qwen35-ple/venv/bin/python \
  --rows-dir /dev/shm/qwen38-rows \
  --model-dir /root/autodl-tmp/qwen35-ple/models/qwen38_ple \
  --model /root/autodl-tmp/qwen35-ple/models/Qwen3.5-0.8B \
  --qa-file data/qa-expanded-150.json \
  --official-reader-path data/official_ple_reader.pt \
  --output-dir /root/autodl-tmp/qwen35-ple/outputs/phase2-full \
  --device cuda \
  --scale 0.00019931793212890625 \
  --steps 500 \
  --seeds "0 1 2" \
  --max-new 96 \
  --qa-batch-size 16 \
  --qa-batch-max-tokens 2048 \
  --corpora PURE_WIKI PURE_FINEWEB PURE_STEM PURE_CODE FW_CODE FW_STEM
```

日志与 PID：

```text
/root/autodl-tmp/qwen35-ple/logs/phase2-full.log
/root/autodl-tmp/qwen35-ple/logs/phase2-full.pid
```

输出：

```text
/root/autodl-tmp/qwen35-ple/outputs/phase2-full/phase1-<CORPUS>.json
```

## 4. 已做的关键修正

### 4.1 QA 批处理

原始 `_qa_exact_match` 逐题、逐步重算全序列，单题 96 token 约 11s。
新增批处理路径：

- 按 prompt 长度排序；
- 限制每批最多 16 题、2048 padded tokens；
- attention mask 处理 padding；
- 已验证 batch=4 + token budget 与 batch=1 在 real/control 下生成结果完全一致。

### 4.2 OOM 修正

首次全量运行在长 BoolQ 题上 OOM：

```text
torch.OutOfMemoryError: CUDA out of memory
```

修正：

- 引入 `--qa-batch-max-tokens`，按 token 预算动态分组；
- 设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；
- 每批结束后 `torch.cuda.empty_cache()`。

单模式 150 题 × 96 token 冒烟通过，耗时约 3.5 分钟。

## 5. 当前状态与后续

已启动全量矩阵，单 seed 三线约 16 分钟，预计全部 6 语料约 4–5 小时。

跑完后：

1. 下载 `outputs/phase2-full/phase1-*.json`；
2. 用 `scripts/summarize_phase1_matrix.py` 汇总 PPL / contains / extracted EM；
3. 用 `scripts/evaluate_generated_answers.py` 重评分；
4. 检查 real < control < no-reader 在 PPL 和任务级指标上是否稳定；
5. 通过后再进入 5M / 20M scaling 与 unseen-KB 正式实验。
