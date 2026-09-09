# Round 141：关机期间本地准备与重启检查清单

> 日期：2026-09-09  
> 背景：4090 实例关机，准备把数据盘扩到 ~100GB；期间在本地完成下一轮纯 PLE 诊断所需的代码与审计工具。

---

## 0. 当前状态

```text
远程：已关机 / 等待扩容
本地：已完成本轮 P0 准备，并保持 CI 绿色
数据：Phase 2 六个 corpus JSON 已归档
rows：/dev/shm 已丢，重启后从持久盘或 ModelScope 恢复
```

---

## 1. 本轮新增的本地能力

### 1.1 qwen38-rows 持久化与校验

| 脚本 | 作用 |
|---|---|
| `scripts/verify_qwen38_rows.py` | 校验 128 个 shard 的存在性、固定大小、可选 sha256 manifest |
| `scripts/ensure_qwen38_rows.sh` | 优先使用 `/dev/shm`；否则从持久盘复制；都没有则从 ModelScope 抽取 |
| `scripts/remote_manifest.py` | 记录版本、GPU、模型/tokenizer/rows 路径与校验结果 |
| `scripts/bootstrap_remote.sh` | 重启后一键检查持久资产、恢复 tokenizer、确保 rows、生成 manifest |

推荐持久化布局：

```text
/root/autodl-tmp/qwen35-ple/
  repo/
  venv/
  models/Qwen3.5-0.8B
  models/Qwen3.8-Flash-Next-FP8-tokenizer
  qwen38-rows/                  # 51.2GB 持久副本
  outputs/
  logs/
  tmp/                          # 抽取时的临时 checkpoint shard
```

如果训练/QA 需要 RAM 速度：

```text
/root/autodl-tmp/qwen35-ple/qwen38-rows
  -> /dev/shm/qwen38-rows       # ensure 脚本自动复制
```

### 1.2 QA 协议

| 变更 | 说明 |
|---|---|
| `src/qwen35_ple/eval/prompting.py` | instruction-style prompt：`Question: {question}\nAnswer:`；BoolQ 可单独覆盖 |
| `scripts/run_phase0.py --qa-prompt-template` | QA 生成 prompt 可配置 |
| `scripts/run_phase0.py --qa-boolq-prompt-template` | BoolQ 专用模板 |
| `scripts/run_phase1_matrix.sh` | 透传 `QA_PROMPT_TEMPLATE` / `QA_BOOLQ_PROMPT_TEMPLATE` |
| `scripts/check_phase2_gates.py` | 按预注册门禁检查 PPL + 任务级指标，并标记 severe regression |

### 1.3 断点续跑与备份

| 变更 | 说明 |
|---|---|
| `src/qwen35_ple/eval/resume.py` | 每个 `(mode, seed)` 写 partial JSON，并合并回最终结果 |
| `run_phase0.py --resume` | 跳过已完成的 `(mode, seed)` |
| `run_phase0.py --partial-dir` | 默认 `<output-stem>-partial/` |
| `run_phase0.py --backup-dir` | 默认 `<output-dir>/backup/` |
| `run_phase1_matrix.sh` | 默认透传 partial/backup；`--resume` 可用 |
| `scripts/pull_remote_results.sh` | 把远程结果 rsync 回本地 `artifacts/` |

中断粒度从“一个 corpus”降到“一个 `(mode, seed)`”。

### 1.4 小规模诊断矩阵

`scripts/run_phase2_diagnostic.sh` 默认：

```text
corpora : PURE_WIKI PURE_CODE FW_STEM
modes   : real control no-reader
seeds   : 0 1 2
steps   : 500
QA      : instruction-style prompt
reader  : --save-reader
resume  : on
```

---

## 2. 扩容后重启流程

### Step 1：确认数据盘

```bash
df -hT /root/autodl-tmp /dev/shm
```

目标：`/root/autodl-tmp` 约 100G。

### Step 2：一键 bootstrap

```bash
cd /root/autodl-tmp/qwen35-ple/repo
bash scripts/bootstrap_remote.sh --pull
```

它会：

1. 检查 repo/venv/model/tokenizer；
2. 缺失 tokenizer 文件时从 ModelScope 恢复；
3. 调 `ensure_qwen38_rows.sh` 确保 rows；
4. 写 `remote-manifest.json`；
5. 打印下一轮诊断命令。

### Step 3：确认 rows 与 manifest

```bash
ls -lh /dev/shm/qwen38-rows | head
python scripts/verify_qwen38_rows.py \
  --rows-dir /dev/shm/qwen38-rows \
  --sha256-manifest /dev/shm/qwen38-rows/manifest.json
```

### Step 4：运行诊断矩阵

```bash
cd /root/autodl-tmp/qwen35-ple/repo
export PYTHONPATH=$PWD/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

bash scripts/run_phase2_diagnostic.sh \
  --pyenv /root/autodl-tmp/qwen35-ple/venv/bin/python \
  --rows-dir /dev/shm/qwen38-rows \
  --model-dir /root/autodl-tmp/qwen35-ple/models/qwen38_ple \
  --model /root/autodl-tmp/qwen35-ple/models/Qwen3.5-0.8B \
  --output-dir /root/autodl-tmp/qwen35-ple/outputs/phase2-diagnostic
```

### Step 5：汇总与门禁

```bash
python scripts/summarize_phase1_matrix.py \
  --files /root/autodl-tmp/qwen35-ple/outputs/phase2-diagnostic/phase1-*.json \
  --protocol v2 \
  --output /root/autodl-tmp/qwen35-ple/outputs/phase2-diagnostic/summary-v2.json \
  --markdown /root/autodl-tmp/qwen35-ple/outputs/phase2-diagnostic/summary-v2.md

python scripts/check_phase2_gates.py \
  --files /root/autodl-tmp/qwen35-ple/outputs/phase2-diagnostic/phase1-*.json \
  --protocol v2 \
  --metric extracted_exact \
  --output /root/autodl-tmp/qwen35-ple/outputs/phase2-diagnostic/gates-v2.json \
  --markdown /root/autodl-tmp/qwen35-ple/outputs/phase2-diagnostic/gates-v2.md
```

---

## 3. 当前旧结果的门禁判定

用 `check_phase2_gates.py` 对旧 Phase 2 结果（v2 协议）离线检查：

```text
PPL gate: PASS
  real < control corpora: 5/6
  real < no-reader corpora: 6/6
  real < control seeds: 15/18
  real < no-reader seeds: 18/18

Task gate: FAIL
  candidates:
    FW_CODE / boolq
    PURE_CODE / boolq
    PURE_STEM / boolq
  severe regressions:
    PURE_FINEWEB / boolq
    PURE_WIKI / boolq

Overall: FAIL
```

结论：

- PPL 方向稳定；
- BoolQ 有候选增益；
- 但存在 PURE_FINEWEB / PURE_WIKI 的 severe regression；
- 不能进入 5M/20M。

这正是下一轮诊断矩阵要回答的问题：

> 用 instruction-style prompt + 保存 reader，real 是否能稳定超过 control，并消除 severe regression。

---

## 4. 重启后仍需在 GPU 上验证的事情

本地无法替代：

1. 新 prompt 下真实模型生成质量；
2. real vs control 的任务级差异；
3. reader checkpoint 保存/加载一致性；
4. 持久盘 vs `/dev/shm` 的吞吐差异；
5. 断点续跑在真实 OOM/重启下的行为；
6. 最终预注册门禁。

---

## 5. 重启检查清单

- [ ] 数据盘已扩到 ~100G；
- [ ] `bash scripts/bootstrap_remote.sh --pull` 成功；
- [ ] `remote-manifest.json` 显示 rows valid；
- [ ] `/dev/shm/qwen38-rows` 128 shards 完整；
- [ ] Qwen3.5 模型和 Qwen3.8 tokenizer 路径正确；
- [ ] 小规模 diagnostic matrix 启动；
- [ ] 每个 corpus/mode/seed 有 partial JSON 和 reader checkpoint；
- [ ] 结果拉回本地；
- [ ] `check_phase2_gates.py` 输出 overall pass/fail；
- [ ] 只有 overall pass 才进入 5M/20M。
