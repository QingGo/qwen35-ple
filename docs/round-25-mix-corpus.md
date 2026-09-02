# Round 25：1M 混合语料 M1–M5 构建与严格污染审计

> 日期：2026-09-02
> 状态：语料已构建并通过污染审计；WSL 三线 QA 尚未开跑

---

## 1. 本轮做了什么

1. 新增 `scripts/build_mix.py`：
   - 从 general / chat / wiki / cot / tool 五类来源按 token 比例采样；
   - 支持 Qwen tokenizer 直接输出 `tokens.npy`；
   - 输出 `corpus.txt` + `manifest.json`（路径、比例、token 数、SHA256）；
   - 支持 `--exclude-qa` 在记录级别过滤 QA 答案/问题，避免训练语料直接包含评测答案。
2. 新增 `scripts/audit_contamination.py`：
   - 对每一题检查答案、问题、QA 组合在语料中的精确子串和 n-gram 重叠；
   - 输出逐题 JSON + 严重级汇总。
3. 新增 `scripts/run_mix_batch.sh`：
   - 在 WSL 上一次跑 `M1–M5 × real/control/no-reader × 150 QA exact-match`。
4. 改进 `scripts/run_phase0.py`：
   - `_normalize_answer` 增加英文数字词归一化（`six` ↔ `6`）；
   - `_qa_exact_match` 增加逐题进度打印。

---

## 2. 语料来源

| 类别 | 来源 | 本地文件 | 说明 |
|---|---|---|---|
| general | 原有网页混合文本 | `data/wet-1m-one.txt` | 保留旧 1M 基线可比性 |
| chat | ModelScope `AI-ModelScope/alpaca-cleaned` | `data/sources/alpaca_data_cleaned.json` | 51k instruction/output |
| wiki | ModelScope `Salesforce/wikitext` | `data/sources/wikitext.jsonl` | WikiText-2 raw train |
| cot | ModelScope `nohurry/Opus-4.6-Reasoning-3000x-filtered` | `data/sources/distilled_corpus_400k_with_cot-filtered.jsonl` | 含 problem/thinking/solution |
| tool/agent | ModelScope `iic/MSAgent-Bench` dev | `data/sources/dev.jsonl` | 中文 agent 会话 |

`data/sources/` 与 `data/mixes/` 均被 gitignore，不进版本库。

---

## 3. M1–M5 比例与 token

所有实验 seed=0，目标 1M token，均通过 `--exclude-qa data/qa-expanded-150.json` 过滤。

| Mix | general | chat | wiki | cot | tool | 实际 token |
|---|---:|---:|---:|---:|---:|---:|
| M1 | 50% | 20% | 20% | 6% | 4% | 1,001,390 |
| M2 | 40% | 30% | 20% | 6% | 4% | 1,001,079 |
| M3 | 30% | 40% | 20% | 6% | 4% | 1,001,046 |
| M4 | 30% | 30% | 30% | 6% | 4% | 1,000,875 |
| M5 | 20% | 40% | 20% | 10% | 10% | 1,001,090 |

每个 mix 的 manifest 在：

```text
data/mixes/M1/manifest.json
...
data/mixes/M5/manifest.json
```

---

## 4. 污染审计结果

对 150 题 QA，使用 8-gram + 精确答案/问题/QA 检查：

| Mix | total | low | medium | high | critical |
|---|---:|---:|---:|---:|---:|
| M1 | 150 | 150 | 0 | 0 | 0 |
| M2 | 150 | 150 | 0 | 0 | 0 |
| M3 | 150 | 150 | 0 | 0 | 0 |
| M4 | 150 | 150 | 0 | 0 | 0 |
| M5 | 150 | 150 | 0 | 0 | 0 |

即：**过滤后没有任何 QA 答案短语、完整问题或 QA 组合出现在 1M 训练语料中**。
报告保存在 `outputs/contamination-M*.json`。

> 说明：当前过滤口径是“短语答案（≥2 词）、完整问题、QA 组合”。单字/常用词答案
> （如 `yes`、`no`、`six`、`water`）仍可能在自然文本中出现；这类命中在审计中不按
> critical 处理，后续分析“新做对且不在语料”时会用逐题语料命中进一步剔除。

---

## 5. 下一步

在 WSL 一次性跑 M1–M5 三线 QA：

```bash
bash scripts/run_mix_batch.sh \
  --mixes M1 M2 M3 M4 M5 \
  --seeds 0
```

跑完后用 `scripts/analyze_qa_lines.py` 分别对比：
- real vs no-reader
- real vs control
- 并检查新做对题目是否仍不在语料中（已有污染审计，可自动支持）。

## 6. 当前执行状态

- 已在本机完成 M1–M5 构建、污染审计和 smoke test。
- 已通过 Windows Scheduled Task 在 WSL 后台启动 **M1 三线 150 QA**：
  - 日志：`/home/zeng/mix-M1.log`
  - 产物：`/home/zeng/qwen35-ple/outputs/phase0-M1-seed0.json`
- 为避免 CPU 争抢，M2–M5 将在 M1 完成或用户确认后继续。
