# Round 134：Phase 1 语料与评测工具落地

> 日期：2026-09-08

## 1. 做了什么

按 Round 133 的 Phase 1 计划，先把“语料矩阵 + KB split + 评测协议”三个基础设施落地，
不改变核心实验方向（纯 PLE 嫁接）。

### 1.1 语料矩阵构建

新增脚本：`scripts/build_phase1_corpora.py`

生成 6 个 1M-token 语料：

| 名称 | 内容 | 配比 |
|---|---|---|
| PURE_WIKI | WikiText | 100% Wiki |
| PURE_FINEWEB | 本地 FineWeb token 流 | 100% FineWeb |
| PURE_STEM | 本地 STEM/数学 CoT QA | 100% STEM |
| PURE_CODE | 本地 Python 代码 | 100% Code |
| FW_CODE | FineWeb + Code | 70/30 |
| FW_STEM | FineWeb + STEM | 60/40 |

实现特点：

- 纯语料优先复用 `scripts/build_mix.py` 的可复现构建（记录 manifest / hash / 污染过滤）。
- FineWeb 走“预分词 token npy 直接切片”快速路径，避免重复 tokenize 大文件。
- 混合语料由纯语料 token 级拼接生成，比例在 token 层面可控。
- 所有构建均支持 `--exclude-qa data/qa-expanded-150.json` 污染过滤。

### 1.2 KB / QA split

新增脚本：`scripts/build_phase1_kb_split.py`

按 Round 132 的 contamination 原则生成：

```text
kb.train.jsonl
kb.eval.jsonl
qa.train.jsonl
qa.eval.jsonl
manifest.json
```

并输出双向审计：

```text
eval QA -> KB.train
eval QA -> KB.eval
train QA -> KB.eval
```

### 1.3 评测协议改进

新增模块：`src/qwen35_ple/eval/answers.py`

- `normalize_answer`：SQuAD 风格归一化 + 数字词展开；
- `contains_match`：允许“解释式输出”包含正确答案时判对；
- `extract_answer`：从长生成中提取简洁答案候选；
- `score_answer`：同时返回 strict exact / contains / extracted 指标。

新增脚本：`scripts/evaluate_generated_answers.py`

读取 `run_phase0.py --qa-exact-match` 输出的 `qa_exact.answers`，重新计算：

```text
exact
contains
extracted_exact
extracted_contains
```

并输出 JSON + Markdown 报告。

新增批处理入口：`scripts/run_phase1_matrix.sh`

- 遍历六个 Phase 1 语料；
- 每个语料跑 3 seeds × real/control/no-reader；
- 使用长生成（默认 96 tokens）与 `--qa` 对数似然；
- 输出可被 `evaluate_generated_answers.py` 重评分的 JSON。

## 2. 配套修改

- `scripts/build_mix.py`：新增 `code` / `stem` 两类语料源，支持纯领域语料构建。
- `.github/workflows/ci.yml`：lint 列表加入三个新脚本。
- `tests/test_answers.py`：新增归一化 / 提取 / lenient match 单测。

## 3. 已完成/未做

已完成：

- 本机已实际生成 6 个 1M-token 语料，位于 `data/phase1/`（git-ignored）：
  - PURE_WIKI / PURE_FINEWEB / PURE_STEM / PURE_CODE / FW_CODE / FW_STEM；
  - 每个纯语料均有 `tokens.npy` + `manifest.json`；
  - 混合语料为 token 级 70/30、60/40 拼接。
- 已生成示例 KB/QA split：`data/phase1/kb-wiki/`。

未做：

- 尚未跑 Phase 2 的 3-seed real/control/no-reader 矩阵（需要完整 GPU/CPU 训练环境）。
- 尚未做 unseen KB 的 PLE 读取实验。

## 4. 运行方式

```bash
# 构建 1M 语料矩阵（需要可加载 Qwen tokenizer 的环境）
python scripts/build_phase1_corpora.py \
  --tokenizer data/models/Qwen3.5-0.8B \
  --fineweb-tokens "/Volumes/My Passport/engramdb-data/p2-work/tokens/fineweb/fineweb.txt.u32.npy" \
  --output-root data/phase1

# KB split
python scripts/build_phase1_kb_split.py \
  --source data/sources/wikitext.jsonl \
  --qa data/qa-expanded-150.json \
  --output-dir data/phase1/kb-wiki

# 重新评分已有生成
python scripts/evaluate_generated_answers.py \
  --results outputs/phase0-M1-seed0.json \
  --output outputs/phase1-answer-report.json
```
