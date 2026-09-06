# Round 114：论文补齐实验推进（TriviaQA / NGM / HumanEval / 10k 准备）

> 日期：2026-09-06
> 状态：部分完成，继续推进

---

## 1. 已完成

### 1.1 TriviaQA 100 条真实 exact-match

- 数据：`mandarjoshi/trivia_qa` RC，validation，100 条；
- 指标：
  - exact match = 0.0；
  - mean repetition rate = 0.0048；
- 说明：当前 0.8B base 模型在 TriviaQA 短答案上的 exact match 为 0，生成质量较差但真实。

### 1.2 NGM 基线

- 使用官方 `PioneerQyw/NGM` 的 NgramMemoryHook；
- 200 个局部续写点，layer 1/10，n-gram 2/3，output_scale 0.1；
- 结果：

| 条件 | NLL | hit |
|---|---:|---:|
| base | 2.5208 | 0.550 |
| NGM | 2.5206 | 0.550 |

> 官方 NGM 在当前 0.8B + 局部任务上几乎无变化，说明该强基线并未超过 PLE Projector。

### 1.3 生成质量 / 可复现工具

- `scripts/run_llm_judge.py`：LLM-as-judge 脚手架；
- `scripts/run_triviaqa_real_eval.py`：TriviaQA exact match + repetition；
- `scripts/run_ngm_baseline.py`：NGM 基线；
- `Dockerfile` + `docs/evaluation-card-paper.md`；
- HumanEval 脚本增加 repetition rate。

---

## 2. 进行中

### HumanEval 20 题

- 使用官方 `openai/openai_humaneval` 前 20 题；
- 条件：base / bm25 / base_ple / bm25_ple；
- 输出：`outputs/humaneval-real-20.json`；
- 预计运行时间较长（每 2 题约 15 分钟）。

---

## 3. 待运行

- 10k 数据 3 seed：
  - 脚本 `scripts/run_10k_projector_seeds.sh` 已准备好；
  - 使用 10k dataset，train 7000 / eval 300 / steps 100；
  - 等待 GPU 空闲后运行。
- pass@k：HumanEval 当前为 pass@1，扩大样本后可补 pass@k。
- LLM-as-judge：已完成工具，待选 judge 模型并运行。
