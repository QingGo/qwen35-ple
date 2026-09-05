# Round 102：P1/P2 工具链落地——正式评测、Purified OPSD、CPU 基准

> 日期：2026-09-05  
> 状态：工具链完成，实际远程训练/评测待下一次 GPU 可用时执行  
> 内容：把剩余计划中的关键“能力层/产品层”组件做成可运行脚本，保证 CI 通过。

---

## 1. 正式评测工具链

### 1.1 `scripts/build_formal_benchmarks.py`

生成四类确定性本地正式评测：

| 类别 | 说明 |
|---|---|
| GSM8K-like | 算术文字题 |
| MATH-like | 代数/表达式求值 |
| HumanEval-like | Python 函数补全 |
| MBPP-like | 小编程任务 |

输出：

```text
data/formal-benchmarks/
  manifest.json
  gsm8k-like.jsonl
  math-like.jsonl
  humaneval-like.jsonl
  mbpp-like.jsonl
```

### 1.2 `scripts/run_formal_benchmark_eval.py`

对生成的正式评测文件跑模型评测：

- teacher-forced answer log-prob；
- first-token hit；
- 支持 base / adapter；
- 输出 `outputs/formal-benchmark-eval.json`。

---

## 2. Purified OPSD

### `scripts/run_purified_opsd.py`

实现本地化 Purified OPSD 数据准备：

- 输入候选 self-distill / RAG 数据；
- 对 math/arithmetic 做最终数值验证；
- 对 code 做 AST 语法/函数定义验证；
- 对 other 做长度过滤；
- 输出：
  ```text
  data/purified-opsd-train.jsonl
  data/purified-opsd-train-meta.json
  ```

后续可以直接把过滤后的数据喂给：

```bash
python scripts/run_lora_distill.py \
  --data data/purified-opsd-train.jsonl \
  --use-qlora / --use-mora
```

---

## 3. CPU 基准

### `scripts/bench_cpu_tok_s.py`

最小 CPU throughput 探针：

```bash
python scripts/bench_cpu_tok_s.py \
  --model data/models/Qwen3.5-0.8B \
  --new-tokens 32 \
  --device cpu
```

输出 tokens/sec，并对标 100 tok/s 目标。

---

## 4. 后续执行清单

1. 在 GPU 可用时：
   - 跑 `build_formal_benchmarks.py`；
   - 跑 `run_formal_benchmark_eval.py`（base / LoRA / QLoRA / MoRA）；
   - 用 `run_purified_opsd.py` 生成纯化数据；
   - 再训练 Purified OPSD adapter；
   - 对 adapter 重复正式评测。
2. 在产品化阶段：
   - 量化模型；
   - 跑 CPU benchmark；
   - 做 bundle / manifest / e2e。

---

## 5. CI

新增脚本已加入 CI lint 列表：

```text
build_formal_benchmarks.py
run_formal_benchmark_eval.py
run_purified_opsd.py
bench_cpu_tok_s.py
```

本地验证：

```text
90 passed, 9 skipped, 1 xfailed
ruff: all new scripts passed
```
