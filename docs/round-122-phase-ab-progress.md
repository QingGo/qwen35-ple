# Round 122：Phase A 证据与 Phase B artifact 推进

> 日期：2026-09-08
> 状态：Phase A 主实验已启动，Phase B 公共 artifact 已发布。

---

## 1. 已完成

### 1.1 Phase B：HF artifact 已发布

- 仓库：https://huggingface.co/DefEki/qwen35-ple-auditable-ngram-memory
- 内容：
  - PLE Projector 10k seeds 0–4；
  - Purified OPSD MoRA adapters seeds 0–2；
  - 1k / 10k projector 数据集；
  - fusion / router / token-policy 配置；
  - 评测与训练脚本；
  - evaluation card / reproducibility manifest / evidence notes；
  - `artifact-manifest.json` + `SHA256SUMS`。
- 打包脚本：`scripts/build_hf_artifact_release.py`

### 1.2 Phase A：10k PLE Projector 5 seeds 完成

已运行 seed 3、4，并汇总 5 seeds：

```text
seeds 0-4
projector vs fixed NLL mean = +0.2249
bootstrap 95% CI = [0.1436, 0.3255]
positive seeds = 5 / 5
```

### 1.3 Phase A：HumanEval 50 完成

```text
base      pass@1 = 0.22 (11/50), rep = 0.0250
BM25+PLE  pass@1 = 0.10 (5/50),  rep = 0.0412
```

- base 通过：16,18,23,25,32,33,38,41,43,46,49；
- BM25+PLE 通过：0,10,27,35,41；
- 两者仅共同通过 41；
- BM25+PLE 仍恢复了 4 个 base 未通过题，但整体在 50 题上不如 base。

### 1.4 LLM judge 已有数据补全

| 数据 | 样本 | mean judge score |
|---|---:|---:|
| HumanEval 20（base+bm25_ple） | 40 | 0.375 |
| TriviaQA 100 | 100 | 0.450 |

### 1.4 CPU 基线

```text
float32 CPU: 1.63 tok/s
dynamic int8 CPU: 2.03 tok/s
```

> 这是未优化基线；100 tok/s 仍是产品目标，距离尚远。

---

## 2. 进行中

- TriviaQA 200（`outputs/triviaqa-real-200.json`）；
- 后续将跑（自动化链已挂起）：
  - pass@k 扩大样本；
  - LLM judge 全量 HumanEval 50 / TriviaQA 200；
  - sensitivity sweep（n-gram order / memory size）；
  - token policy on/off 消融（已有早期 open/policy 输出可复用）。

---

## 3. 关键文件

```text
scripts/build_hf_artifact_release.py   # HF 打包上传
scripts/run_sensitivity_sweep.sh       # sensitivity sweep 入口
```
