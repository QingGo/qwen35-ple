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

### 1.2 Phase A：新增 10k seeds 3、4（5 seeds 完）

已运行：

```text
seed3: ple-projector-10k-seed3.json
seed4: ple-projector-10k-seed4.json
```

### 1.3 LLM judge 已有数据补全

| 数据 | 样本 | mean judge score |
|---|---:|---:|
| HumanEval 20（base+bm25_ple） | 40 | 0.375 |
| TriviaQA 100 | 100 | 0.450 |

### 1.4 CPU 基线

```text
float32 CPU: 1.63 tok/s
```

> 这是未优化的真实基线；100 tok/s 仍是产品目标，距离尚远。

---

## 2. 进行中

- HumanEval 50（`outputs/humaneval-real-50-fast.json`）；
- TriviaQA 200（`outputs/triviaqa-real-200.json`）；
- 后续将跑：
  - pass@k 扩大样本；
  - LLM judge 全量 HumanEval 50 / TriviaQA 200；
  - sensitivity sweep（n-gram order / memory size / data size）；
  - token policy on/off 消融；
  - CPU 量化/吞吐补充。

---

## 3. 关键文件

```text
scripts/build_hf_artifact_release.py   # HF 打包上传
scripts/run_sensitivity_sweep.sh       # sensitivity sweep 入口
```
