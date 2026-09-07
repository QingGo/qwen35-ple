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

### 1.4 Phase A：TriviaQA 200 完成

```text
n=200
exact match = 0.005
mean repetition rate = 0.0076
```

### 1.5 Phase A：pass@k 扩大样本完成

```text
10 problems × 3 samples, 2 conditions
base      pass@k = 0.40 (4/10), rep = 0.0014
BM25+PLE  pass@k = 0.70 (7/10), rep = 0.0083
```

> 采样下 BM25+PLE 明显高于 base；与 greedy pass@1 的负向结果形成对比，说明
> 该记忆对多样性搜索模式有不同影响。

### 1.6 LLM judge 补全

| 数据 | 样本 | mean judge score |
|---|---:|---:|
| HumanEval 20（base+bm25_ple） | 40 | 0.375 |
| HumanEval 50（base+bm25_ple） | 100 | 0.400 |
| TriviaQA 100 | 100 | 0.450 |
| TriviaQA 200 | 200 | 0.465 |

> 并行 API 复跑得到 HumanEval 0.300 / TriviaQA 0.430，说明 judge 存在一定波动；论文中应报告多次运行或说明单次值。

### 1.7 Sensitivity / Ablation

- n-gram order sweep：`outputs/sens-order-{2,3,4,5}.json`
- memory bank size sweep：`outputs/sens-mem-c{40,120,300}-w{80,240,600}.json`
- token policy on/off：已有 `outputs/code-gen-projector-open.json`、
  `code-gen-projector-policy.json`、`code-gen-projector-policy2.json`

### 1.8 CPU 基线

```text
float32 CPU: 1.63 tok/s
dynamic int8 CPU: 2.03 tok/s
```

> 这是未优化基线；100 tok/s 仍是产品目标，距离尚远。

---

## 2. 当前状态

Phase A 主实验已全部跑完：
- HumanEval 50 / TriviaQA 200 / 10k 5 seeds / pass@k / full LLM judge / sensitivity / token-policy 复用；
- 自动化链已生成 HumanEval/TriviaQA judge 与 sensitivity 数据；
- 下一步：把结果整理进论文、重新上传完整 artifact bundle，并补最终 CI 验证。

---

## 3. 关键文件

```text
scripts/build_hf_artifact_release.py   # HF 打包上传
scripts/run_sensitivity_sweep.sh       # sensitivity sweep 入口
scripts/build_judge_input.py           # HumanEval judge 输入构建
```
