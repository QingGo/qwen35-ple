# Round 28：机制验证第一批结果（CKA / Procrustes / reader / patching）

> 日期：2026-09-03
> 状态：Phase A 第一批机制证据已产出，继续做 case 分析与结构修正。
> 目标：回答“PLE e_t 与 Qwen hidden 是否可对齐、reader 是否真的在学习、注入是否因果影响行为”。

---

## 1. 本轮新增工具

| 脚本 | 作用 |
|---|---|
| `scripts/mechanism_alignment.py` | 计算 PLE e_t 与 Qwen hidden 的 CKA、Procrustes、kNN overlap、intrinsic dimension，并输出 reader 参数/gate 统计 |
| `scripts/mechanism_logit_patch.py` | 单次 forward 的 logit-level activation patching：no-reader / real / control / random / zero 五条件对比 |
| `scripts/mechanism_patching.py` | 逐 token 生成的 patching 脚本（较重，适合小样本生成观察） |

运行环境：

```bash
# WSL
cd /home/zeng/qwen35-ple
.venv/bin/python scripts/mechanism_alignment.py \
  --features data/ple-books-160k --reader outputs/reader-real-seed0.pt \
  --layers 1 8 16 23 --max-tokens 2048 --sample-size 256
.venv/bin/python scripts/mechanism_logit_patch.py \
  --tasks boolq --limit 8
```

---

## 2. 流形/语义对齐结果（2048 tokens，ple-books-160k）

### 2.1 PLE e_t vs Qwen hidden

| layer | CKA | Procrustes alignment | kNN overlap (k=10) | hidden PR |
|---|---:|---:|---:|---:|
| 1 | 0.222 | 0.051 | 0.079 | 77.9 |
| 8 | 0.151 | 0.034 | 0.075 | 41.2 |
| 16 | 0.192 | 0.023 | 0.084 | 58.1 |
| 23 | 0.151 | 0.010 | 0.068 | 37.5 |

- PLE intrinsic dimension（participation ratio，2048 样本）≈ **765.6**。
- Qwen hidden intrinsic dimension ≈ **37–78**。
- 随机 kNN overlap baseline ≈ **0.039**，实际约 **0.068–0.084**，只略高于随机。
- 说明：**两个空间的全局线性对齐很弱，局部邻域重叠也接近随机水平；PLE 是远高维的 n-gram 空间，Qwen hidden 是低维上下文空间。**
- 注意：CKA/Procrustes 对采样敏感；小样本 512 时 layer 8 CKA 为 0.198，因此应把数值看作“低对齐”的方向性证据，而非精确常数。

### 2.2 结论

当前 reader 更接近“一个可训练的投影/门控”，而不是已经完成流形对齐的稳定记忆读取器。  
如果要继续走 PLE 嫁接，可能需要：

- 更强的多跳/局部邻域对齐损失；
- 更低维记忆压缩或 top-k 选择；
- 在更合适的层级做注入；
- 重新评估当前 reader 是否真的把 PLE 当作“记忆”使用。

---

## 3. Reader 参数与 gate 统计

### 3.1 训练前后参数（official_source_qwen_v1）

| 参数 | trained norm | fresh norm | 说明 |
|---|---:|---:|---|
| query_bridge.weight | 58.73 | 58.42 | 高维随机投影，cosine-to-fresh ≈ 0.0003，方向已完全变化，但范数接近 |
| out_proj.weight | 4.25 | 0.00 | 从 zero-init 训练到非零，说明 out_proj 确实被训练 |
| key/value/norm/conv | frozen | frozen | 官方源侧保持冻结 |

real 与 control 的对比：

| 指标 | real | control |
|---|---:|---:|
| out_proj norm | 4.25 | 3.64 |
| gate mean (512 sample) | 0.450 | 0.130 |
| gate >0.5 fraction | 42.3% | 4.6% |
| contribution / hidden norm | 25.3% | 18.7% |

- control reader 的 gate 被训练得更低，对随机打乱 e_t 更“保守”。
- 但 control 仍会改变 BoolQ，说明即使低 gate 也能造成行为扰动。

### 3.2 2048 token real reader 前向统计

- gate mean = 0.405，std = 0.302；
- gate >0.1 = 89.5%，gate >0.5 = 36.3%；
- contribution norm / hidden norm ≈ 24.9%。

---

## 4. Activation / Logit-level patching 结果

### 4.1 完整 150 题（50 BoolQ + 50 NQ + 50 TriviaQA）

| 条件 | BoolQ logprob | BoolQ entropy | NQ logprob | NQ entropy | Trivia logprob | Trivia entropy | 总体 logprob | 总体 entropy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no-reader | -10.01 | 0.84 | -6.90 | 4.02 | -9.57 | 2.49 | -8.83 | 2.45 |
| real | **-7.62** | 2.23 | -6.80 | 4.69 | -9.39 | 2.83 | **-7.94** | 3.25 |
| control | -8.09 | 2.33 | **-6.76** | 4.79 | **-9.26** | 2.96 | -8.04 | 3.36 |
| random | -9.74 | 0.91 | -6.90 | 4.05 | -9.58 | 2.51 | -8.74 | 2.49 |
| zero | -10.01 | 0.84 | -6.90 | 4.02 | -9.57 | 2.49 | -8.83 | 2.45 |

### 4.2 real vs control 逐题差值

- 150 题中 real 更优 76 题，control 更优 74 题——接近抛硬币。
- 总体 mean(real - control) = **+0.10 logprob**，说明真实 e_t 有轻微优势，但远不足以称为稳定的语义对齐。
- BoolQ 上 real 优势最明显：-7.62 vs -8.09 (≈ +0.47 logprob)。
- NQ / Trivia 上 control 略优，说明真实顺序在小样本 logprob 上仍未形成稳定正效应。

### 4.3 解读

1. **real 和 control 都显著提高 next-token entropy**，而 random/zero 基本保持 no-reader 水平。
   - 说明 reader 会放行“PLE 形状的 e_t”（real 或 shuffled），但会抑制随机高斯向量。
   - 也就是说：**当前主要效应来自“是否注入 e_t 类向量”，而不是“e_t 的具体 token 顺序/语义内容”。**
2. **完整 150 题上 real 相对 control 只有极微弱优势**：
   - 总体 logprob +0.10；
   - 逐题胜负 76:74；
   - 仅在 BoolQ 上 real 优势较明显。
3. 这说明 M1 中 real 与 control 的 QA 差异尚不能归因于“真实 PLE 语义对齐”，更可能是局部格式/分布差异。

### 4.4 小样本早期结果（保留）

| 条件 | BoolQ 8 题 logprob | BoolQ 8 题 entropy | Trivia 10 题 logprob | Trivia 10 题 entropy |
|---|---:|---:|---:|---:|
| no-reader | -9.35 | 1.19 | -10.24 | 2.51 |
| real | -7.47 | 2.57 | -10.21 | 2.72 |
| control | -7.54 | 2.83 | -10.02 | 2.90 |
| random | -9.16 | 1.26 | -10.28 | 2.58 |
| zero | -9.35 | 1.19 | -10.24 | 2.51 |

---

## 5. 下一步

1. ✅ 已完成完整 150 题 logit-level patching，并已按 task 分层。
2. 加入 `zero-init reader` / `random reader` / `无 reader 但注入零` 等更高分辨率对照。
3. 做 layer/scale/gate 扫描，寻找注入强度最低但仍能提供真实信号的配置。
4. 设计 contrastive / neighbor-preserving / KL-to-no-reader loss，先在小语料上验证是否提高 CKA/kNN overlap。
5. 若提高后 real 仍不优于 control，则记录为负面机制证据，暂不进入 5M–20M/RL。

---

## 6. 产物

```text
outputs/mechanism-real-smoke.json
outputs/mechanism-control-smoke.json
outputs/mechanism-real-layers2048.json
outputs/mechanism-logit-patch-boolq8.json
outputs/mechanism-logit-patch-trivia10.json
outputs/mechanism-logit-patch-full150.json
```
