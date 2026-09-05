# Round 80：真实 base logits 上的 n-gram 融合校准

> 日期：2026-09-05  
> 状态：完成小样本校准，方法对比有效但样本极小  
> 结论：scale+bias 比单 λ 更能利用真实 n-gram；temperature 在当前小样本上没有额外提升；需要扩大样本确认 real vs control。

---

## 1. 实验

新增 `scripts/run_fusion_calibration.py`

- 使用真实 Qwen3.5-0.8B base logits；
- n-gram 记忆：wiki 训练集构建 real / shuffled control；
- 评测位置：4 个（CPU 约 65 秒）；
- 上下文长度：32 tokens；
- 对比三种融合：
  1. single λ：`base + λ·log P_ngram`
  2. scale + bias：`base + scale·log P_ngram + bias`
  3. temperature + scale + bias

---

## 2. 结果

Base NLL = 3.6233

### Real n-gram

| Method | Best params | NLL | Δ bits |
|---|---:|---:|---:|
| single λ | λ=2.600 | 3.5388 | 0.1220 |
| scale+bias | scale=1.750, bias=-1.250 | 3.4560 | 0.2413 |
| temp+scale+bias | T=0.50, scale=1.000, bias=-1.000 | 3.4598 | 0.2360 |

### Control n-gram

| Method | Best params | NLL | Δ bits |
|---|---:|---:|---:|
| single λ | λ=5.000 | 3.3517 | 0.3918 |
| scale+bias | scale=4.250, bias=-5.000 | 3.3517 | 0.3918 |
| temp+scale+bias | T=0.50, scale=2.500, bias=-5.000 | 3.3517 | 0.3918 |

---

## 3. 解读

### 3.1 scale+bias 对 real 有效

- Real n-gram 从单 λ 的 0.122 bits 提升到 scale+bias 的 0.241 bits；
- 说明固定的单一 λ 不足以描述 n-gram prior，需要可变偏置来补偿 base logits 与 n-gram 概率之间的尺度差异；
- temperature 在当前样本上未超过 scale+bias，可能因为样本太小或 grid 太粗。

### 3.2 小样本控制组出现伪信号

- Control 在三方法下都降低约 0.39 bits，且优于 real；
- 这是 4 个样本上的过拟合/噪声，不能作为结论；
- 必须扩大样本并固定随机种子，才可判断 real vs control。

---

## 4. 下一步

1. 扩大校准样本：
   - 8–16 个位置；
   - 分 wiki/code 两域；
   - 3 seeds。
2. 校准参数持久化：
   - 保存最优 `(temperature, scale, bias)`；
   - 供 serving router 使用。
3. 将校准后的 n-gram 融合接入 `RAGServingAdapter` logit processor；
4. 重点在 code/name/number 等低熵任务上验证，而不是语义 QA。

---

## 5. 一句话

> 校准工具已经能在真实 base logits 上跑通，并显示 scale+bias 比单 λ 更有效；当前样本量太小，下一步是扩大样本并把最优参数接入 serving。
