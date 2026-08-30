# 第十六轮调研：下一步实验方向（2026-08-30）

> 本轮仅做网络调研与方向分析，不改变现有实验结论。
> 目标：基于外部证据重新校准“冻结 PLE 能否给 0.8B 小模型带来增益”的实验路径。

---

## 1. 外部关键证据

### 1.1 XMemTransfer：训练量是一切的前提

- 论文/仓库：
  - [Cross-Model Memory Transfer via Target-Side Reader Adaptation (arXiv 2608.17050)](https://arxiv.org/html/2608.17050v2)
  - [OLAResearch/XMemTransfer GitHub](https://github.com/OLAResearch/XMemTransfer)
  - [Hugging Face Paper Page](https://huggingface.co/papers/2608.17050)
- 最关键的一句话：
  > Transferred memory is already competitive after only **5M target-side tokens** and essentially saturates by **20M**.
- 模型集合也印证：
  - [xmemtransfer-qwen35-4b-from-pythia160m-20m](https://huggingface.co/OLAResearchX/xmemtransfer-qwen35-4b-from-pythia160m-20m)
  - [xmemtransfer-mistral-from-llama2-r4-10m10m](https://huggingface.co/OLAResearchX/xmemtransfer-mistral-from-llama2-r4-10m10m)
  - [xmemtransfer-mistral-from-llama2-r4-20m20m](https://huggingface.co/OLAResearchX/xmemtransfer-mistral-from-llama2-r4-20m20m)
- **我们当前差距**：
  - 现有 reader 实验只用约 **46k tokens**。
  - XMemTransfer 的报告表明：5M 才开始“有竞争力”，20M 才基本饱和。
  - 我们比最低可比训练量少了约 **100 倍**，这几乎可以解释为什么 real 一直高于 no-reader baseline。

### 1.2 官方 Qwen PLE / DeepSeek Engram 的 reader 结构

- 官方 Qwen PLE 层关键结构（本仓 `src/qwen35_ple/official_ple_snapshot.py` 已固定）：
  - `key_proj`: `ple_embed_dim -> hc_count * hidden_size`
  - `value_proj`: `ple_embed_dim -> hidden_size`
  - 每个 branch 独立 RMSNorm
  - gate 非线性：
    ```text
    gate = (key_normed * query_normed).sum(-1) / sqrt(hidden_size)
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    gate = sigmoid(gate)
    ```
  - ShortConv：对 `hc_count * hidden_size` 的 gated value 做 depthwise causal conv，再与 gated value 相加
  - PLE 输出直接加到 hidden states 上
- DeepSeek Engram 的非官方实现（`engram-peft/src/engram_peft/layer.py`）已经对齐了几乎同一套数学：
  - `ContextAwareGating`
  - `ShortConv`
  - 3D hidden 时自动 expand 到 `hc_mult` 个 branch，最后 sum 回 3D
- **我们当前 reader 的差距**：
  - 当前用 raw `sigmoid(dot + gate_bias)`，没有官方 `abs*sqrt*sign` 非线性。
  - 当前 `ShortConv` 是简化版，默认 kernel=2/dilation=2，与官方/Engram 默认不一致。
  - 当前未使用零初始化 W_V 的“先静默，再学”策略。
  - 当前只有单层注入，没有 dual-layer / 多层 reader。
  - 官方 Qwen4 是 `hidden.repeat(1,1,hc_count)` 的四流结构；我们的 Qwen3.5 是单流，最接近的适配是“expand 到 4 个逻辑分支 -> gating -> ShortConv -> sum 回单流”，这正是 engram-peft 对 3D hidden 的做法。

### 1.3 Memory Grafting：离线冻结记忆的可行性

- [Memory Grafting: Scaling Language Model Pre-training via Offline Conditional Memory](https://ar5iv.labs.arxiv.org/html/2605.20948)
- 方法要点：
  - 离线构造冻结的 latent memory
  - 精确 n-gram + hash fallback
  - 轻量 projection / gating 嫁接到目标模型
- 这从侧面支持“把 PLE 当成离线冻结记忆，不重训大表”的方向是正确的。
- 但 Memory Grafting 的实验规模明显大于我们当前的 46k token / 40 step，因此我们当前不能据此判定方法失败。

### 1.4 Prometheus Mind：冻结模型会忽略信号

- [Prometheus Mind: Retrofitting Memory to Frozen Language Models](https://ar5iv.labs.arxiv.org/html/2601.15324)
- 教训：
  - 单纯往冻结模型里注入信号，模型可能“视而不见”。
  - 需要恰当的 stage-wise training / 注入位置 / 部分解冻。
- 含义：
  - 下一步不能只调 reader 超参，还应测试 **reader + LoRA** 或 **reader + 部分解冻深层**。

### 1.5 PWC / 公开排行榜

- WikiText-103 语言建模：
  - [Papers with Code WikiText-103](https://paperswithcode.github.io/sotabench-eval/wikitext103/)
- 现有记忆类工作排名很高：
  - Cross-model / XMemTransfer 方向在 WikiText-103 达到 **PPL 8.5**，排名 **#2**。
  - TriviaQA dual-layer reader 约 **72.5**。
- 说明“记忆迁移 + target-side reader”不是伪命题，而是当前公开榜单上成立的方向。

---

## 2. 对当前实验的重新解读

### 2.1 为什么 real > shuffled 但仍高于 baseline？

- 这是一个“弱阳性但没有超越基线”的状态：
  - real 确实比 shuffled 好，说明 e_t **内容**有信息；
  - 但 reader 的初始扰动、训练量、任务、注入方式可能都还不对，导致净收益被抵消。
- 潜在原因排序（按证据强度）：
  1. **训练量差 100 倍以上**（最强证据）；
  2. **reader 结构与官方不一致**（代码可证明）；
  3. **评测任务不对**：LM next-token 可能不是 PLE 的强项，知识 QA 更可能体现价值；
  4. **评测协议不严谨**：验证集可能被训练采样到，导致数字不可信；
  5. **冻结 backbone 过强**：Prometheus Mind 提示可能需要部分解耦。

### 2.2 哪些结论仍然可靠？

- 真实 PLE `e_t` 的线性可分性：72.7% vs 16.7%，这个正信号是可靠的。
- 真实 PLE > shuffled control：所有已跑组合一致，说明 e_t 内容不是噪声。
- 当前“无稳定增益”只能限制为：
  > 在 46k token / 40 step / 简化 reader / 单层 / 冻结 backbone / LM next-token 条件下，未见稳定增益。

不能外推为“PLE 嫁接失败”。

---

## 3. 下一步实验方向（建议优先级）

### P0：把评测做干净（成本最低，必须做）
- 固定 train / val 分割，val 绝不参与训练采样。
- 每个配置跑 3 个 seed，报告均值 ± 标准差。
- 同时记录：
  - no-reader baseline
  - real reader
  - shuffled control reader
  - 同参数量但 e_t 置零 / ffn-only 对照
- 记录训练/验证 loss 曲线，观察是否过拟合。

### P1：至少把训练量推到 1M–5M tokens（最关键变量）
- 证据：XMemTransfer 在 5M 才开始有竞争力，20M 饱和。
- 本机策略：
  - 先预计算更大语料的 e_t（本地只有 ~45k token 语料，需新增语料）。
  - 若 Intel Mac CPU 太慢，按用户已批准的选项走 SSH/WSL/GPU。
- 最低目标：
  - 1M token 的 quick signal（能区分“训练量不足”和“方法无效”）
  - 5M token 的 XMemTransfer 可比结果
- 不要再用 40 step / 46k token 下结论。

### P2：实现忠实版 Engram/Qwen reader（次关键）
- 直接对齐 `engram-peft` 的：
  - `ContextAwareGating`
  - `ShortConv`
- 或在本仓复制其数学：
  - `W_V` 零初始化或极小初始化
  - `hc_mult=4` 个 `W_K` + 独立 RMSNorm
  - 官方 gate 非线性
  - `ShortConv(kernel=4, dilation=max_ngram)`，残差在 ShortConv 内部
  - 3D hidden 时 expand -> 4 branch -> sum
- 建议小矩阵：
  - `hc_mult ∈ {1, 4}`
  - `zero_init ∈ {True, False}`
  - 注入层：浅层 / 深层 / dual-layer
  - real / shuffled / no-reader

### P3：换知识 QA 评测（可能改变结论）
- LM PPL 很可能不是 PLE 的优势任务。
- 建议加入：
  - TriviaQA（或子集）
  - Natural Questions（NQ-open）
  - BoolQ
  - OpenBookQA
  - SciQ
  - RTE
- 评测口径：
  - 0-shot / few-shot
  - exact match / accuracy
  - 同一批 seed 下 real / control / baseline 三线对比
- 若 QA 上 real > control 且 ≥ baseline，比 PPL 更有说服力。

### P4：Backbone 策略矩阵
- 冻结 only（当前）
- reader + LoRA
- reader + 最后 N 层解冻
- reader + 全量小学习率 CPT
- 记录可训练参数量和 loss 变化。

### P5：EngramDB live 闭环 + CPU 基准
- 仅当 P1–P4 出现稳定正增益后再做。
- 验证预计算 e_t == live Store 读取。
- CPU decode A/B：baseline vs PLE，目标 100 tok/s、PLE 尾差 ≤2%。

---

## 4. 建议的下一轮具体实验计划

### 第一批（本周末/下一轮可执行）
1. 新写 `run_ple_adapter_v2.py`：
   - 真 train/val split
   - 3 seeds
   - no-reader / real / shuffled
   - 忠实 `ContextAwareGating + ShortConv`
   - 可选 hc_mult / layers / zero_init
2. 用现有 46k token 先做 30 分钟 smoke：
   - 目的不是定论，而是验证代码/评测协议正确。
3. 同时扩大本地语料：
   - 从本地可访问文本/文档中尽量凑 0.5M–1M token；
   - 若不够，使用远程/GPU 和更大公开语料。

### 第二批（如果第一批通过）
4. 1M–5M token 训练。
5. 添加知识 QA 评测。
6. 增加 LoRA / 部分解冻。

### Go / No-Go 条件
- **Go**：在 ≥3 seeds 下，real 稳定超过 shuffled，且至少在 PPL 或 QA 之一超过 no-reader baseline。
- **No-Go**：在 5M token / 忠实 reader / 正确评测下仍无任何稳定正增益，则记录负结果并停止放大。

---

## 5. 关键引用

- [XMemTransfer arXiv HTML](https://arxiv.org/html/2608.17050v2)
- [XMemTransfer GitHub](https://github.com/OLAResearch/XMemTransfer)
- [XMemTransfer HF Collection](https://huggingface.co/collections/OLAResearchX/xmemtransfer)
- [XMemTransfer Qwen3.5-4B from Pythia-160M model](https://huggingface.co/OLAResearchX/xmemtransfer-qwen35-4b-from-pythia160m-20m)
- [DeepSeek Engram 官方仓库](https://github.com/deepseek-ai/Engram)
- [Conditional Memory via Scalable Lookup](https://huggingface.co/papers/2601.07372)
- [Memory Grafting](https://ar5iv.labs.arxiv.org/html/2605.20948)
- [Prometheus Mind](https://ar5iv.labs.arxiv.org/html/2601.15324)
- [Papers with Code WikiText-103](https://paperswithcode.github.io/sotabench-eval/wikitext103/)
