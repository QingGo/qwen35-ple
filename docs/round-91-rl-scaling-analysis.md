# Round 91：业界为什么采用 RL Scaling——合理性、不合理性与我们的决策

> 日期：2026-09-05  
> 状态：完成  
> 方法：20+ 轮搜索，围绕 RLVR/GRPO、post-training scaling law、reward hacking、on-policy 遗忘、小模型 RL、行业趋势等。  
> 结论：RL scaling 有真实技术理由，但不是万能；对 0.8B 低资源项目，应先走 OPD/RAG 蒸馏，RL 作为后期可验证任务的增强。

---

## 1. 业界为什么转向 RL Scaling

### 1.1 预训练数据/模型收敛，差异化转移到 post-training

- 基础模型能力接近平台期；
- 2025/2026 年越来越多观点认为“post-training 是新护城河”；
- 大型实验室开始把算力从纯预训练扩展到 rollout 采样、环境构建和 RL 训练。

### 1.2 可验证奖励让 RL 可以直接优化“结果正确性”

- RLVR：数学、代码、规则型任务可以自动判断对错；
- 这类信号不需要大量人工标注，可以无限生成任务；
- 模型可以通过自己的 rollout 探索，而不是只模仿固定数据。

### 1.3 长思维链 / test-time scaling 需要 RL 激励

- DeepSeek-R1、GRPO、Ring-Zero 等证明：
  - 通过奖励激励，模型能学会更长、更结构化的推理；
  - 推理长度/搜索深度可以在测试时进一步扩展；
  - RL 把“评测信号”变成“训练信号”。

### 1.4 On-policy 数据带来更好的分布匹配与遗忘控制

- RL 使用模型自身 rollout，减少 SFT 外部数据分布失配；
- 多项工作表明：
  - RL 比 SFT 更能保留已有电路/知识；
  - “Retaining by Doing”、RL’s Razor、RLFT preserve prior knowledge；
- 这是 RL scaling 被广泛接受的重要技术原因。

### 1.5 环境和数据工厂

- 行业开始建设“RL environments / data foundries”；
- 可以合成无限多任务、多智能体、工具调用、科学实验环境；
- 这使 RL 不只是“调参”，而是“造数据”。

### 1.6 开源工具链成熟

- GRPO、DAPO、RLVR、TRL、async GRPO、SLIME 等开源；
- 连 Qwen3.5-0.8B 也有公开 GRPO-Math 实验；
- 技术门槛下降，导致“RL scaling 成为默认路径”。

---

## 2. 合理性分析

### 2.1 对可验证任务确实有效
- 数学、代码、规则推理有自动 verifier，RL 能直接优化正确率；
- 这是 SFT 难以替代的：SFT 只能模仿已有解，RL 可以探索新解。

### 2.2 能超越 teacher
- 如果 teacher 只能提供有限数据，RL 可以通过搜索/试错发现超出 teacher 的路径；
- 这是“self-evolution”的重要来源。

### 2.3 与推理缩放兼容
- RL 训练出的长 CoT 在测试时可用更多采样/搜索；
- 推理算力可以继续 scaling，形成“训练+推理双 scaling”。

### 2.4 减少 distribution shift
- 因为训练分布来自当前策略，梯度路径更贴合最终部署分布；
- 这也是为什么 RL 比“外部数据 SFT”更不容易遗忘的机制解释之一。

---

## 3. 不合理 / 风险分析

### 3.1 信息论与采样效率极低
- 每个 trajectory 只给一个标量奖励，模型需要大量采样才能学到；
- 有观点认为这是“frontier RL 的极度低效”；
- 对中小团队，成本可能不可持续。

### 3.2 Reward Hacking / Goodhart
- RLVR 可以“骗 verifier”；
- Spurious Rewards Paradox：RLVR 可能激活记忆捷径，而不是真正推理；
- Outcome optimization 可能导致 reasoning shortcuts；
- 即使验证器看起来正确，也可能学会表面正确、内在错误。

### 3.3 Overoptimization scaling law
- 随着 KL 距离/训练步数增加，奖励模型分数继续上升，但真实性能先升后降；
- 存在“奖励过度优化”的 scaling law；
- 需要 KL 约束、Rényi 正则、对抗验证等控制。

### 3.4 不是所有任务都可验证
- 开放式问答、写作、创意、对话偏好难以自动判分；
- 用 reward model 又引入新的被攻击面；
- 所以 RL scaling 主要在 “verifiable domains” 最强。

### 3.5 大模型收益与小模型收益不同
- 已有证据显示 RLHF/RL 的收益随模型规模、数据、方法变化；
- 小模型 RL 能提升，但通常不如大模型显著；
- 0.8B 的 GRPO 数学实验存在，但最终能力仍受基础模型限制。

### 3.6 行业集中度与“护城河”问题
- RL 需要大量算力、环境和评测基础设施；
- “RL 是新护城河”也意味着小团队更难追赶；
- 对低资源项目而言，直接模仿大厂 RL scaling 不一定划算。

### 3.7 仍可能遗忘
- RL 比 SFT 遗忘更少，但并非不会遗忘；
- 尤其在持续 post-training 中，需要 retain 数据、replay、on-policy 目标选择等。

---

## 4. 数学视角

### 4.1 RL 目标

\[
\max_\theta
\mathbb E_{x\sim D,\,y\sim\pi_\theta}\big[R(y\mid x)\big]
-\beta\,\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}}).
\]

最优解：

\[
\pi^*(y|x)\propto \pi_{\text{ref}}(y|x)\exp\big(R(y|x)/\beta\big).
\]

- \(\beta\to0\)：集中到最高奖励；
- \(\beta\to\infty\)：回到参考；
- RL scaling 本质是“用采样算力在指数族上找奖励加权分布”。

### 4.2 采样复杂度

RL 每个 trajectory 只产生一个标量奖励，信息量低：

\[
\text{有效信息增益}\approx O(\log N)
\]

而 rollout 成本 \(O(N)\)。因此 RL scaling 是“用大量采样换少量高价值信号”，在可验证任务上值得，在不可验证任务上效率更低。

### 4.3 Overoptimization

设 reward model 与真实目标有误差 \(\epsilon\)，优化偏离参考分布越远，真实目标越可能先升后降。形式上：

\[
J_{\text{true}}(\pi)\approx J_{\text{proxy}}(\pi)-\lambda \cdot \text{dist}(\pi,\pi_{\text{ref}})
\]

因此需要 KL/正则约束，否则会出现 Goodhart。

### 4.4 On-policy 的优势

SFT 用外部分布 \(P_{\text{data}}\)，RL 用当前策略 \(\pi_\theta\)。梯度估计更少受到分布偏移影响：

\[
\nabla_\theta J_{\text{RL}}
=
\mathbb E_{y\sim\pi_\theta}\big[\nabla_\theta\log\pi_\theta(y|x)\,R(y|x)\big].
\]

OPD 也采用 \(\pi_\theta\) 采样，所以它在“on-policy 减少失配”这一点上与 RL 相似，但监督来自 teacher 而非奖励。

---

## 5. 对 qwen35-ple 0.8B 项目的建议

### 5.1 当前不要急于 RL scaling

理由：
- 我们没有大规模 verifier/环境/rollout 基础设施；
- 0.8B 模型通过 RL 获得的边际收益有限；
- 更容易 reward hacking；
- 资源应优先用于“确定有效”的路径。

### 5.2 推荐路线

1. **OPD / Purified OPSD**：
   - 用模型自身 rollout + 验证/过滤；
   - 比 RL 便宜，且比纯外部 SFT 分布更贴合；
2. **RAG self-distillation**：
   - 外部知识通过检索进入训练；
   - 我们已有 CAP-1 基础；
3. **MoRA / QLoRA**：
   - 参数高效，保留基础能力；
   - 已在 GTX1070 上验证可训练；
4. **PLE / RAG 外部记忆**：
   - 长尾知识不写进权重，减少遗忘；
5. **后期小规模 RLVR**：
   - 仅限数学/代码等可验证任务；
   - 小步、低 KL、加 verifier 审计；
   - 可参考 Qwen3.5-0.8B-GRPO-Math 等公开 recipe。

### 5.3 如果未来做 RL，必须监控
- reward hacking 与 spurious rewards；
- true performance 是否随 proxy reward 下降；
- 长 CoT 是否只是变长而非变好；
- 通用能力/知识是否退化；
- 是否有 verifier 可被学生模型“学会绕过”。

---

## 6. 关键论文/资源

- *DeepSeek-R1 / GRPO*
- *RLVR: Reinforcement Learning with Verifiable Rewards*
- *Ring-Zero: Scaling Zero RL to a Trillion Parameters*
- *Scaling Behaviors of LLM RL Post-Training (ACL 2026)*
- *Does RLHF Scale?*
- *Scaling Laws for Reward Model Overoptimization*
- *IsoCompute Playbook: Optimally Scaling Sampling Compute for LLM RL*
- *LLMs Gaming Verifiers: RLVR can Lead to Reward Hacking*
- *Spurious Rewards Paradox*
- *The Paradox of Outcome Optimization: A Causal Information-Theoretic Bound*
- *RL’s Razor / Retaining by Doing / Why RLFT Preserves Prior Knowledge*
- *Purified OPSD / Self-Distilled Reasoner / OPD*
- *The Extreme Inefficiency of RL for Frontier Models*
- *Post-Training in 2026: GRPO, DAPO, RLVR & Beyond*
- *RL Environments and Data Foundries*
- *Qwen3.5-0.8B-GRPO-Math*（公开小模型 RL 例子）

---

## 7. 一句话总结

> 业界采用 RL scaling 的合理性在于：可验证任务能直接优化结果、on-policy 数据减少分布失配、能激发长链推理和超 teacher 的自我进化；不合理性在于：采样成本极高、易 reward hacking、存在 overoptimization、不可验证任务仍需 reward model，且对 0.8B 小模型并非最优。当前项目应先用 OPD/Purified OPSD + RAG + MoRA/QLoRA，后期再在可验证任务上小规模 RLVR。
