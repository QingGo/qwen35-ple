# Round 27：暂停 M2–M5，转向机制验证、流形对齐与数学工具调研

> 日期：2026-09-03
> 状态：M2–M5 已暂停；下一步以机制验证、case 分析和语义空间对齐研究为主

---

## 1. 为什么暂停 M2–M5

- 当前瓶颈不是“混比微调”；
- M1 已经显示：
  - `real < no-reader`
  - `control` 也退化
  - control 也能做对 Newton/Shakespeare 等
- 继续跑 M2–M5 对科学判定帮助有限；
- 应先验证：
  - reader 参数是否真的有效；
  - PLE 与 Qwen 语义空间是否可对齐；
  - BoolQ 退化机制。

---

## 2. 下一步机制验证与 case 分析

### 2.1 Case 维度

对 150 题建立错误分类表：

| 类型 | 说明 |
|---|---|
| 格式失败 | 答案被推迟、变成解释、生成重复 |
| 知识失败 | 不知道事实、给出错误实体 |
| BoolQ 极性反转 | 正确答案 yes 但输出 no |
| Passage 忽略 | 没有依据 passage，直接凭记忆回答 |
| PLE 干扰 | 真实/control 改变了原本正确的答案 |
| 无差异 | 三种模式表现一致 |

对每个 case，记录：

- real / control / no-reader 生成文本；
- 是否命中；
- PLE 实际检索的 n-gram；
- 答案 token 的 logit 变化；
- reader 输出 norm / gate。

### 2.2 机制实验

| 实验 | 目的 |
|---|---|
| reader 参数变化 | 判断是否真的训练 |
| activation patch | 判断 PLE 是否因果影响输出 |
| CKA / Procrustes / 局部邻域重叠 | 判断两个空间是否可对齐 |
| layer / scale / gate 扫描 | 找最佳注入位置和强度 |
| BoolQ logit lens | 定位错误发生的层 |

---

## 3. 流形 / 语义空间对齐思路

### 3.1 是否可以把两边看成两个流形

可以，而且很可能合理。

- PLE `e_t ∈ R^2560`：每个 token 的 n-gram 记忆向量，来自冻结大模型。
- Qwen hidden state：每个 token 的上下文表示，来自 0.8B 主干。

两个空间都满足常见假设：

> 高维空间中的语义表示主要分布在低维流形附近。

### 3.2 两边可能一致的部分

1. **语义聚类结构**  
   相似实体 / 主题 / 关系在两边都应当相近。

2. **局部邻域结构**  
   相近 token / n-gram 在两边应当保持近邻关系。

3. **部分线性子空间**  
   可能存在共享的低维语义方向：
   - 实体性
   - 主题
   - 情感/极性
   - 语言风格
   - 任务类别

### 3.3 两边可能不一致的部分

1. **内在维度不同**  
   PLE 可能更接近“检索索引”，Qwen 更接近“上下文预测”。

2. **度量尺度不同**  
   不同特征维度、不同 variance、不同 norm。

3. **非线性差异**  
   可能不是简单的旋转/缩放，而是局部曲率不同。

4. **n-gram 哈希噪声**  
   PLE 带了 hash 碰撞，Qwen 是 contextual representation。

5. **任务目标不同**  
   PLE 的目标是记忆检索，Qwen 的目标是语言建模。

6. **上下文依赖不同**  
   PLE e_t 可能更“词袋”式，Qwen hidden 更“上下文相关”。

---

## 4. 可以借鉴的数学工具

### 4.1 线性对齐

| 方法 | 适用场景 |
|---|---|
| Orthogonal Procrustes | 两个空间维度相同、寻求旋转/正交变换 |
| CCA / RCCA | 找共同低维子空间 |
| PLS | 有监督地找公共潜变量 |
| 岭回归 / MLP projection | 当前 reader 本质 |

### 4.2 非线性流形对齐

| 方法 | 适用场景 |
|---|---|
| Manifold Alignment (Laplacian) | 用局部图结构对齐两个流形 |
| Diffusion Maps | 保留局部扩散几何 |
| Manifold Warping | 时间/序列上的动态对齐 |
| 半监督流形对齐 | 有少量对应点，大量未对应点 |
| 局部 tangent space alignment | 处理曲率不一致 |

### 4.3 最优传输与度量空间对齐

| 方法 | 适用场景 |
|---|---|
| Gromov-Wasserstein | 对齐两个不同维度的度量空间 |
| 2-Sided Wasserstein Procrustes | 同时做置换与正交对齐 |
| Unbalanced OT | 处理噪声/缺失对应 |
| Wasserstein Barycenter | 找两个分布之间“中间表示” |

### 4.4 表示相似度与诊断

| 方法 | 适用场景 |
|---|---|
| CKA | 衡量两个表示是否在功能上相似 |
| SVCCA / PWCCA | 找公共主方向 |
| Procrustes residual | 测量可线性对齐程度 |
| kNN overlap | 衡量局部邻域是否一致 |
| Intrinsic dimension estimation | 比较两个空间的复杂度 |
| Persistent homology | 比较拓扑结构 |

### 4.5 学习目标设计

| 方法 | 适合作为训练 loss |
|---|---|
| Contrastive / InfoNCE | 拉近对应表示、推远无关表示 |
| MMD / HSIC | 对齐两个分布 |
| 邻域保持损失 | 保持局部流形结构 |
| Gromov-Wasserstein loss | 对齐 pairwise metric structure |
| KL to no-reader | 防止注入破坏原有能力 |
| Task loss / QA loss | 直接优化下游行为 |

---

## 5. 对当前 reader 的具体建议

### 5.1 诊断优先

先不做大改，先回答：

```text
1. PLE e_t 与 Qwen hidden 的 CKA 是多少？
2. Procrustes 残差是多少？
3. kNN overlap 是多少？
4. 两者内在维度差多少？
```

如果 CKA 很低、neighbor overlap 很低，说明当前 reader 只是“硬投影”，没有真正做流形对齐。

### 5.2 训练 loss 可以增加什么

当前主要是：

```text
next-token prediction loss
```

建议增加：

```text
L = L_lm
  + λ1 * L_contrastive(PLE, Qwen_after_reader)
  + λ2 * L_neighbor(PLE_neighbors, Qwen_neighbors)
  + λ3 * L_kl(reader_on, reader_off)   # 防止破坏基座
  + λ4 * L_task(BoolQ/QA)
```

### 5.3 网络结构可以考虑

- 在 reader 后加 per-layer gate；
- 增加 top-k memory selection；
- 增加 residual connection / gating 让模型可以选择“用或不用”PLE；
- 增加 projection 到 Qwen hidden 的 tangent space；
- 增加短路路径，让 BoolQ 等简单任务可以绕开 PLE。

### 5.4 优化器可以考虑

| 方向 | 理由 |
|---|---|
| 分开 reader / gate / backbone 学习率 | 避免破坏基座 |
| 大 warmup + 小最终 lr | 稳定小模型训练 |
| weight decay 只作用 reader | 防止参数无意义增大 |
| gradient clipping | 防止 PLE 注入导致梯度爆炸 |
| 尝试 AdamW / Muon | 论文中的优化器 Scaling Law 思路 |

---

## 6. 优先实验清单

1. 测 PLE 与 Qwen hidden 的 CKA/Procrustes/neighbor overlap。
2. 做 reader 参数有效性和 activation patching。
3. 做 BoolQ 错误分类和 logit lens。
4. 做 layer/scale/gate 扫描。
5. 如果以上显示“可对齐”，再设计 manifold alignment loss。
6. 如果显示“不可对齐”，考虑换投影结构或重新审视 PLE 适配方式。
