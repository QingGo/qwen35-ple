# Round 56：更紧的 PLE 记忆上界/下界与实现路径

> 日期：2026-09-04
> 状态：提出可计算的分层界和实现方案
> 目标：不再只停留在 \(I(Y;E|H)\) 这个宽泛必要条件，而是给出可测量、可比较、可指导实现的更紧上界和下界。

---

## 1. 动机

我们已经知道：

\[
\text{端到端增益}\le I(Y;E\mid H)
\]

但这是一个很宽的上界，因为它没有考虑：

- reader 的表达能力；
- backbone Jacobian 是否保留注入方向；
- 训练/优化是否逼近最优；
- 任务需要的信息是否真的在 PLE 中。

因此需要建立一组更紧的界，并给出计算/训练方法。

---

## 2. 界的分层

定义：

- \(H\)：某层 hidden；
- \(E\)：PLE 特征；
- \(E_\perp = E - \mathbb{E}[E\mid H]\)：去掉 H 线性可预测部分；
- \(R = Y - \mathbb{E}[Y\mid H]\)：基座不可预测的任务残差；
- \(Y\)：任务标签（可以是 next-token、答案 token、one-hot 等）；
- \(J = \partial \ell / \partial h\)：frozen backbone 从 hidden 到 logits 的 Jacobian；
- \(V = \mathrm{range}(J^\top)\)：hidden 注入能影响输出的“可见子空间”。

### 2.1 第 0 层：完整信息上界

\[
B_0
=
I(Y;E\mid H)
\]

这是硬上界，但通常不可达。

在 Gaussian/线性回归近似下：

\[
B_0 \approx -\frac12\log(1-\Delta R^2_{\text{full}})
\]

其中：

\[
\Delta R^2_{\text{full}}
=
\frac{\|P_{E_\perp}R\|^2}{\|Y\|^2}
\]

### 2.2 第 1 层：线性/低秩可实现上界

如果 reader 是线性、且只能使用前 \(r\) 个 PLS/CCA 方向，则：

\[
B_1(r)
=
-\frac12\log\left(1-\Delta R^2_r\right)
\]

其中：

\[
\Delta R^2_r
=
\frac{\|P_{E_{\perp,r}}R\|^2}{\|Y\|^2}
\]

\(E_{\perp,r}\) 是 \(E_\perp\) 在“与 R 最相关的前 r 个 PLS 方向”上的投影。

- \(r=0\)：0；
- \(r\) 增加：单调递增；
- \(r\to d\)：趋于 \(B_0\)。

这比 \(B_0\) 更紧，因为它反映了线性 reader 的实际容量。

### 2.3 第 2 层：backbone 可见子空间上界

hidden 注入后，输出只通过 \(J\) 受影响。近似有：

\[
I(Y;H+\Delta\mid H)
\lesssim
I(Y;P_V\Delta\mid H)
\]

因此对 hidden 注入，更紧的上界是：

\[
B_2
=
I(Y;P_V E_\perp\mid H)
\]

或者线性近似：

\[
\Delta R^2_{\text{visible}}
=
\frac{\|P_{P_V E_\perp}R\|^2}{\|Y\|^2}
\]

如果：

\[
B_2 \ll B_0
\]

说明：

> 即使 PLE 里有信息，从这一层注入也会被 frozen backbone 抹掉。

需要换层、换注入方式，或解冻 backbone。

### 2.4 第 3 层：logit-space 训练下界

最直接的绕过 \(J\) 的方法是不改 hidden，而是直接改 logits：

\[
\ell_{\text{mem}} = \ell_{\text{base}} + \Delta\ell(H,E)
\]

这等价于在输出端加一个 memory adapter。

设我们实际训练这个 logit adapter 后，测得：

\[
B_3
=
\mathbb{E}[\log P_{\text{base}}(Y) - \log P_{\text{mem}}(Y)]
\]

这是**可实现下界**：

- 如果我们连 logit-space 都做不到 real>control；
- 那么问题几乎可以确定在 PLE 信息本身，而不是 hidden 注入通道。

### 2.5 第 4 层：当前 hidden-injection 实测下界

\[
B_4
=
\text{当前 P1 系统实测的 real−control}
\]

这是当前实现的真实下界。

### 2.6 期望的大小关系

```text
B4 (当前 hidden 实现)
  ≤
B3 (logit-space 训练实现)
  ≤
B2 (backbone 可见上界)
  ≤
B1 (线性/低秩上界)
  ≤
B0 (完整信息上界)
```

实现目标是把这些量都测出来，从而定位瓶颈：

- \(B_3 \approx 0\) → PLE 信息不足，转 RAG/蒸馏；
- \(B_3>0\) 但 \(B_4\approx0\) → 问题在 hidden 注入通道，应改 logit-space 或解冻 backbone；
- \(B_3\approx B_2\approx B_0\) → 当前训练/接口已经接近理论上限，只差更多训练或更大容量。

---

## 3. 如何实现

### 3.1 数据收集

新建或扩展一个脚本，导出：

```text
H.npy   [N, d_model]
E.npy   [N, d_mem]
E_perp.npy
Y.npy   [N, d_y] 或 [N] one-hot/索引
```

建议：

- 从 `data/rare-kb-v1.json` 收集答案位置；
- 同时收集普通 LM 位置作对照；
- 每个样本保留：
  ```text
  is_rare, layer, source, item_id
  ```

### 3.2 计算 \(B_0\) 和 \(B_1(r)\)

用岭回归/PLS：

1. 中心化 H、E、Y；
2. 用岭回归求 \(E_\perp = E - \hat E(H)\)；
3. 用岭回归求 \(R = Y - \hat Y(H)\)；
4. 做 PLS/CCA between \(E_\perp\) and \(R\)；
5. 输出 \(\Delta R^2_{\text{full}}\)、\(\Delta R^2_r\)、以及 \(B_0\)、\(B_1(r)\) 曲线。

实现参考：

```python
# E_perp, R already centered
# For scalar/multi Y, use SVD of cross-covariance:
M = E_perp.T @ R
U, s, Vt = np.linalg.svd(M, full_matrices=False)
# rank-r projection directions are columns of U[:, :r]
E_r = E_perp @ U[:, :r]
coef = np.linalg.lstsq(E_r, R, rcond=None)[0]
R_hat = E_r @ coef
delta_r2 = (R_hat**2).sum() / (Y**2).sum()
```

对多输出 Y，可先做 CCA 或对 Y 做 PCA/one-hot 化。

### 3.3 计算 \(B_2\)：backbone 可见子空间

1. 在若干样本上，对 hidden 层做反向传播；
2. 每次取一个随机 logit 投影 \(w\)，计算：

   \[
   v = \nabla_h (w^\top \ell)
   \]

3. 收集 \(v\) 组成矩阵 \(V\)；
4. 对 \(V\) 做 SVD，得到前 \(k\) 个右奇异向量 \(U_V\)；
5. 计算：

   \[
   E_{\text{vis}} = U_V U_V^\top E_\perp
   \]

6. 用岭回归计算：

   \[
   \Delta R^2_{\text{visible}}
   \]

这可以直接回答：

> “如果 PLE 完全可用，当前 frozen backbone 在某一层最多能接收多少？”

### 3.4 测量 \(B_3\)：logit-space 下界

实现一个最小 logit adapter：

```python
class LogitMemoryHead(nn.Module):
    def __init__(self, d_mem, vocab_size, hidden=256):
        self.mlp = MLP(d_mem -> hidden -> vocab_size)
    def forward(self, e_perp):
        return self.mlp(e_perp)   # additive logits
```

训练目标：

```python
final_logits = base_logits + scale * memory_logits
loss = CE(final_logits[:, :-1], targets[:, 1:])
```

关键：

- 不加 hidden 注入；
- 只训练 memory head；
- real bank 和 control bank 各训练一遍；
- 用同一个评估 protocol 测 real−control。

如果 \(B_3\) 依然接近 0，则 PLE 信息不足这一结论会非常强。

### 3.5 测量 \(B_4\)：当前 P1

已有：

```text
scripts/eval_p1_memory.py
outputs/p1-memory-eval.json
outputs/p1-memory-eval-control-ckpt.json
```

### 3.6 输出报告

建议每个实验输出：

```text
{
  "B0_full": {...},
  "B1_rank_curve": {...},
  "B2_visible": {...},
  "B3_logit_adapter": {...},
  "B4_hidden_p1": {...},
  "gap_analysis": "..."
}
```

---

## 4. 如果这些界仍然不紧怎么办

可以继续收紧：

1. **非线性容量上界**：
   - 用固定宽度的 MLP 做 oracle，得到该容量下的可实现 \(\Delta R^2\)；
   - 用更大的 MLP 看是否饱和；
   - 这能区分“容量不够”还是“信息不够”。

2. **任务离散指标上界**：
   - 对分类/QA，用 Fano 不等式估算准确率上界：
     \[
     \text{accuracy} \le 1 - \frac{H(Y\mid H,E)}{\log |\mathcal Y|}
     \]
   - 与真实 EM/first-token 比较。

3. **有限样本界**：
   - 用 cross-validation 和置信区间；
   - 报告 \(\Delta R^2\) 的 lower confidence bound；
   - 防止小样本过拟合。

4. **信息分解**：
   - 分别测 \(I(Y_{\text{rare}};E|H)\) 与 \(I(Y_{\text{common}};E|H)\)；
   - 分别测 next-token 与 task 标签；
   - 指出 PLE 到底在哪类信息上“看起来有用”。

---

## 5. 预期结论

如果实施后得到：

```text
B0 ≈ 0.01 nats
B1 ≈ 0.003 nats
B2 ≈ 0.0005 nats
B3 ≈ 0.0001 nats
B4 ≈ 0.0001 nats
```

那么就能精确地说：

> PLE 的信息量本来就小，而 frozen backbone 只保留了其中很小一部分；连 logit-space 直接训练都无法放大它，因此应该转 RAG/蒸馏。

如果得到：

```text
B3 ≈ 0.01 but B4 ≈ 0
```

则应继续做 logit-space / 解冻 backbone，而不是放弃 PLE。
