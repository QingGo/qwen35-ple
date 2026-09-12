# Round 165 预注册：修好 read-out，「零」还在不在？

状态：**规则已冻结，实验尚未运行。**

本文 §4.7 自己写明了一处未分离：

> the content-independence we measure under the standard recipe has two candidate
> explanations, and our experiments **separate their consequences without
> separating the explanations themselves**: either the window has nothing to give
> beyond the hidden state, or the read-out does not transmit what the window has.

这两者在信息论上**后果不同**：前者是界（定理，不可通过改 reader 修复），
后者是缺陷（可修复）。审稿人第 4 条要的正是分离它们。这一轮做这件事。

---

## 1. 这一轮问什么

已确立的事实（round 164，600 条，0.8B，第 2 层）：

| 量 | 值 |
|---|---|
| 注入向量有效维数 `PR(c)` | **1.0014** [1.0012, 1.0015] |
| 隐藏态 `PR(h)` | 1.480 [1.433, 1.535] |
| `cos(c_real, c_shuf)`（real vs 打乱行） | **0.9966** |
| `‖c‖` / `‖h‖` | 0.53 ± 0.33 / 1.33 ± 0.06 |
| `cos(c, h)` | 0.04 |

c 几乎是输入的常函数：换掉整张表的行内容，注入向量只转 ~5°。

而表本身**是有内容的**：同一批行上，显式 count trigram 达到 0.2535 top-1，
majority floor 为 0.0537。所以窗口里确实有可传的东西，问题是它有没有传出去。

> **本轮问题：把 read-out 换成一个不塌缩的版本，输出会不会开始对内容有反应？**

## 2. 设计

固定：主干（0.8B，冻结）、第 2 层、提示词与条目（`eval-600b.jsonl`，600 条）、
行表（`qwen38-rows`，128 shard）、reader 检查点（`round162-0.8B-nosft600/reader-wiki-seed0.pt`）、
PLE 权重尺度（0.0002）。

**只改一件事**：把 `e_t → c_t` 这个映射换成别的。所有变体**逐条范数匹配**到
production 读取器在该条上的 `‖c_t‖` —— 即「同样的注入预算，不同的 read-out」，
正是审稿人要求的「相同信息预算」。

### 2.1 五个 read-out 变体

| 变体 | 定义 | 预期 `PR(c)` |
|---|---|---|
| `production` | 训练出的 `OfficialSourceQwenReader` | 1.001（基线） |
| `centered` | `c − c̄`，再缩放回 `‖c‖`（`c̄` 为 600 条均值） | ≫ 1 |
| `random_proj` | `W · value_proj(e_t)`，`W` 固定随机高斯，缩放回 `‖c‖` | ~1.4 |
| `oracle_ungated` | `out_proj(value_proj(e_t))`（去掉 gate 与短卷积），缩放回 `‖c‖` | > 1 |
| `positive_control` | `λ · E[gold_first_token]`（gold 首 token 的输入嵌入），缩放回 `‖c‖` | —— |

> **实施前的一处改名（在产生任何结果之前）**：`oracle_ungated` 在本文档初稿里叫
> `oracle_linear`。载入检查点后发现 `out_proj` 不是一个 Linear，
> 而是一个 3 层 `Sequential(Linear, GELU, Linear)`，所以该变体把**这个模块整体**
> 作用在 `value_proj(e_t)` 上，仍按定义去掉 gate 与短卷积。名字随之改准。
> 变体的定义、终点、阈值与判定规则均未改动。

前四个测「内容能不能传」；第五个是**反空转对照**：如果连直接注入 gold token 的
嵌入都不能把输出推向正确答案，那么第 2 层的残差瓶颈本身就是死的，
本轮任何「没反应」都不能算作关于界的证据，实验判为无功效。

### 2.2 行内容条件

每个变体跑两个条件：
- `real`：`e_t` 来自真实行表；
- `shuf`：`e_t` 按 `run_phase0` control 模式同样地逐条置换。

`real` 与 `shuf` 的差就是「表内容」这一路信号。对 `production` 而言这个差已知极小
（cos 0.9966）；对 `centered`/`random_proj`/`oracle_linear` 而言理论上不小。

### 2.3 测量

注入只在**最后一个位置**（答案位置）。因为注意力是因果的，在该位置注入只影响
该位置及其后的 logits，所以「只在末位注入」与「全位置注入」对末位 logits **完全等价**，
并可省去 600 次多余前向。

对每个变体 × 条件，记录末位 logits，得到：

- `PR(c_real)`、`cos(c_real, c_shuf)`、`‖c‖` —— read-out 的内容依赖度；
- `TV(P_real, P_shuf)` —— 内容到达输出的量；
- `ΔNLL(gold) = NLL_shuf − NLL_real` —— **主终点**：正值表示真实行让 gold 更可能。

另记 `TV(P_variant_real, P_off)` 作为「注入有没有效果」的参照尺度（`P_off` 为 PLE 关闭）。

## 3. 判定规则（先冻结）

主终点为 `ΔNLL(gold)`，600 条配对，2000 次 bootstrap 的 95% 区间。

- **`UNDERPOWERED`** —— 若 `positive_control` 的 `TV(P_real, P_off)` 未能显著大于 0
  （bootstrap 区间下界 ≤ 0）。此时第 2 层注入通路本身无响应，本轮**不构成任何证据**，
  必须如实这样报告，不得把「没反应」读作支持界。
- **`READOUT_IS_THE_CONSTRAINT`** —— 若存在某个非塌缩变体
  （`centered` / `random_proj` / `oracle_linear`），其 `ΔNLL(gold)` 的 95% 区间
  **完全在 0 以上**，且其 `TV(P_real, P_shuf)` 显著大于 `production` 的。
  此时零结果的一部分来自 read-out 缺陷，论文的归因必须改写。
- **`BOUND_SURVIVES_REPAIRED_READOUT`** —— 若所有非塌缩变体的 `ΔNLL(gold)`
  95% 区间**都包含 0**，且 `positive_control` 通过。此时即使把 read-out 换成
  不塌缩的版本，内容也传不到输出，零结果不能归因于 read-out。

三个标签互斥，按上序判定。**不允许**在看到结果后改阈值、改变体定义或改终点。

## 4. 这一轮不做什么

- 不训练任何新 reader（本轮是**推理期**替换，不引入新的拟合自由度）。
- 不碰提示词模板、条目集、层位置。
- 不做生成（只测答案位置的下一 token 分布），因为 round 162 已证明生成被
  答案长度上限压住，不适合做灵敏度探针。

## 5. 与既有结论的关系

- 若判为 `BOUND_SURVIVES_REPAIRED_READOUT`：§4.7 那句「没有分离」可以改成
  「用一个不塌缩的读取器做了分离尝试，内容仍未到达输出」，Limitations 相应收紧。
- 若判为 `READOUT_IS_THE_CONSTRAINT`：§4.7、§6、§9 与摘要中「只 59%」那一段的
  归因都要改，并且 round 162 的零结果需要重新表述为「在我们这个 read-out 下」。
- 无论哪个结果，`positive_control` 的通过与否都要报，因为它决定本轮是否有功效。
