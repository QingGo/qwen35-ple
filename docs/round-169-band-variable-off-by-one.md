# Round 169:分频带变量错位一格(`context_counts` 数的是**前一个**三元组)

**状态**:实测结论,不是推断。发现于 2026-09-15 夜,在写阶梯预注册的机制段时。
**影响面**:B1 的分频带表(§5.1)、§5.2 的"150k 无上下文位置"、以及 round 168 §4 的分频预测
**全部band 错了变量**。**主判决(Δ、R、t)不受影响** —— 它们不分带。

---

## 1. 结论

对同一个 eval 流上的同一批对齐位置:

```
context_counts[t] == count(结束于 t-1 的三元组)     297,998 / 297,998   (100.0%)
context_counts[t] == count(结束于 t   的三元组)      23,910 / 211,671   ( 11.3%)
context_counts[t] == 0  <=>  count(结束于 t-1 的三元组) < 1     152,729 : 152,729,零反例
```

**位置 `t` 注入的行是"结束于 `t` 的三元组"那一行;而 `context_counts[t]` 数的是"结束于 `t-1` 的三元组"。**

## 2. 为什么"注入的是结束于 t 的三元组"是可证的

三条独立证据,任一条都指向同一结论:

1. **编码本身**。`trigram_codes` 是**单射**不是哈希:
   `codes[i] = (t[i]*V + t[i+1])*V + t[i+2]`,`V = 248_320`。所以 `codes[i]` 描述的是
   **结束于 `i+2`** 的三元组。eval 用 `seen_pos = flatnonzero(hit_all) + 2` 把码下标换成
   结束位置 —— 这正是这个 `+2` 的来历。
2. **快照自检**。eval 用 `fetch_e_t(all_rowids[probe])` 与 `E0[probe]` 对拍,
   `probe = seen_pos[pick]`,`max|diff| = 0.000e+00`。即 `E0` 在位置 `t` 的行**就是**
   分片表在位置 `t` 的行。分片表的行 id 来自 `rowids_from_tokens`,是 `(t-2, t-1, t)` 的函数。
   ⇒ `E[t]` = 结束于 `t` 的三元组的行。**这条自检从一开始就在跑,它证明的正是这件事。**
3. **因果结构**。`per_position_nll` 里 `logits = model(win).logits[0, :-1]`、`tgt = win[0, 1:]`,
   所以 `out[t]` 是"用位置 `t` 的状态预测 `t+1`"的 NLL。PLE 在位置 `t` 的记忆向量
   送进位置 `t` 的隐状态,其 key 是**含 `t` 的**上下文。⇒ 行必须是"结束于 `t`"。

反证也成立:若注入的真是"结束于 `t-1`"的行(即少一格),那么被预测的 `t+1` 与行所覆盖的
上下文之间就少了一个 token 的信息,而且 `t` 本身会**不在**行里 —— 若反过来错位成
"行覆盖了被预测的 token",冻结臂的 NLL 会塌陷成接近 0。实测 frozen 2.569 vs backbone 2.745,
**没有泄漏**,所以错位不在 eval 这一侧。

## 3. `context_counts` 为什么会错

`bench_ngram_reference.py:2171` 的契约是:

```python
def trigram_context_counts(stream, positions, raw3, model):
    """Raw TRAIN count of the trigram context that ends at ``position - 1``.
    ...
    so for a scored target at index j the relevant context is the trigram ending at j-1.
    """
    pos_ctx = positions - 1
    ck = context_keys_at(stream, pos_ctx, 2, model.vocab)
    return model.row_counts(raw3, idx, valid, stream[pos_ctx])
```

它的入参被**定义为目标位置 `j`**(要预测 `token[j]` 的那个位置),
此时"预测 `token[j]` 所用的行" = 结束于 `j-1` 的三元组 —— **函数内部自洽**。

但 margin 侧是把**对齐位置**数组直接传进去的,而 eval 侧把同一个数组当作**行位置 `t`** 用。
两边对同一个数组的语义假设差一格。函数没错,**调用契约错了**。

## 4. 后果:哪些说法要撤回

| 说法 | 状态 |
|---|---|
| §5.1 那张分频带表(7 个带、n、Δ、t) | **band 错变量**;数值是"前一个三元组频带"的,不是"注入行频带"的 |
| §5.2 "150,317 个位置自身三元组计数为 0,继承了别人投票的行" | **撤回**。`own ≥ 1` 对**所有**打分位置成立(命中 `codes_uniq` 就蕴含计数 ≥ 1),`trigram_codes` 又是单射,根本没有"继承"这回事 |
| round 168 §4 分频预测"被证伪" | **待重判** —— 它是拿错变量判的 |
| 主判决 Δ = −0.00272、R = 0.985、t = −5.22 | **不受影响**(不分带) |
| "越稀有伤害越大"的定性方向 | **未定**。相邻三元组的频次高度相关,方向**可能**存活,但这是要测的,不是可以假设的 |

**这正是本项目已经踩过两次的同一类坑**:一个看起来合理的机制故事,配上一张看起来合理的表,
而错的是那个把两者连起来的映射。区别只在于这次是在**写进文档之前**被计数本身抓住的。

## 5. 修正

不需要重训,也不需要重跑推理 —— 只要**逐位置记录**在。这就是
`round169_eval_rows.py` 新增 `<out>.deltas.npz` 的原因,它带三个键:

| 键 | 含义 |
|---|---|
| `snapshot_index` | **训练真正移动的那一行**的下标 |
| `trigram_code` | 决定这个位置有没有被寻址到的那个码 |
| `context_count` | 旧的(错位)变量,保留下来以便对照 |
| `rowid` | 分片表的行 id |

于是"这个位置自己那一行的训练计数"是精确的:
`own = trigram-train-count.npy[snapshot_index]`,由 `scripts/round169_reband.py` 算。

`own` 的性质与旧变量不同,这一点必须说清:**`own ≥ 1` 是构造性的**,所以正确的分带
**没有 0 带**,而旧变量 34% 的零是被造出来的。

## 6. 仍未做

1. **lr=1e-3 与 lr=1e-4 两个已知点的正确分带表**需要一个带逐位置记录的 re-eval
   (每个 ≈11 min GPU)。阶梯的点自带记录。
2. §4 的分频预测需要在新变量下重判,才能说"证伪"还是"存活"。
