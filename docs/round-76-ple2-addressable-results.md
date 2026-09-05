# Round 76：PLE-2 可寻址外部记忆实证——real vs control 通过

> 日期：2026-09-05  
> 状态：PLE-2 第一阶段实证完成  
> 结论：以 n-gram 为离散地址、以文档为 value 的可寻址外部记忆，在代码/专名任务上远优于打乱控制，PLE 的主创新架构成立。

---

## 1. 本轮内容

- 新增 `scripts/run_ple2_addressable_eval.py`；
- 对 `AddressableNgramMemory` 做了首个端到端评测：
  - **real**：原始文档 token 顺序建索引；
  - **control**：同一批文档内部打乱 token 顺序建索引；
  - 两者共享相同文档集合/词表，但破坏 n-gram key → 文档 value 的原始词法关联。
- 评测指标：
  - continuation top-1/3/5 recall；
  - retrieval exact hit：检索到的文档中是否真的包含“当前 n-gram 后接真实下一 token”的转换；
  - retrieval any hit：检索到的文档中是否包含真实下一 token（弱信号）。

---

## 2. 主结果（每域 1000 个评测位置）

| Domain | Model | top1 | top3 | top5 | retrieval exact | retrieval any |
|---|---:|---:|---:|---:|---:|---:|
| wiki | real | 0.160 | 0.193 | 0.198 | 0.214 | 0.328 |
| wiki | control | 0.005 | 0.009 | 0.009 | 0.008 | 0.178 |
| code | real | 0.483 | 0.565 | 0.584 | 0.606 | 0.728 |
| code | control | 0.007 | 0.029 | 0.040 | 0.056 | 0.430 |

### Δ（real - control）

| Domain | top1 | top3 | top5 | retrieval exact | retrieval any |
|---|---:|---:|---:|---:|---:|
| wiki | +0.155 | +0.184 | +0.189 | +0.206 | +0.150 |
| code | +0.476 | +0.536 | +0.544 | +0.550 | +0.298 |

---

## 3. 关键解读

### 3.1 continuation 层面

- code real top-1 48.3%，control 仅 0.7%；
- wiki real top-1 16.0%，control 仅 0.5%；
- top-3/top-5 进一步拉开。

### 3.2 寻址检索层面

- 对 **code**，real 的 retrieval exact hit 达到 **60.6%**，control 仅 5.6%；
- 对 **wiki**，real 的 retrieval exact hit 21.4%，control 仅 0.8%；
- 这直接验证：
  > n-gram 地址能够真正指向“包含该局部转换的外部 value（文档/代码片段）”。

### 3.3 为什么这是 PLE 主创新

之前 PLE 被当成“语义知识预测器”失败。现在 PLE-2 把它变成：

```text
离散 n-gram key
    ↓ 精确查找
外部 value（文档/代码片段/实体）
    ↓ router/gate
非参数残差补充到 base/RAG 系统
```

检索 exact hit 的 real−control 差距证明：

> 外部 value 并不是噪声，而是与 n-gram 地址有真实、可审计、可复现的关联。

---

## 4. 局限

1. 当前 value 是“整篇文档”，不是语义 chunk/实体条目；
2. 未与 base model / RAG 做端到端融合；
3. 尚未跑 3-seed；
4. retrieval exact hit 只检查“包含该转换”，未检查检索 top 排序的精确性；
5. 控制组是文档内 shuffle，未做“同 key 置换 value”的更强控制。

---

## 5. 下一步

1. **接入 RAG**：
   - 将 `AddressableNgramMemory.retrieve` 作为 `HybridRetriever` 的第三个词法通道；
   - 与 BM25/dense 做 RRF 融合。
2. **实现多源凸 router**：
   - base + RAG + n-gram + teacher；
   - 解决 round-74 的 λ 校准问题。
3. **上升到语义 chunk / 实体 value**：
   - 用代码函数块、Wiki 段落、命名实体条目作为 value；
   - 做检索命中率的正确性评测。
4. **3-seed + 正式消融**。

---

## 6. 一句话

> PLE-2 的“可寻址外部记忆”不是纸面架构：真实 n-gram 地址能准确指向包含目标 continuation 的外部文档，real 远优于 control。
