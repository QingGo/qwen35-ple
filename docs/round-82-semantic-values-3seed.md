# Round 82：语义 value + 3-seed 可寻址记忆结果

> 日期：2026-09-05  
> 状态：完成  
> 结论：把 value 从整篇文档提升为代码函数块 / 维基段落后，代码域 retrieval exact 平均约 69.8%，3-seed 稳定。

---

## 1. 新增

`scripts/run_ple2_semantic_values_3seed.py`

- Code value：AST 抽取函数/类定义块；
- Wiki value：文档级段落；
- 每个 value 作为独立外部记忆条目；
- 3 seeds（0/1/2），每个 seed 重新划分 train/eval；
- 每域每 seed 最多 300 个评测位置。

---

## 2. 结果

### 2.1 Code（函数块 value）

| seed | Cont@1 | Cont@3 | Ret exact | Ret any |
|---:|---:|---:|---:|---:|
| 0 | 0.550 | 0.667 | 0.657 | 0.750 |
| 1 | 0.640 | 0.710 | 0.723 | 0.773 |
| 2 | 0.627 | 0.710 | 0.713 | 0.783 |

**平均**：

- Cont@1 = 0.606 ± 0.049
- Cont@3 = 0.696 ± 0.025
- Ret exact = 0.698 ± 0.036
- Ret any = 0.769 ± 0.017

### 2.2 Wiki（段落 value）

| seed | Cont@1 | Cont@3 | Ret exact | Ret any |
|---:|---:|---:|---:|---:|
| 0 | 0.147 | 0.170 | 0.197 | 0.263 |
| 1 | 0.157 | 0.183 | 0.183 | 0.300 |
| 2 | 0.100 | 0.130 | 0.147 | 0.237 |

**平均**：

- Cont@1 = 0.134 ± 0.030
- Cont@3 = 0.161 ± 0.028
- Ret exact = 0.176 ± 0.026
- Ret any = 0.267 ± 0.032

---

## 3. 解读

- **Code 语义 value 明显更强**：函数块作为外部 value，地址命中率接近 70%；
- Wiki 段落的局部结构不如代码强，但仍远高于 0 基线；
- 3-seed 波动可控：
  - code ret exact std ≈ 0.036；
  - wiki ret exact std ≈ 0.026；
- 相比 round-76 整篇文档作为 value，语义块/函数块让 code retrieval exact 从约 60.6% 提升到约 69.8%（虽数据切分略不同，但趋势一致）。

---

## 4. 下一步

1. 把语义 value 接入 RAG serving：
   - 函数块/段落作为 `NgramKeyRetriever` 的 value；
2. 增加实体条目 value（从 QA/Wikidata 抽实体）；
3. 在真实 RAG 生成任务上对比：
   - 整篇文档 value vs 语义块 value；
4. 与校准后的 logit fusion 联合消融。

---

## 5. 一句话

> PLE-2 从“整篇文档可寻址”升级到“语义块/函数块可寻址”，并在 3-seed 下证明 code 域稳定达到约 70% 的精确寻址命中。
