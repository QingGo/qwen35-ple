# 未执行对照登记册（Round 167 Stage 0.2，TD-4）

> 建立原因：同一份决定性对照清单被写下**至少三次**，执行**零次**。
> **项目的风险不是缺灵感，是灵感被写下然后跳过。**
>
> 出处：
> * `round-20` §3 —— dual-layer / multi-layer reader 未系统测试
> * `round-146` §4 —— ①gate selectivity ②dual-layer injection ③head/order ablation
>   ④hot-row adaptation 上界 ⑤memory/compute allocation
> * `session-log` :1253 —— 测 `hc_mult ∈ {1,4}`、zero-init、注入层、dual-layer
> * `session-log` :3030 —— P0：dual-layer injection（layer 2 + middle）、
>   V4.1 式 per-dim q/k gate、per-head/branch gate、special-token gate mask
> * `round-153` §4 —— Q1（G0 + 正交性审计）、Q2a（hot-row 共适应）、
>   Q2b（比例扫描 + iso-budget 参数基线）、Q3（2-gram-only / 量化 / tiny 共训练）
> * `round-157` §5 —— token 级 EM 接线、同预算纯参数基线、区间实验、计数通道

---

## 使用规则

1. **分栏必须诚实**：`已完成` / `未执行` / `已否决` 三态，不允许"部分"含糊带过。
2. **每个"未执行"必须带成本估计与优先级**，否则它会被无限期推迟。
3. **纪律**：低成本的决定性对照未跑完前，**不允许新增科学论断**。
   本条是 round-167 §3 TD-4 的修复，也是这份登记册存在的唯一理由。
4. **"已否决"必须附能推翻该否决的实验**（TD-5），否则否决不可复审。

---

## A. 未执行（按成本 × 决定性排序）

| # | 对照 | 出处 | 为什么决定性 | 成本 | 阶段 |
|---|---|---|---|---|---|
| U1 | **effective depth**（logit-lens KL / top-5 overlap / 残差余弦 / 跳层 / 残差擦除，PLE 开/关） | **从未出现在任何清单里** —— round-26 V185、round-27、roadmap 只有 `logit lens` 三个字，无规格 | Engram 的机制主张是"释放早期层"，实测收益在 BBH/ARC；**我们从没测过这一轴** | 前向为主，GPU 数小时 | Stage 1a |
| U2 | **read-out 初始化**（`zero_init_out ∈ {True, False}`） | `session-log:1253` | rank-collapse 理论预测零初始化 + 深度**必然**塌缩，而实测 `branch_sum` PR 2.676 → `c_t` PR 1.020 | 两次短训练 | Stage 1b（**Stage 0.3 已解除其不可达**） |
| U3 | **read-out 深度**（`out_mlp=False/True`，即 1 层 vs 2 层） | 同上 | 深度是隐式低秩偏置的自变量（ICML 2026） | 同上，与 U2 同批 | Stage 1b |
| U4 | **双层注入**（Engram: layer 2+15；V4.1: 1+14） | `round-20` §3、`round-146` §4.2、`session-log:3030` | 官方设计**两处**注入，我们一处在 layer 2；这是最直接的官方结构借用 | GPU 周级 | Stage 2.1 |
| U5 | **gate 选择性**（per-dim q/k、per-head/branch bias、special-token mask、熵/稀疏正则） | `round-146` §4.1、`session-log:3030` | round-146 已**诊断**出 gate 经 SFT 饱和常开 0.77–0.98，而 always-on 注入伤害 open QA（chat −0.0144） | GPU 周级 | Stage 2.2 |
| U6 | **两阶段课程第二阶段**（利用 → 遵从） | TokenMem 借鉴项 | warmup250 只做了 warmup；TokenMem 报告去掉第二阶段遵从度→0 | GPU 天级 | Stage 2.3 |
| U7 | **hot-row 表共适应**（只训 SFT 语料触及的行，SparseAdam 5×LR） | `round-153` §4 Q2a、`round-146` §4.4 | 区分"读不出"与"表里没有"的唯一手段 | GPU 天级 | Stage 3 的前哨 |
| U8 | **同预算纯参数基线**（iso-parameter / iso-FLOPs） | `round-153` §4 Q2b、`round-157` §4「最大的方法论缺口」 | 至今所有比较都是 *PLE vs 不注入 PLE*；公平对手是**同样 51 GiB 花在参数上** | GPU 天级 | Stage 4.2 |
| U9 | **联合训练小表**（可训练表 + 主干参与 + 长尾 LM 目标） | `round-157` §7.2/§8、`round-153` §4 | **原版三家真正验证过的配置，我们一次都没测过** | GPU 周级，最贵 | Stage 3 |
| U10 | **`hc_mult ∈ {1, 4}`** | `session-log:1253` | 分支数是读出容量的直接自变量 | GPU 天级 | Stage 2 |
| U11 | **比例扫描**（取模剪表，零重训） | `round-153` §4 Q2b | 给出"质量 vs 记忆字节"曲线，零训练成本 | **无 GPU** | Stage 4 |
| U12 | **表内容正交性审计**（合成绑定探针） | `round-153` §4 | 区分"读不出"与"没有" | 无 GPU | Stage 4 |

## B. 已完成（不要再重跑）

| # | 对照 | 结果 | 出处 |
|---|---|---|---|
| D1 | **head/order 消融**（2-gram vs 3-gram） | `keep2gram ≈ full`，`keep3gram` 大幅变差 → **2-gram 主导**，与"多阶必需"假设相反 | round-148，记于 `round-153` §2 |
| D2 | **G0（4B，hidden 2560 精确对齐源空间）** | 对齐解决不了结构问题 → 只有格式效应 | `round-152` |
| D3 | **LoRA 共适应** | LoRA 真实生效（+2.8～3.5 点），但 real vs control 差 **0.0000 EM / −0.018 nat** | `round-152`、`round-153` §2 |
| D4 | **解冻官方 source 投影** | generation +1.5 点但 gold NLL −0.002 → reader 容量不是瓶颈 | round-148-B |
| D5 | **oracle 路由可学性** | 5-fold AUC 0.504–0.635，leave-one-task-out 低于随机 | round-148 |
| D6 | **数据规模** | 88 → 6000 条 SFT：standard open-QA 无增益 | `round-153` §2 |
| D7 | **全参 FT** | 三臂同时崩塌成 BoolQ-only（TriviaQA/NQ ≈ 0.002）→ **配方无效**，非 PLE 结论 | round-149 |
| D8 | **更长键（k8/k16/longest17）** | 等存储下几乎全负，窗口越长输得越多 | `round-160` |
| D9 | **两阶段课程的 warmup 阶段** | warmup250 已实现并跑完 | `round-145`/`round-146` |
| D10 | **计数参照系** | 32 MB 4-gram → NLL 5.3368 / top-1 25.25%；天花板 4.94 | `round-158` |
| D11 | **长度混淆审计** | 散文标记随长度上升；0.8B 的 +0.6933 校正为 +0.4159 | `round-166` |
| D12 | **read-out 推理期修复** | production / centered / random_proj / oracle_ungated 四变体，端到端均无效；阳性对照 TV 0.295（注入路径是活的） | `round-165B` |
| D13 | **配置空间可扰动** | `zero_init_out` 由字面量变为 CLI 字段，默认不变 | **本 session Stage 0.3** |
| D14 | **指标滋扰审计框架** | 合成上可复现 round-166 混淆；真实臂上发现 0.8B 格式结论被长度完全解释、4B 不受影响 | **本 session Stage 0.1** |

## C. 已否决（附"能推翻该否决的实验"）

| # | 被否决项 | 否决理由 | 能推翻它的实验 |
|---|---|---|---|
| R1 | **专用 cross-attention 注入通道**（TokenMem） | "不照搬 cross-attention channel；我们仍是 residual PLE"（`round-142`） | 在**已诊断出 gate 饱和**之后重做 A/B：residual 加性 vs 专用通道，同课程、同数据、同预算。**若通道变体显著提升 selectivity，则原否决错误** |
| R2 | **Memory Grafting 的 "frozen memory" 读法** | 记成"读取接口，不改变 PLE 表"（`round-50`） | 实现其**真实核心**：把 grafting model 的隐状态存为 memory value（而非学 embedding），比较同预算下的 NLL。**若 representation-level 表显著更好，则原读法错误** |
| R3 | **再跑冻结嫁接的知识型 QA 消融** | `round-157` §1.1 的界说它必然为零（可证明） | 无需推翻：这是**唯一一条有证明背书的否决**。但它**只覆盖知识轴**，不覆盖 U1 的深度轴 |
| R4 | **从零联合预训练大表**（320M 行规模） | 资源不可行 | 不需要推翻；应标注为**资源受限而未验证**，而非被前述否证顺带否定 |

---

## 本登记册的即时结论

* **未执行清单里成本最低的三项是 U1、U11、U12**（U11/U12 甚至不需要 GPU），
  而它们恰好覆盖"机制轴"与"iso-budget 相图"两个一级子目标。
* **U2/U3 在 Stage 0.3 之前是不可达的**，不是"未做"——这条值得单独立债（TD-3a）。
* **R1/R2 是两条在事后看来站不住的否决**，其理由已在 `round-167` §3 TD-5 记录。
