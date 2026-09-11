# Round 153: 终极目标、技术债与本轮之后的开发计划

> 日期：2026-09-11
> 输入证据：Round 148-A/148-B（gold NLL、head 消融、oracle 探针、解冻 source）、
> Round 149（full-FT 崩塌）、Round 152（修正后的 LoRA 行、G0 4B、G3 磁盘基准）。
> 本文档回答：**我们到底在追什么、这一轮暴露出哪些债、接下来怎么走更稳**。
> 前序：`docs/round-149-systematic-roadmap-and-tech-debt.md`（G0–G4）、
> `docs/round-151-session-consolidation-and-handoff.md`（全量交接）、
> `docs/round-152-g3-engram-edge-storage-benchmark.md`（G3 实测）。

---

## 0. TL;DR

1. **目标需要一次精确化**：不是"证明 PLE 能提升 0.8B 通用性能"，而是
   **找出外部 n-gram 记忆相对于"同预算的纯参数/纯算力"真正占优的那一小块区域**，
   并把它做成端侧可部署的形态。前者是假设，后者才是可交付的结论。
2. **本轮最重要的科学进展是"排除法的收敛"**：reader 容量、gate 可学性、
   backbone 适应性（LoRA 已验证真实生效）三条路都已排除，
   因此"冻结跨模型表 + 嫁接"**没有正交信息可注入**这一解释，现在是唯一剩下的主假设。
   G0（4B，hidden 2560 = 源空间）是它最后的判决点。
3. **本轮最重要的工程发现是"科学诚信债"**：我们三次产出了**看起来正常、实际无效**
   的实验（gate override 全关、LoRA 权重未进 optimizer、NLL 在变但生成不变）。
   根因不是人手疏忽，而是**缺少不变量校验基础设施**。这是当前第一优先级的债。
4. **下一步的重心应该从"验证一个设定"转向"定位优势区间"**：
   比例扫描（取模剪表，今晚可跑）+ 表内容正交性审计（无 GPU）+
   共训练小表（真正的产品形态）。这三件事共同回答"这块记忆什么时候才值得存在"。

---

## 1. 终极目标（三层，且每层都可判定）

### 1.1 科学目标：从"能否提升"改为"何时占优"

原来的问法——*冻结的 Engram 表能否提升 0.8B 的通用性能*——已经被数据回答为
**否**（见 §2）。继续追问它只会得到更多负结果。真正有信息量的问题是：

> **在固定的"激活算力 + 内存预算"下，外部 n-gram 记忆在哪些任务签名上，
> 能超过同等预算的纯参数方案？**

这个问法有三个好处：

* 它承认 Engram 论文的结论本来就是 **iso-parameter / iso-FLOPs** 下的比较，
  而不是"记忆一定更好"；
* 它把我们的负结果变成**边界条件的一部分**（负结果同样是交付物）；
* 它直接对应端侧的真实约束（flash 便宜、RAM/算力贵）。

### 1.2 工程目标：0.8B 激活 + flash 常驻记忆的端侧形态

```text
0.8B 级激活算力
  + 51.2B FP8 PLE 表在 UFS/SD/NVMe（EngramDB Store-P，磁盘优先）
  + 256MB–2GB RAM 热缓存 + 异步预取
  + 轻量 backbone 适应（LoRA，已验证不会崩塌）
  + reader/gate 查询外部记忆
  + 行级热插拔知识补丁（秒级，不动 backbone 权重）
```

G3 已实测：NVMe 热缓存 462K rows/s（≈28.9K tokens/s，2048-token 预填充约 71 ms），
RSS ≈400 MB → **吞吐与内存预算通过**；USB SSD/SD 与移动端功耗/寿命仍未测。

### 1.3 平台目标：每层唯一真源 + 可复现

```text
hash/rowid   -> engram-peft / Qwen 官方语义 + golden 测试（已有）
storage      -> EngramDB Store-I/Store-P（已有）
reader/gate  -> qwen35_ple.reader（本仓库，已有）
backbone 适应 -> HF PEFT（本轮接入）
serving      -> EngramDB bundle（已有接口）
评测          -> locked standard suite + gold NLL + 不变量校验（待补）
```

---

## 2. 本轮（及之前）已经证伪的清单

写清楚"什么已经死了"，是防止团队（和我自己）反复回到同一口井的前提。

| 假设 | 证据 | 状态 |
|---|---|---|
| 冻结表 + 训 reader 可以注入知识 | real ≈ control（gold NLL 差 0.02 nat，1500 条） | **已证伪** |
| reader 容量/源空间对齐是瓶颈 | 解冻官方 source 投影：generation +1.5 点但 gold NLL −0.002 | **已证伪** |
| oracle 路由可被线性 gate 学到 | 5-fold AUC 0.504–0.635，leave-one-task-out 低于随机 | **已证伪** |
| 数据量不够 | 88 → 6000 条 SFT：standard open-QA 无增益 | **已证伪** |
| 全参 FT 能解锁 | 三臂同时崩塌成 BoolQ-only（TriviaQA/NQ ≈0.002） | **配方无效**，非 PLE 结论 |
| LoRA 共适应能解锁 | 本轮：LoRA 真实生效（+2.8～3.5 点），但 real vs control 差 **0.0000 EM / −0.018 nat** | **已证伪** |
| 3-gram/多阶是必需的 | keep2gram ≈ full，keep3gram 大幅变差；2-gram 主导 | 与本假设相反 |

**剩下的主假设**：这块表携带的主要是**局部延续统计**，对已在同一语料上预训练过的
backbone 而言大部分**冗余**；信息论上表的边际贡献极小（51.2B 字节 ≈ 1.6M token 级
的"逐字上下文"，而 backbone 权重是对 2–3T token 的有损压缩）。因此增量只可能出现在
**权重压不住的部分**：罕见实体与绑定、逐字延续、近重复文本。

**G0 是这条假设的最后判决点**：4B 的 hidden 2560 与 PLE 源空间精确对齐，
若连它都没有内容效应，则"跨模型冻结迁移"整条线关闭，重心转向共训练与重定位。

---

## 3. 本轮发现的债（按优先级，含根因）

### 3.1 第一优先级：科学诚信债（会产生"看起来对"的错结论）

| 债 | 本轮实例 | 根因 | 结构性修复 |
|---|---|---|---|
| 无效实验静默通过 | LoRA 权重没进 optimizer，500 步后 `lora_B` 全 0，adapter 加载后 logits 与基座 `max diff = 0.0`，gold NLL 与 frozen 基线 1500 条 bit-identical | "训练跑了没报错"被当作"训练生效了" | **不变量校验**：每个 adaptation arm 必须断言 `‖θ_trained − θ_base‖ > 0` 并把该值写进结果 JSON |
| 全关的对照被当成对照 | 早期 `--gate-override 0.0` 把贡献全关（后由 `--ple-off` 语义取代） | 对照臂的"是否真的关掉了"没有可观测证据 | 对照臂必须记录**实际贡献范数**（已有 `_last_reader_contribution` 钩子，未纳入断言） |
| 指标变了但行为没变 | `lora-nople` 生成指标变化，而 gold NLL 与 frozen 完全一致 | 未做"改动是否真的影响了输出分布"的自检 | 记录 trained/base 的 logits 或 token 级一致率作为健全性指标 |
| 关机依据不可信 | finisher 只看 DONE 标记，G0 失败仍关机；`chain/*.status` 的 `$?` 在 helper 之后读取，失败被记成 `rc=0` | 用"控制流的副产品"当"成功判据" | 退出码就地捕获；以**产物存在性 + 脚本自身 DONE** 为唯一判据（已修） |
| 训练/评测模板不一致 | G0 reader 用裸问题模板训练，评测臂用 `Question:…\nAnswer:` | 模板未纳入 config 记录 | 模板写入 config 并在两臂断言一致 |

### 3.2 第二优先级：统计与评测债

| 债 | 现状 | 修复 |
|---|---|---|
| 单 seed 决策 | 本轮 LoRA 行 1 seed；148-A 显示 seed 间差可达 0.05 nat | 机制性结论要 3 seed；单 seed 只允许作筛选 |
| 只有测试集、无验证集 | 反复在 standard 1500 上比较不同 recipe | 切出 locked validation，测试集只在定稿时碰 |
| 任务签名窄 | 只有 BoolQ/TriviaQA/NQ | 补 reasoning / 长上下文 / 噪声鲁棒性小集（PLE 的假设优势区） |
| 效应量下限未声明 | 出现过 SEM 0.0068 的"显著"差值 | 机制结论要求 paired SEM ≤0.01 nat；产品结论要求 ≥0.05 nat / 2 点 |
| 无可部署上限指标 | 未测"同激活算力纯参数方案" | 每个质量声明必须带参数量/FLOPs/内存三元组 |

### 3.3 第三优先级：工程与运维债

| 债 | 本轮实例 | 修复 |
|---|---|---|
| 无统一 runner | 队列靠 `run_roundNNN.sh` 复制粘贴，bug 随复制扩散（本轮 4 个队列脚本各带一份 skip/校验逻辑） | 统一 runner + `status.json` + 断点续跑 |
| 评测慢 | gold NLL 逐条跑到 28 分钟；本轮重写为分批后 3× 加速且数值等价（已验证） | 继续 batched eval 化；把等价性测试固化 |
| 无 artifact manifest | 结果散落 outputs/，靠时间戳考古；README 里 4B 的"NQ 0.748"其实是 3 段拼接的第 1 段 | manifest + 单一 summary JSON + 文档数字自动生成 |
| 远程代码漂移 | 手工 tar 同步，remote 落后于 local；remote git 报 dubious ownership | 单一 sync 脚本 + 启动前哈希校验 |
| 磁盘紧张 | 12–13G free，`/dev/shm` 行表重启即失 | retention 策略；LoRA 只存 adapter（已做） |
| 本地环境缺 torch | 本地 pytest 17 个 collection error，只能靠远程/CI | 本地加 torch extra 或 skip 标记 |
| 依赖未 pin | peft 临时 pip 安装；engramdb/engram-peft 版本未在实验流程固定 | 写进 requirements 并在启动时校验版本 |

---

## 4. 之后的开发计划：从"验证设定"转向"定位优势区间"

### 4.1 三个待答问题（取代原来的 G0–G4 线性 gate）

```text
Q1 判决：冻结跨模型迁移到底有没有正交信息？
   手段：G0（4B，运行中）+ 表内容正交性审计（②，无 GPU）
   出口：有 → 转 Q2a；无 → 关闭该方向，直接进 Q2b

Q2a 表示对齐：若 Q1 有信号，差距来自哪里？
   手段：G2 hot-row 表共适应（只训 SFT 语料触及的行，SparseAdam 5× LR）
   出口：real > control 显著 → 小规模共适应表成为产品路线

Q2b 优势区间：在什么规模/比例下外部记忆才占优？
   手段：③ 比例扫描（取模剪表，零重训，今晚可跑）
         + iso-parameter/iso-FLOPs 对照（同预算纯参数基线）
   出口：给出"质量 vs 记忆字节 vs 激活算力"的曲线与相变点

Q3 产品形态：最小可用记忆是多少、放在哪种存储上？
   手段：④ 2-gram-only / 量化 / 剪枝蒸馏 / tiny co-trained
         + G3 扩展到 USB SSD 与真实移动存储
   出口：端侧 bundle（参数量/内存/吞吐/热插拔时间四元组）
```

### 4.2 时间盒与停止规则

```text
今晚      ① 源空间上限（~15 min） + ③-lite 取模剪表比例扫描（数小时）
          → 产出第一版"比例-质量"曲线
本周      Q1 判决（G0 + ②）→ 明确关闭或保留跨模型迁移
          统一 runner + 不变量校验落地（§5）
          表内容正交性审计（合成绑定探针，区分"读不出"与"没有"）
下周      Q2a 或 Q2b 二选一深入；G3 扩到 USB SSD；补 reasoning/长上下文小集
停止规则  若 Q2b 显示在任何比例下都不优于同预算纯参数方案，
          则把 PLE 重新定位为"格式先验 + 热插拔知识补丁"，
          通用性能目标改由参数/数据侧解决，本项目转入平台化交付
```

### 4.3 必须固化的两条新纪律

1. **无效实验不可静默通过**：任何 adaptation/对照 arm 都要记录并断言
   *训练前后参数距离*、*对照贡献范数*、*trained/base 输出一致率*。
   任一项不满足即 fail，而不是产出结果。
2. **结论必须带三元组**：每个质量声明同时给出 *激活参数量 / FLOPs / 内存（含表）*。
   没有三元组的比较不许进文档——这是"iso-budget"思维的强制执行。

---

## 5. 从相似项目借鉴什么（按"我们缺的机制"组织，而非按项目罗列）

原则：**只借鉴机制，不复制实现；每层唯一真源；跨仓库 golden 对拍；只新增不改语义**
（与 `docs/integration-contract.md` 一致）。

| 我们缺的机制 | 从谁借鉴 | 具体接缝 | 明确不做什么 |
|---|---|---|---|
| 表写入/打包（我们只会读，不会写） | EngramDB Store-P / `engramdb-python` | 用现有 API 把"频率剪枝后的行"重新打包成新表 | 不自创行格式、不写第二个存储引擎 |
| 行级热插拔与精确寻址 | ortegaalfredo ngram-knowledge-injector | 行覆盖 + 版本化 patch 文件，经 EngramDB 更新路径 | 不引入独立 patch 格式 |
| hash/gate/短卷积的权威语义 | engram-peft（v1.2.7）+ Qwen 官方 modeling | 保持 pinned 版本 + 跨仓 golden 测试；只新增字段 | 不写第二套 hash/gate 实现 |
| 频率/碰撞统计（决定剪哪些行） | TN-gram / tensorized Engram 的碰撞与每字节质量视角 | 先在语料上统计 n-gram 频率桶，指导剪枝优先级 | 未过 golden 对拍前不替换 hash |
| 共训练配方（表与 backbone 同训） | DeepSeek Engram 论文（U 形分配、SparseAdam 5× LR、层位选择） | 把小表共训练做成 config 变体，复用同一 runner | 不盲目套用大模型规模与 4-gram 假设 |
| iso-budget 实验方法学 | Engram 论文的 iso-parameter/iso-FLOPs 协议 | 每个结论附三元组，纯参数基线同算力对照 | 不用"参数量不同"的比较下结论 |
| 轻量 backbone 适应 | HF PEFT（本轮已接入，adapter-only 落盘） | LoRA/DoRA 配置 + 合并；已验证不崩塌 | 不自研 LoRA |
| offload/预取与运行时集成 | llama.cpp-NLTM / qwen4exp、vLLM/SGLang PLE offload | 只借异步预取、pinned host memory、offload 调度思路 | 不 fork serving 引擎 |
| 端侧存储实测方法 | EngramDB 自带磁盘基准 | 复用我们的 `bench_engram_edge.py`，扩到 USB SSD | 不把 `/dev/shm` 结果当端侧结论 |
| 知识基线与对照 | RAG / 检索项目 | 把 RAG 作为 standard suite 里的对照臂 | 不把检索与 PLE 混为一谈 |

**冲突规避五条**（沿用并强化 round-149）：

1. 每层唯一真源，边界用 adapter；2. 跨仓 golden 对拍 + 版本 pin；
3. 共享评测协议，claim 不用仓库私有指标；4. 一个统一 trainer 覆盖 frozen/LoRA/全参/hot-row；
5. 存储永远 EngramDB，serving 永远是 bundle。

---

## 6. 稳定性清单（做完才算"更稳"）

```text
[ ] 统一 runner：一个入口、config 注册、status.json、断点续跑、失败即停
[ ] 不变量校验：参数距离 / 对照贡献范数 / trained-vs-base 输出一致率
[ ] artifact manifest：每次运行登记产物、指标、config 哈希、代码 commit
[ ] locked validation + locked test 两段式；测试集只在定稿时使用
[ ] 文档数字自动生成（禁止手抄指标；README 的 4B 数字需更正标注）
[ ] 远程同步脚本 + 启动前哈希校验；依赖版本写入 requirements
[ ] 队列模板化：新实验只写 config，不复制脚本
[ ] 关机/长任务：退出码就地捕获；成功判据=产物存在性（已修）
```

---

## 7. 与终极目标的距离（诚实评估）

* **已经拿到**：G3 端侧存储可行性（NVMe 行通过）、LoRA 温和适应可用（不崩塌 +3 点）、
  一套可复现的 standard 评测与 gold NLL、清晰的三重排除证据链。
* **还没拿到**：任何"外部记忆优于同预算参数"的正向证据。
* **因此当前最有价值的产出**是一个**边界条件结论**：*在什么规模、什么比例、
  什么任务签名下，外部 n-gram 记忆才值得存在*。这既服务端侧产品决策，
  也是这个方向能对领域做出的诚实贡献。

> 一句话：**别再把"证明它有用"当目标，把"测出它何时有用"当目标。**
