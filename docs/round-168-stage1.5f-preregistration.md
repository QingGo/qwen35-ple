# Round 168 Stage 1.5f 预注册：寻址步骤销毁了什么

> 日期：2026-09-13
> 前置：`docs/round-168-ultimate-goal-tech-debt-and-plan-v3.md`（能力阶梯）、
> `docs/round-168-stage1.5c-preregistration.md`（天花板）、`round-161`（阶③ 表探针）
> **本文在任何真实语料的寻址数值被计算之前写成。** 实现见
> `src/qwen35_ple/addressing.py`、`scripts/round168_addressing_resolution.py`。

---

## 0. 为什么单列这一级

能力阶梯有四级，而**第②级（寻址：从 trigram 到 16 个行 id）从未被单独隔离**：

```text
① trigram 天花板   I(Y; w_t | h_t)              round-158 代理：渐近 NLL 4.94
② 寻址             trigram -> 16 个行 id          ← 本文
③ 行内容           冻结行上探针能回收多少          round-161：约 59% 的 trigram top-1
④ 读出             生产 reader                    round-167：约 0
```

**自然猜测是②必然损失巨大**：bigram 键空间 $V^2 = 6.15\times10^{10}$，
trigram 键空间 $V^3 = 1.53\times10^{16}$，而表总共只有
**3.200014e8 行**（16 个头各约 2.0e7；8 个 bigram 头 + 8 个 trigram 头，实测自 `real_spec()`）。

**本轮的第一次手测暗示相反**，所以把它做成可证伪的测量。

---

## 1. 测什么（两个互相独立的统计量）

1. **分辨率**：同一批位置上 **distinct trigram 数** vs **distinct 16-tuple 数**。
   由于 tuple 是 trigram 的**函数**，恒有 `distinct tuples ≤ distinct trigrams`；
   **取等当且仅当寻址在该语料上单射**。精确、无估计量、无模型。
2. **续接 top-1**：在训练半段上分别建立
   `trigram → 众数续接` 与 `tuple → 众数续接` 两张表，在**同一**留出半段上评 top-1。
   若寻址无损，两张表只差一个重命名，**准确率必须逐位相同**；差值就是寻址扔掉的东西。
   （没有熵估计量，因此没有估计量偏置可争。）

切分是按位置**连续**的，两张表看到**完全相同**的训练半段。

---

## 2. 双向验证（TD-14 规则，冻结）

只在真 spec 上得到"无损"是**空洞的**——必须证明这个仪器**看得见损失**。
因此同一语料上跑**三个人为压扁的 spec**（`crushed_spec`，把头模数除以
$10^3 / 10^5 / 2\times10^6$；最后一个每头只剩 10 行）。

| 量 | 要求 |
|---|---|
| `real_injective` | distinct tuples **==** distinct trigrams |
| `real_top1_gap` | **== 0.0**（在单射下这是数学必然，作为交叉校验测量） |
| `control_lossy` | `crushed_2000000` **非**单射 |
| `control_top1_gap` | **> 0.05** |
| `instrument_is_sensitive` | 上面两条同时成立 |

### 2.1 控制语料必须"有结构"，这个选择现在就写下来

第一次设计控制时用的是一条 **Zipf 流**，结果压扁后的 spec **top-1 gap = 0**。
原因不是仪器坏了，而是**Zipf 流上每个上下文的众数续接都是同一个高频 token**，
所以 top-1 对碰撞天生不敏感。

**因此控制语料改用"重复固定短语"构成的流**——这恰好是**精确 n-gram 记忆本该赢的那个区间**
（也与设计文档"窄带"假设一致）。这个选择**在此刻记录**，不能事后被读成挑语料。
Zipf 流上"看不见损失"这一事实本身写进 `tests/test_addressing.py` 留档。

### 2.2 修订（2026-09-13，**在看到 wiki 一个域的数字之后、其他域之前**）

初版把灵敏度门槛写成 `control_top1_gap > 0.05`（**在真实语料上**）。
**wiki 域的实际数字让这条门槛失效，而且是可解释的**：

```text
real:           768,628 distinct trigrams -> 768,628 tuples   injective True   top1 gap +0.0000
crushed_1000:   768,628 -> 765,802                             injective False  top1 gap -0.0002
crushed_100000: 768,628 ->  40,000                             injective False  top1 gap -0.0003
trigram top-1 on WikiText = 0.0839
```

**top-1 在 WikiText 上看不见损失**，因为**众数续接几乎处处是同一个高频 token**
（top-1 仅 8.4%，被边缘分布主导）。这不是仪器坏了，是**这个统计量的适用条件**——
它在单元测试里（重复短语流）本来就被记录过。

**修订**：

1. 灵敏度门槛改为 **`control_lost_tuples > 0`**（即：同一个枚举在压扁 spec 上**确实登记了损失**）。
   **理由**：分辨率统计量是**精确枚举**，其有效性不依赖统计阈值；控制臂只需证明
   "有损失时它看得见"（765,802 < 768,628；40,000 ≪ 768,628，显然可见）。
2. **新增 applicability 臂**：在**进程内生成的重复短语流**上跑同一对 real/crushed，
   **专门用来证明 top-1 统计量在它适用的地方确实灵敏**，而不是悄悄报一个空。
3. wiki 域上 top-1 "看不见控制"这一事实**写进结果文档与测试**，不许省略。

**这次修订是否削弱真 spec 的结论？** 不：真 spec 的判据是
**768,628 distinct trigrams == 768,628 distinct tuples**，这是一次**直接枚举**，
没有阈值、没有估计量、与语料无关；控制臂只用来证明这个枚举**不是恒真的**
（它在压扁 spec 上给出了不等）。**修订记录在此，不得事后删除。**

---

## 3. 判定与后果（冻结）

```text
IF real_injective AND resolution_sensitive      => ADDRESSING_IS_LOSSLESS
ELSE IF resolution_sensitive                    => ADDRESSING_LOSES_INFORMATION
ELSE                                            => CONTROL_FAILED（不得下任何结论）

其中 resolution_sensitive = (control 非单射) AND (control_lost_tuples > 0)   [见 §2.2]
top1_sensitive_on_this_corpus 单独报告，不作门槛                          [见 §2.2]
```

**预注册的后果**（在任何数值之前写下）：

| 结局 | 动作 |
|---|---|
| `ADDRESSING_IS_LOSSLESS` | **"哈希碰撞把信息在读取前就毁了"这一假设被退役。** TD-11 改写为**梯度预算错配**：每个 trigram 有自己的行，但那行收到多少更新由它在预训练里的出现次数决定，与骨干*需要*它的程度无关。阶梯的损失归因收窄到 **③→④**。 |
| `ADDRESSING_LOSES_INFORMATION` | 这本身是一级发现：必须把寻址瓶颈加进天花板，重算 1.5c/1.5e 的"够不够"的判据。 |
| `CONTROL_FAILED` | 仪器无效，1.5f 不给任何结论，退回改仪器。 |

**注意本级的范围**：它**只**说寻址。它**不**说明任何一行是否学到了正确的值
（那是阶③），也**不**说明读出能否提取（阶④）。

---

## 4. 域面板与产物

wiki / stem / code 三个评测流各跑一次。
产物：`outputs/round168/addressing/<tag>-addressing.{json,md}`。

**CPU only、无 GPU、无训练**；可与任何 GPU 任务并行（这正是把 1.5f 排在 1.5c 的
backbone 阶段等锁期间做的原因——低利用率的时段用来做不需要 GPU 的实验）。
