# Round 154: 本轮完整整理与压缩上下文 handoff

> 日期：2026-09-11 08:15 CST
> 用途：**上下文压缩后第一份要读的文档**。把本轮的计划、发现、尝试、坑、
> 已完成、未完成、未来计划集中到这一份。
> 上游：`docs/round-151-session-consolidation-and-handoff.md`（上一版全量交接）、
> `docs/round-153-goal-tech-debt-and-development-plan.md`（目标/债/计划）、
> `docs/round-152-lora-row-and-g0-4b.md`（本轮实验）、
> `docs/round-152-g3-engram-edge-storage-benchmark.md`（G3 实测）。

---

## 0. 一句话状态

**修正后的 LoRA 行已完成并且有效：LoRA 共适应确实有效（+2.8～3.5 点，且不再崩塌），
但 PLE 的内容效应依然为零（real vs control 生成指标完全相同，gold NLL 反而差 0.018 nat）
→ 预注册的 G1 判定为不通过。** 这把"reader/适应不够好"这最后一个解释也排除了。
G0（4B，hidden 2560 精确对齐 PLE 源空间）正在跑，是"冻结跨模型迁移是否还有正交信息"
的判决点。G3 磁盘基准已实测通过 NVMe 行。

**本轮最重的债不是性能债，而是"科学诚信债"**：我们三次产出"看起来正常、实际无效"的
实验（对照被全关、LoRA 权重没进 optimizer、指标变了但输出分布没变）。根因是缺少
不变量校验基础设施，这是下一优先级（见 `round-153` §3.1）。

---

## 1. 本轮目标与计划

用户原话："继续，尽可能自动化地往前推进，我要睡觉了"+"明天一早我会再检查进度"，
随后要求准备压缩上下文。因此本轮 = **一次夜间自主实验 + 一次完整整理**。

计划（实际执行顺序）：

```text
1. 实现 LoRA 行（0.8B × {frozen, LoRA} × {no PLE, real, control}）
   —— 因为 Round 149 的 full-FT 行配方崩塌，无法判定 PLE interaction
2. G0：Qwen3.5-4B（hidden 2560 == PLE 源空间）冻结嫁接
3. G3：EngramDB 磁盘服务微基准（与 GPU 解耦，可并行）
4. 拉结果 → 总结 → commit/push → 关机
```

---

## 2. 本轮发现

### 2.1 科学发现（最重要）

**修正后的 LoRA 行（standard 1500, raw prompt）**：

| Arm | BoolQ | TriviaQA | NQ | Mean EM | Gold NLL |
|---|---:|---:|---:|---:|---:|
| frozen no-reader（R148-A 参照） | 0.672 | 0.128 | 0.062 | 0.2873 | 8.7429 |
| lora-nople（PLE 关） | 0.770 | 0.144 | 0.052 | **0.3220** | 2.4025 |
| lora-real | 0.764 | 0.128 | 0.054 | 0.3153 | 2.4025 |
| lora-control | 0.758 | 0.140 | 0.048 | 0.3153 | **2.3845** |

配对（正 = 前者更好）：

```text
lora-real vs lora-control   EM  +0.0000 ± 0.0049   NLL −0.0180 ± 0.0068
lora-real vs lora-nople     EM  −0.0074 ± 0.0055   NLL +0.0008 ± 0.0088
```

结论：**LoRA 适应有效（+2.8～3.5 点、不崩塌），PLE 内容效应仍为零 → G1 不通过。**

**G3 磁盘服务基准**（`docs/round-152-g3-engram-edge-storage-benchmark.md`）：

```text
NVMe 热缓存  462K rows/s @2048 rows → ≈28.9K tokens/s；2048-token 预填充 ≈71 ms
NVMe 冷缓存  ~70K rows/s
/dev/shm     276–673K rows/s
抓取路径 RSS ~400 MB；bench read ~140 MB
→ NVMe 行的吞吐与内存预算【通过】；USB SSD/SD 与移动端功耗【未测】
```

**chat 次要协议（有 caveat）**：`lora-nople` chat 复评 mean EM 0.3593 / gold NLL 5.0249，
而同一 adapter 在 raw 协议下 gold NLL 2.4025。这个 2.40 → 5.02 的差距**量化了协议不匹配的
代价**：chat 复评复用的是 **raw 训练**的 adapter/reader，属协议不匹配，
因此**只有 real vs control 的差值可解释，绝对值不可与 R148-A 的 chat 行比较**
（R148-A 是按协议分别训练的）。教训：**训练与评测必须同协议**。

### 2.2 工程发现（本轮真正花时间的地方）

**关键 bug #1（最严重）：LoRA adapter 从未进入 optimizer。**
`_train_reader` 只在 `train_backbone=True` 时收集模型参数，而 LoRA 路径传的是 `False`
（`--finetune-backbone` 才有）。后果：LoRA 权重有梯度但从不更新，
**500 步后 96/96 个 `lora_B` 全为 0**。证据链：

```text
保存的 adapter        : 96/96 lora_B 全 0，A 仍是初始随机值
载入 adapter vs 基座   : max logit diff = 0.0，argmax 相同
gold NLL vs frozen    : 1500/1500 条 bit-identical
修复后重训 30 步       : 96/96 lora_B 变为非零
```

已修（`e5b8678`）+ 加硬守卫：请求了 adaptation 时，所有 trainable 模型参数必须在
optimizer 内，否则直接 raise，而不是静默产出无效 arm。

**关键 bug #2：reader dtype 不匹配，G0 秒崩。**
hook 只把**输出**转成 hidden dtype，没把**输入**转成 reader 的 fp32 →
4B bf16 backbone 报 `expected mat1 and mat2 to have the same dtype: BFloat16 != float`。
已修（hook 现在双向转 dtype），并在真实 4B bf16 路径实测通过。

**关键 bug #3：finisher 无条件关机。**
它只看链路 DONE 标记、不看各步骤退出码，于是在 G0 失败的情况下
**在 01:15:39 执行了 `shutdown -h now`**，浪费约 6 小时 GPU。
已重写：现在以**产物存在性 + 脚本自身 DONE** 为唯一判据，任一缺失就写
`FINISH_BLOCKED` 并保持开机。**该修复已实际生效**——本轮重启 finisher 时它打印
`REFUSING SHUTDOWN: g0 DONE marker missing` 并退出，正是期望行为。

**关键 bug #4：`chain/*.status` 的 `$?` 在 helper 之后读取**，失败被记成 `rc=0`
（`g0.status` 里那句 `g0 rc=0` 其实是 01:15 的失败）。因此 finisher 的判据
不再信任 status 文件。

**关键 bug #5：`--ple-off` 顺带跳过 QA SFT cache**，导致 no-PLE 格只在语料上训练
（与 PLE 格差两个变量）。已修：SFT cache 只要有 `--qa-sft-file` 就构建，
lazy entry 允许无 `e_t`。

**关键 bug #6：`mode=no-reader` 在训练前 return**，所以 2×2 的 no-PLE 格必须是
`--ple-off`（reader 存在但贡献被抑制），否则拿到的是未训练 backbone。

**性能工程（本轮顺手做掉）**：
* gold NLL 重写为分批（每 item 一行 + 右填充，保证 reader 因果卷积不跨 item）。
  与逐条版本**数值等价**（max diff 2.9e-5 float32），评分阶段 **3× 加速**
  （1500 条 28 分钟 → 约 9 分钟）。
  *中途踩坑*：先把多 item **打包成一条序列**，实测 bs=1 0.751 vs bs=8 12.07 —— 
  打包会改变短卷积语义，**必须每 item 一行**。
* 修掉自 round 148 起就红的 CI 测试（`test_oracle_routing_probe` 的 stratified folds
  断言与实现矛盾：strata 只有 2 个样本却要求 ≥2 折）；
  远程全量套件 **143 passed / 7 skipped**。

### 2.3 科学框架层面的发现（写进 round-153）

三重排除已完成：reader 容量（148-B 解冻 source 无效）、gate 可学性（oracle AUC≈0.5）、
backbone 适应性（本轮 LoRA 真实生效但不改变 PLE 冗余）。
**剩下唯一主假设**：这块表主要携带局部延续统计，对已在同一语料预训练过的 backbone
大部分冗余（51.2B 字节 ≈ 1.6M token 级逐字上下文 vs 权重对 2–3T token 的有损压缩）；
增量只可能出现在权重压不住处（罕见实体绑定、逐字延续、近重复文本）。
这与"2-gram 主导"、"control 打平"完全自洽。

---

## 3. 做的尝试与产出物

### 3.1 代码（本轮）

```text
scripts/run_phase0.py
  --lora / --lora-r / --lora-alpha / --lora-dropout / --lora-target-modules
  --load-lora-adapter     复用已训 adapter 复评（免重训）
  --backbone-dtype        float32 | bfloat16 | float16（2B/4B 必用 bf16）
  --ple-off               真正的 no-PLE 格（reader 在、贡献被抑制）
  --qa-max-items          smoke 用评测截断（禁止用于正式数字）
  分批 gold NLL（数值等价、3× 加速）
  optimizer 参数收集修复 + adaptation 不变量守卫
  save_lora_adapter / load_lora_adapter（adapter-only 落盘，不写整份 backbone）
scripts/summarize_arms.py      generation EM + gold NLL + 逐任务配对 Δ（新增）
scripts/bench_engram_edge.py   G3：热/冷页缓存、batch 扫描、RSS/IO（新增）
scripts/run_round152_lora.sh   LoRA 三臂队列
scripts/run_round152_g0_4b.sh  G0 4B 队列（bf16，自带 reader）
scripts/run_round152_lora_chat.sh  chat 协议复评（复用 adapter）
scripts/ssh_autodl.sh          固定选项 SSH 封装（host key 变更绕行）
src/qwen35_ple/reader.py       hook: peft 解包、dtype 双向转换、_ple_disabled
tests/test_oracle_routing_probe.py  修正 CI 红测试
```

### 3.2 远程产物

```text
outputs/round152/          lora-nople/real/control.json + arms-summary.md + gold-nll-raw.md
                           adapter-<arm>-seed0（reader）+ .lora/（adapter）
outputs/round152chat/      chat 协议复评（lora-nople 已完成，real 进行中）
outputs/round152g0/        train-real/control.json + reader-4b-*.pt（评测进行中）
outputs/g3/                nvme-persistent.{json,md} + shm.{json,md}
outputs/_aborted_round152/ 两次无效尝试的归档（无效 LoRA 行、01:15 失败）
logs/round152-*.log        队列与 finisher 日志
```

### 3.3 提交

```text
97994a3  --qa-max-items、G0 队列、G3 findings
a75704c  ci: lint 新脚本
eca3560  ple-off 与 PLE 格保持同一 QA SFT mix
c2502de  summarize_arms 修复（detail KeyError、逐任务 fallback）
fe33fe6  round-152 文档骨架
dd837f4  tests: 修正自 round 148 起的 CI 红测试
f19f2d5  docs: round-151 §12
e5b8678  【关键】LoRA adapter 未进 optimizer 的修复 + 守卫
0705dee  --load-lora-adapter + chat 复评队列
ebc7522  docs: round-153 目标/债/计划
```

---

## 4. 坑与教训（按重要性）

1. **"训练没报错" ≠ "训练生效了"**。三次无效 arm 都源于此。
   修复不只是改 bug，而是**加不变量**（参数距离、对照贡献范数、trained-vs-base 输出一致率）。
2. **关机判据不能用"控制流的副产品"**。用产物存在性；退出码就地捕获，
   不要在 helper 调用之后读 `$?`。
3. **打包 vs 每 item 一行**：改变序列布局会改变因果卷积语义。任何"加速重写"
   都要先证明数值等价（本轮用逐条对拍 + float32 容差）。
4. **训练与评测必须同协议**。复用 raw-trained adapter 跑 chat 评测
   只能看差值，绝对值无意义（本轮 2.40 vs 5.02 就是代价）。
5. **bf16 backbone 必须双向转 dtype**：输入给 reader 前转 reader dtype，输出转回 hidden dtype。
6. **AutoDL 重启后 SSH host key 变更**：用 `scripts/ssh_autodl.sh`，不要用 `BatchMode` 裸连。
7. **`pkill -f run_phase0.py` 会杀掉自己的 SSH 命令**（命令行含同串）→ 用 `[r]un_phase0`。
8. **`/dev/shm` 重启即失**：用 `scripts/ensure_qwen38_rows.sh --python <venv python>` 恢复（129 文件）。
9. **实例可能被外部关机**（01:24 起 connection refused，非我方 finisher 所致）；
   长任务要有可续跑的产物设计（本轮靠 per-arm JSON + skip 逻辑救回）。
10. **tmux 引号陷阱**：经 tmux 传含 `\n` 的模板时，务必核对 `/proc/<pid>/cmdline`
    确认是真实换行（本轮核对过，正常）。
11. **文档数字必须机器生成**：README 里 4B 的"NQ 0.748"其实是三段拼接的第 1 段
    （见 round-153 §3.3），需要更正或标注。

---

## 5. 已完成 / 未完成

### 已完成

* [x] LoRA 路径完整实现（peft 0.20.0），adapter-only 落盘，含复评能力
* [x] 6 个关键 bug 的定位与修复，其中 2 个会让实验静默失效
* [x] **修正后的 LoRA 行完成**（三臂 + 汇总），G1 判定：不通过（PLE 无内容效应）
* [x] G3 磁盘基准（NVMe 行通过）+ 文档
* [x] 分批 gold NLL（等价 + 3× 加速）
* [x] CI 恢复绿色（远程 143 passed / 7 skipped）
* [x] finisher 关机判据加固（并实际拦下一次误关机）
* [x] `docs/round-153` 目标精确化 / 技术债 / 开发计划
* [x] `/dev/shm` 行表恢复

### 未完成

1. **G0 4B 结果**（`g0-real` 生成中 → 之后 `g0-control`）——**决定 Q1 的唯一依据**
2. chat 复评剩余两臂（`lora-real` 进行中 → `lora-control`）
3. G0 / chat 结果写进 `docs/round-152-lora-row-and-g0-4b.md` 的 `<!-- RESULTS:G0 -->` 占位
4. 源空间上限实验（①）：4B hidden 2560 直接作为 reader 输入的 oracle 上限
5. 表内容正交性审计（②，无 GPU）：合成绑定探针，区分"读不出"与"没有"
6. 比例扫描（③）：取模剪表（`mixed % size` 天然支持），零重训
7. G2 hot-row 表共适应（仅当 Q1 有信号）
8. 统一 runner + 不变量校验 + artifact manifest（round-153 §6 清单）
9. G3 扩到 USB SSD/SD；小型 reasoning/长上下文评测集

---

## 6. 未来计划（详见 round-153 §4）

```text
Q1 判决   G0 + 表内容正交性审计 → 关闭或保留"冻结跨模型迁移"
Q2a 对齐  若 Q1 有信号：G2 hot-row 表共适应
Q2b 区间  比例扫描（取模剪表，零重训）+ 同预算纯参数基线 → 质量/字节/算力曲线
Q3 形态   最小可用记忆（2-gram-only / 量化 / 剪枝蒸馏 / tiny co-trained）+ G3 扩 SSD
停止规则  若任何比例下都不优于同预算纯参数方案 → PLE 重定位为
          "格式先验 + 热插拔知识补丁"，通用性能目标改由参数/数据侧解决
```

两条必须固化的纪律（round-153 §4.3）：
**① 无效实验不可静默通过；② 每个质量声明必须带「激活参数量 / FLOPs / 内存」三元组。**

---

## 7. 环境与运维手册

```text
连接（host key 会变，用封装脚本）:
  cd <repo> && scripts/ssh_autodl.sh '<command>'

目录:
  /root/autodl-tmp/qwen35-ple/repo           代码（本地是唯一真源，手工 tar 同步）
  /root/autodl-tmp/qwen35-ple/venv           Python 环境（已装 peft 0.20.0、pytest）
  /root/autodl-tmp/qwen35-ple/models         0.8B / 2B / 4B + tokenizer + qwen38_ple
  /root/autodl-tmp/qwen35-ple/qwen38-rows    持久 PLE 行（129 文件 ~48G）
  /dev/shm/qwen38-rows                       快速行（重启即失，用 ensure 脚本恢复）
  /root/autodl-tmp/qwen35-ple/outputs        实验产出
  /root/autodl-tmp/qwen35-ple/logs           日志

模型规格（实测）:
  0.8B hidden 1024 / 24 层 / vocab 248320；2B hidden 2048 / 24 层；
  4B hidden 2560 / 32 层（== PLE 源空间，checkpoint 以 bf16 存储 9.3GB）
  三者架构均为 Qwen3_5ForConditionalGeneration，tokenizer 与 Qwen3.8 共享

当前状态（08:15）:
  tmux: g0（4B 评测）、lorachat（chat 复评）、finish（等待 G0 DONE 后校验并关机）
  GPU: 两个任务并行；磁盘 /root/autodl-tmp ~12G free
  自动关机: finisher 会在【校验全部产物】后执行 shutdown -h now
           → 若发现机器已关，用 AutoDL 控制台开机即可；产物在持久盘上不会丢

重启后恢复:
  bash scripts/ensure_qwen38_rows.sh --python /root/autodl-tmp/qwen35-ple/venv/bin/python
```

---

## 8. 压缩上下文后的恢复动作

1. 读本文档 + `docs/round-153-goal-tech-debt-and-development-plan.md`。
2. 确认实例状态（可能已被 finisher 关机 → 控制台开机）。
3. 读远程 `outputs/round152g0/`（G0 结果）与 `outputs/round152chat/`（chat 复评），
   以及 `logs/round152-g0.log` / `logs/round152-lora-chat.log` 的末尾确认是否跑完。
4. 把 G0 结果填进 `docs/round-152-lora-row-and-g0-4b.md` 的 `<!-- RESULTS:G0 -->`，
   更新 `<!-- RESULTS:CONCLUSION -->`，commit + push。
5. 按 Q1 判定分支：
   * **G0 无内容效应** → 关闭"冻结跨模型迁移"，直接做 Q2b（取模剪表比例扫描）
     与 ②（正交性审计）；
   * **G0 有内容效应** → 做 G2（hot-row 表共适应）。
6. 无论走哪条，**先落地不变量校验**（round-153 §4.3），否则下一轮还会产出无效 arm。
7. 不要在 Q1 判决前开启新的大方向。
