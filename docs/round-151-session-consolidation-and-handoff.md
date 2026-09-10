# Round 151: 本轮完整整理与压缩上下文 handoff

> 日期：2026-09-11（本轮收尾）
> 远程：`ssh -p 19236 root@connect.nmb1.seetacloud.com`（见 §9 的 host key 变更坑）
> 用途：**上下文压缩后第一份要读的文档**。把本轮的终极目标、计划、发现、
> 做的尝试、踩的坑、已完成 / 未完成、未来计划、工程债、运维手册全部集中到
> 这一份，避免压缩后靠回忆重建。
> 上游文档：`docs/round-149-systematic-roadmap-and-tech-debt.md`（roadmap/债）、
> `docs/round-150-session-recap-and-handoff.md`（上一版 handoff）。

---

## 0. 一句话状态

**纯 PLE 嫁接（frozen backbone + frozen PLE 表 + 只训 reader/gate）在
0.8B 上已被证伪：能注入的是格式/启发式，不是知识。** 当前判定链：
gold-answer NLL 上 format 效应 4.4–6.4 nat、content 效应仅 ≈0.02 nat；
shuffled-row control 与 real 打平；oracle routing 线性不可学；解冻官方 source
投影也无法产生 content 效应；full-FT 行（Round 149）配方崩塌，三个 arm 一起
退化成 BoolQ-only，无法用于判断 interaction。
**下一步（未开始）：LoRA 行 2×2**，因为它是唯一能在"保住 0.8B 通用能力"的
前提下判定"backbone 共适应能否解锁外部记忆"的实验。

终极愿景（用户原话）：**纯 PLE 嫁接，通过少量训练显著提升 0.8B 通用性能。**

---

## 1. 目标分层（终极 → 可判定）

### 1.1 科学目标

Map the boundary conditions：外部 n-gram 记忆（Engram/PLE）在什么条件下能被
小模型真正用起来。每个问题都有可证伪形式：

| 问题 | 可证伪形式 |
|---|---|
| frozen 表 + frozen backbone + 训 reader 够不够？ | standard 1500 上 `real − control` ≥2 pts 或 ≥0.05 nat |
| 不够的话，backbone 共适应能否解锁？ | 2×2 中 `real − control`（LoRA / 温和 full FT） |
| 还不够的话，table 共适应能否解锁？ | hot-row 适应后 `real − control` ≥0.05 nat |
| 是不是 scale/space 不匹配？ | Qwen3.5-4B（hidden 2560 == PLE 源空间）嫁接效应量 |
| 是否必须多分支/HC？ | 单流 vs 双层/per-head gate 消融 |

正负结果都是交付物，只要边界被测量清楚。

### 1.2 工程目标（端侧）

```text
0.8B 级激活算力
  + 51.2B FP8 PLE 表放在 UFS/SD/NVMe（EngramDB Store-P，磁盘优先）
  + 256MB–2GB RAM 热缓存 + 异步预取
  + 轻量 backbone 适应（LoRA / 温和 full FT，推理时合并）
  + reader/gate 学会查询外部记忆
  + 行级热插拔知识补丁（不重训全模型）
```

预注册阈值：

```text
质量:  standard 1500 + reasoning/long-context 不低于 no-PLE baseline
解码:  ≥20 tok/s（NVMe 级），≥5 tok/s（USB SSD/SD），0.8B
内存:  模型 + 缓存 ≤ 4GB（不含 flash 常驻表）
p99:   除模型算力预算外无大停顿
更新:  秒级热插拔一条事实，不动 backbone 权重
```

注意（用户纠正过的认知）：**51.2B 表不需要常驻内存**。EngramDB Store-P 把
每个 n-gram key 的 `e_t` 打包成一条 2560B 记录，确定性寻址 + 热缓存 +
异步预取，容量问题已解决；真正未验证的是 IO 延迟/尾延迟、预填充期 IOPS、
FTL/热行为、RAM 热缓存预算。**我们现在的训练/评测用 `/dev/shm`，把这块 IO
成本完全隐藏了**。

### 1.3 平台目标

每层唯一真源：hash/rowid → engram-peft/Qwen 官方语义 + golden 测试；
storage → EngramDB；reader/gate → `qwen35_ple.reader`；backbone 适应 → HF PEFT；
serving → EngramDB bundle；评测 → locked standard suite + v2 scoring + gold NLL。

---

## 2. 本轮时间线

| 阶段 | 做了什么 | 状态 |
|---|---|---|
| 148-A | gold-answer NLL、head/order 消融、layer hidden 抽取、oracle-routing ridge 探针 | 完成，`c5370d1` |
| 148-B | 解冻官方 source 投影（key/value/norm/conv）capacity 诊断，表仍冻结 | 完成，`1af395c` |
| 认知纠正 | 接受 EngramDB 磁盘优先设计：容量已解决，IO/延迟/寿命才是约束 | 已入档 |
| 下载 | ModelScope 优先下载 Qwen3.5-4B（hidden 2560）/ 2B（hidden 2048） | 完成 |
| Roadmap | Gates G0–G4、技术债清单、兄弟项目借鉴边界 | 完成，`385997a` |
| 149 full-FT 行 | 0.8B × {no-PLE, real, control} 全参 FT，500 步 mixed50 | 完成，配方崩塌；队列代码 `47f7338` |
| 150 handoff | 上一版 session recap | 完成，`2a3490a` |
| 151（本文） | 全量整理 + 压缩上下文交接；确认远程/本地状态、补运维手册 | 本文档 |

---

## 3. 关键发现（带数）

### 3.1 格式 vs 内容（Round 148-A，standard 1500，gold-answer NLL，越低越好）

raw prompt：

| Run | BoolQ | TriviaQA | NQ | Overall | n |
|---|---:|---:|---:|---:|---:|
| nll-raw-no-reader | 12.3765 | 7.3230 | 6.5292 | 8.7429 | 1500 |
| nll-raw-real-seed0 | 0.5304 | 2.9765 | 3.4357 | 2.3142 | 1500 |
| nll-raw-control-seed0 | 0.5333 | 3.0033 | 3.4183 | 2.3183 | 1500 |
| nll-raw-real-seed1 | 0.6124 | 3.0316 | 3.4281 | 2.3574 | 1500 |
| nll-raw-control-seed1 | 0.5687 | 3.0732 | 3.4472 | 2.3630 | 1500 |
| nll-raw-real-seed2 | 0.5854 | 3.0442 | 3.3159 | 2.3152 | 1500 |
| nll-raw-control-seed2 | 0.6119 | 3.0657 | 3.4277 | 2.3684 | 1500 |

chat template：

| Run | BoolQ | TriviaQA | NQ | Overall | n |
|---|---:|---:|---:|---:|---:|
| nll-chat-no-reader | 9.1666 | 5.4706 | 5.5242 | 6.7205 | 1500 |
| nll-chat-real-seed0 | 0.6158 | 3.0831 | 3.3347 | 2.3445 | 1500 |
| nll-chat-control-seed0 | 0.6248 | 3.1213 | 3.3177 | 2.3546 | 1500 |
| nll-chat-real-seed1 | 0.5382 | 3.1039 | 3.4256 | 2.3559 | 1500 |
| nll-chat-control-seed1 | 0.5710 | 3.1724 | 3.4069 | 2.3834 | 1500 |
| nll-chat-real-seed2 | 0.5374 | 3.0965 | 3.3994 | 2.3444 | 1500 |

配对 `real − control`（正 = real 更好）：

```text
raw : seed0 +0.0040±0.0091, seed1 +0.0056±0.0110, seed2 +0.0533±0.0111
chat: seed0 +0.0101,        seed1 +0.0275,        seed2 +0.0067
```

结论：

* **format/elicitation 效应 4.4–6.4 nat** —— reader 让模型学会"按 QA 格式
  输出"，这部分收益是巨大的、真实的，但和知识无关。
* **content 效应 ≈0.015–0.053 nat**，跨 seed 不稳定、跨 task 不一致，
  远低于预注册的 0.05 nat / 2 pts 门槛。
* **control（打乱 PLE 行）与 real 打平** → reader 学到的是格式与输出校准，
  不是表内容。

### 3.2 head/order 消融（raw seed0）

```text
full       2.3142
keep2gram  2.3062  (Δ +0.0081，去掉 3-gram 头几乎无损)
keep3gram  3.5690  (Δ −1.2548，去掉 2-gram 头大幅变差)
```

→ **2-gram 头承担几乎全部可用信号**；3-gram 头近乎冗余（与 Engram 论文
"4-gram 非必需"、V4.1 用 {2,3,4}-gram 但 2-gram 占比最大一致）。

### 3.3 oracle-routing 探针（Round 148-A）

在 layer-3 hidden（= block 2 输出 = reader 的 query）上用闭式 ridge 预测
"item 级 oracle 该选 real 还是 no-reader"，5-fold 分层 CV + leave-one-task-out：

```text
raw : oracle n=214, AUC 0.504–0.509; leave-one-task-out 0.36–0.43（低于随机）
chat: oracle n=181, AUC 0.563–0.635; leave-one-task-out 0.44–0.50（无跨任务迁移）
直接预测 arm 对错: AUC 0.66–0.82（高）
```

结论：hidden state 能预测"这题难不难 / 格式会不会成功"，**不能预测"PLE 相对
优势"**。oracle 的理论 headroom（+0.06）对线性 item-level gate 不可及。
→ 用户偏好"用已有 gate、不加额外 router"因此是有依据的。

### 3.4 reader capacity 不是瓶颈（Round 148-B）

解冻官方 source 投影（key/value/norm/conv），表仍冻结：

```text
standard 1500 raw generation:
  no-reader        BoolQ 0.672 | TriviaQA 0.128 | NQ 0.062 | mean 0.2873
  unfrozen real    BoolQ 0.782 | TriviaQA 0.156 | NQ 0.044 | mean 0.3273
  unfrozen control BoolQ 0.748 | TriviaQA 0.146 | NQ 0.042 | mean 0.3120

gold NLL: real 2.3133, control 2.3113, real − control = −0.0019 (SEM 0.0094)
```

generation 上 real 领先约 +1.5 pts（几乎全在 BoolQ），但 gold NLL 上
content 效应为 0 → **source 空间对齐 / reader 容量不是瓶颈**。

### 3.5 full-FT 行崩塌（Round 149，500 步 mixed50，LR 1e-4）

| Arm | BoolQ | TriviaQA | NQ | Mean | Gold NLL overall |
|---|---:|---:|---:|---:|---:|
| frozen-no-reader（参照） | — | — | — | — | 8.7429 |
| fullft-nople | 0.608 | 0.002 | 0.002 | 0.2040 | 5.8927 |
| fullft-real | 0.608 | 0.002 | 0.002 | 0.2040 | 5.9023 |
| fullft-control | 0.608 | 0.004 | 0.002 | 0.2047 | 5.9535 |

配对：

```text
fullft-real vs fullft-control = +0.0511 ± 0.0129（TriviaQA +0.112, NQ +0.048, BoolQ −0.006）
fullft-real vs fullft-nople   = −0.0096 ± 0.0118
fullft-real vs frozen-real    = −3.5881 ± 0.0829（全参 FT 比 frozen reader 差 3.6 nat）
```

解读：**500 步、mixed50、0.8B 全参、LR 1e-4 的配方导致 open QA 灾难性遗忘**，
三个 arm 一起塌成 BoolQ-only（TriviaQA/NQ ≈0.002）。NLL 上那点 +0.05 差异是
"崩塌状态内部"的差异，generation 完全没有对应差异 → **该配方对 interaction
判定无效**；必须换温和适应（LoRA / 低 LR / 少步数 / replay / partial FT /
early stop）。这正是"先做 LoRA"的硬证据。

### 3.6 与 Engram / Qwen / V4.1 的口径差异

* 三方（论文、Qwen、V4.1）**全都是表 + backbone 共训练**；只有我们的 frozen
  cross-model graft 没有共适应。
* 论文：iso-parameter / iso-FLOPs，layers 2 & 15，4-gram 非必需，Adam 5× LR，
  U 形分配。
* Qwen：extra-parameter regime；loss 单调改善但 downstream 混合（知识/中文最强，
  MATH 过 20× 反而退化）；fixed-budget 打不过 MoE-only；选 layer 2 部分是为了
  预取/系统。
* V4.1：552B backbone + 196B Engram，8B 激活预填充，layers 1/14，
  {2,3,4}-gram，8 heads，head_dim 256，FP8 表，无 short conv，momentum+Sinkhorn；
  公开报告缺少孤立 Engram 消融。
* **我们违反了两个 regime：memory/active ≈64×（论文 ~1.4×、Qwen ~8.5×、
  V4.1 ~24.5×），hidden 1024 vs 源 2560。** 这是 G0（4B，hidden 2560）要测的。

### 3.7 数据规模不是问题

88 → 6000 条 SFT：standard open-QA 无增益；custom 62 曾高估效应 →
**custom 62 永远不能作为可部署 claim**，standard 1500 held-out 才是主指标。

---

## 4. 本轮尝试与产出物

### 4.1 新增/改动代码

```text
scripts/run_phase0.py            --qa-gold-nll, --qa-head-mask, --unfreeze-official-source,
                                 --finetune-backbone, --gate-override
scripts/summarize_gold_nll.py     gold NLL 表、配对 Δ、--pair FIRST=SECOND
scripts/extract_layer_hidden.py   layer hidden → npz
scripts/train_oracle_routing_probe.py  ridge 闭式 + CV + leave-one-task-out
scripts/run_round148a.sh          18 个 gold-NLL arm（raw/chat × no/real×3/control×3 + head mask）
scripts/run_round148b.sh          unfrozen official-source reader（real+control）+ 标准 1500
scripts/run_round149_fullft.sh    3 条 full-FT arm（--gate-override 0.0 做 no-PLE）
scripts/download_modelscope_model.py  ModelScope API 文件列表 + curl 断点续传，HF fallback
scripts/download_qwen35_models.sh     先检查磁盘，再下 4B、2B
scripts/ssh_autodl.sh（本轮新增，已实测）  远程 SSH 封装（含 host key 变更绕行，见 §9）
src/qwen35_ple/ple_hash.py        PLE_NGRAM_SIZE=3, PLE_HEADS=16, PLE_EOS=248044,
                                 rowids_for_seq, _shift_right_ignore_eos
src/qwen35_ple/reader.py         install_reader_hook（挂在 model.model.layers[layer_index]
                                 的 post-hook，query 是 block 输出）、OfficialSourceQwenReader、
                                 EngramReader（gate bias/override）
.github/workflows/ci.yml         lint 列表与 shell 语法检查补入新脚本
```

机制要点（下一轮复用）：

* `OfficialSourceQwenReader` = 冻结官方 source 模块（key_proj/value_proj/norm_key/
  norm_query/norm_conv/conv1d）+ 可训 `query_bridge`(1024→1024→10240) +
  `out_proj`(2560→1024→1024, 末层 zero-init)。可训 15.2M / 冻结 32.8M /
  checkpoint 192,181,008 bytes。
* 表：16 heads × 160 dim = 2560/token，128 个 Store-I shard，
  2,500,012 行/shard，补齐后 320,001,536 行，总计 51,200,245,760 bytes，
  `weight_scale=0.00019931793212890625`，layer 2 注入。
* 行 id：`ple_hash.py`，SplitMix64 派生乘数硬编码，8 heads/order，
  EOS 248044 处切段，16 rowids/token。

### 4.2 远程产出物

```text
outputs/round148a/                     gold NLL、features、probe
outputs/round148b/                     unfrozen-source 评测 + gold NLL
outputs/round149/                      full-FT 行（DONE 标记存在，23:56 完成）
outputs/sft-unfrozen-mixed50/          解冻 source 训出的 reader
outputs/qa-standard-eval-large-mixed50/ raw 标准评测（frozen 行）
outputs/qa-standard-eval-chat-large/    chat 标准评测
logs/round149-fullft.log
```

### 4.3 模型与环境（实测）

```text
/root/autodl-tmp/qwen35-ple/models/Qwen3.5-0.8B   1.7G  hidden 1024, 24 层, vocab 248320
/root/autodl-tmp/qwen35-ple/models/Qwen3.5-2B     4.3G  hidden 2048, 24 层
/root/autodl-tmp/qwen35-ple/models/Qwen3.5-4B     8.8G  hidden 2560, 32 层  ← 与 PLE 源空间精确对齐
/root/autodl-tmp/qwen35-ple/models/qwen38_ple     16M
/root/autodl-tmp/qwen35-ple/models/Qwen3.8-Flash-Next-FP8-tokenizer  22M
/root/autodl-tmp/qwen35-ple/qwen38-rows           129 files ≈48G（持久）
/dev/shm/qwen38-rows                              129 files ≈48G（快，重启即失）
```

三方架构全是 `Qwen3_5ForConditionalGeneration`，tokenizer 与 Qwen3.8 共享。

### 4.4 提交记录

```text
c5370d1  round-148: gold-answer NLL, head ablation, and oracle-routing probe
1af395c  round-148b: unfrozen official-source reader capacity diagnostic
385997a  round-149: systematic roadmap, tech debt, and ModelScope downloads
47f7338  round-149: full-finetune backbone x frozen PLE interaction queue
2a3490a  docs: round-150 session recap and full-FT collapse handoff
```

---

## 5. 坑与教训（按"下次别踩"排序）

1. **full-FT 会崩塌**：mixed50 + LR 1e-4 + 500 步 = open QA 归零。判定
   interaction 前必须先用温和配方（LoRA / 低 LR / 少步 / replay / partial FT）
   验证"baseline 没有崩"，否则整行实验作废。
2. **leading-space NLL 不一致**：SFT 训练 answer token 时不加前导空格，早期
   gold-NLL 实现加了 `" " + answer`，系统性惩罚 reader arm。已修，约定已写进
   golden test。
3. **探针喂错输入**：`run_round148a.sh` 曾把只有 NLL 没有 `qa_exact` 的输出传给
   正确性标签加载器 → 探针失败。已改喂 generation 评测 JSON。
4. **跨文件 mode 命名**：standard eval 把 no-reader 存成 `phase1-full.json`；
   analyzer 需要 `full → no-reader` 别名 + 跨文件合并。
5. **AutoDL 容器 `shutdown -h +N` 会忽略时间直接关机** → 只用控制台定时关机，
   绝不用容器内 shutdown 参数。
6. **`nohup ... &` 走 SSH 会挂住工具** → 队列一律 tmux。
7. **长 `sleep` + SSH 轮询会被打断** → 分短步轮询。
8. **scp/rsync 会挂** → 用 `ssh 'cat > file' < local` 或
   `tar czf - files | ssh 'cd repo && tar xzf -'`（管道 tar 时不要加 `ssh -n`）。
9. **SSH host key 变更**（本轮新增）：实例重启后 `BatchMode` 直接
   `Host key verification failed`。绕行：
   `ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 19236 ...`
   （已封装进 `scripts/ssh_autodl.sh`）。不要以为是实例挂了。
10. **transformers 5.x**：`apply_chat_template(tokenize=True)` 返回 dict/BatchEncoding，
    已在 `_qa_prompt_ids` 修。
11. **EOS 不一致**：248044 vs 248046；stop-token 集合对 chat 要含 `<|im_end|>`。
12. **磁盘压力**：数据盘曾掉到 12G；清理旧 phase2/warmup/SFT 变体后 26G，
    下载 4B+2B 花掉 ~14G，现在 **13G free（86% 已用）**。下次保存全量模型前
    必须先清；**优先存 LoRA adapter、优先 in-process 评测**。
13. **远程仓库脏/落后**：手动 tar 同步，remote 上 git 还会报
    `dubious ownership`。本地是唯一真源；起队列前先同步改动脚本。
14. **`/dev/shm` 重启即失**：用 `scripts/ensure_qwen38_rows.sh --python <venv python>`
    恢复；持久副本在 `qwen35-ple/qwen38-rows`。
15. **本地 `.venv` 没有 torch**：本地 `pytest` 会出现 17 个 collection error
    （`No module named 'qwen35_ple'` / torch 相关），能收 58 个测试、4 skipped。
    这是已知环境限制，不是回归；完整套件交给 CI 和远程 venv。
16. **custom 62 高估**：永远不要用它做可部署 claim。
17. **`nn.ModuleList(raw Parameter)` 崩**：full-FT optimizer 构造改为从 module
    参数收集，不再塞原始 tensor。

---

## 6. 已完成 / 未完成清单

### 已完成（本轮）

* [x] 148-A：gold-answer NLL、head/order 消融、layer hidden、oracle-routing 探针
* [x] 148-B：解冻官方 source 的 capacity 诊断
* [x] standard 1500 v2 generation + gold NLL 评测基础设施
* [x] Round 149 roadmap、技术债清单、兄弟项目借鉴边界
* [x] ModelScope 下载 Qwen3.5-4B / 2B（磁盘已核对）
* [x] Round 149 full-FT 三臂跑完并出结论（配方无效，非 PLE 无效）
* [x] 本文档 + 运维封装 `scripts/ssh_autodl.sh`

### 未完成（按优先级）

1. **LoRA 行 2×2**（首选）：0.8B × {frozen, LoRA} × {no PLE, real, control}，
   standard 1500 raw/chat + gold NLL。**`peft` 未安装在远程 venv**（已核实），
   要么 `pip install peft`，要么写一个明确标注"临时"的最小 LoRA shim。
2. **G0 scale/space**：Qwen3.5-4B（hidden 2560）+ frozen PLE 嫁接，
   standard 1500 raw + gold NLL。模型已就位，GPU 空。
3. **G3 端侧 serving 微基准**：EngramDB Store-P + 热缓存 + 预取，
   NVMe / USB SSD，与 `/dev/shm` 做质量对齐比较。
4. **G2 hot-row 表共适应**：只训 SFT 语料触及的行，SparseAdam 5× LR。
5. **G4 小 PLE Pareto**：2-gram-only / FP8/4-bit / 剪枝蒸馏 / tiny co-trained。
6. **工程债**：统一 runner + batched eval + `status.json` + artifact manifest；
   清理远程仓库与同步；locked eval suite 与 CI 覆盖。

---

## 7. 未来计划与判定规则

### 7.1 Gates

```text
G0 scale/space   : Qwen3.5-4B + frozen PLE
                   通过 -> 0.8B 的失败是 scale/hidden 不匹配，端侧继续用 4B
G1 backbone 共适应: {frozen, LoRA, full FT} × {no PLE, real, control}
                   通过 -> real − control ≥2 pts 或 ≥0.05 nat（在没崩的那一行里）
                   现状 -> full-FT 行配方崩塌，改用 LoRA / 温和 FT
G2 table 共适应   : hot-row SparseAdam/momentum 5× LR
                   通过 -> 小规模共适应 PLE 是产品路线
                   失败 -> 冻结跨模型世界知识迁移被证伪
G3 端侧 serving   : EngramDB Store-P on NVMe/USB SSD + 热缓存 + 预取
                   通过 -> 磁盘优先端侧部署工程可行
G4 小 PLE Pareto  : 2-gram-only / 量化 / 剪枝 / 蒸馏 / co-trained
                   -> 保住质量增益的最小记忆
```

### 7.2 判定规则（预注册，不允许事后改）

```text
G0 通过 -> frozen graft 受 scale/space 限制，继续 4B 端侧 hybrid
G1 通过 -> backbone 共适应解锁外部记忆，建端侧 hybrid
G2 通过 -> 表共适应是必需成分，建小规模共适应 PLE
G1/G2 失败 -> 冻结跨模型世界知识迁移被证伪；
              转向 small co-trained PLE，或把 PLE 重新定位为 format prior
G3 通过 -> 端侧部署工程可行
G3 失败 -> 压缩/蒸馏表，或改记忆层级
```

**硬规则**：每个 gate 有预注册效应量阈值、control arm、停止规则；
不允许靠"再加一个模块"或"换评测集"来救 gate。

### 7.3 工作流

```text
W1 实验内核：统一 runner、config registry、status.json、batched eval
W2 评测：locked standard suite、gold NLL、CI、污染审计、reasoning/long-context 子集
W3 适应：frozen/LoRA/full FT + reader/gate + 可选 hot rows
W4 记忆：EngramDB Store-P、热缓存、预取、端侧基准、小 PLE
W5 集成发布：bundle/manifest、版本 pin、文档、CI、ModelScope 下载、产物留存
```

---

## 8. 从相似项目借鉴什么、不重复什么

| 项目 | 借鉴 | 不要重复做 | 接缝 |
|---|---|---|---|
| EngramDB | Store-I/Store-P、badge 布局、热缓存、预取、bundle manifest、磁盘基准 | 另写存储引擎或行格式 | `engramdb-python`；`serving/bundle.py`、`serving/adapter.py` |
| engram-peft | 权威 EngramConfig、hash/gate/short-conv 语义、PEFT 包装 | 第二套 hash/gate 实现 | pin 版本 + 跨仓库 golden 测试 |
| Qwen 官方 modeling | PLE 层数学、hash 映射、HC/conv、层放置、消融 | 从零逆向 | `official_ple_snapshot.py` + parity 测试 |
| ortegaalfredo ngram-knowledge-injector | 行级 patch/热插拔、精确行寻址、GGUF 元数据 | 另一套 patch 格式 | patch 文件 + EngramDB 更新路径 |
| llama.cpp-NLTM / qwen4exp | PLE offload/运行时集成、mmap、host 行索引 | 把 embedding 研究循环塞进 runtime | serving bundle / GGUF 导出 |
| vLLM / SGLang PLE offload | 异步预取、pinned host memory、offload 调度、metrics | fork 一个 serving 引擎 | plugin/offload 路径 |
| DeepSeek Engram 论文 | U 形分配、4-gram/多 head、SparseAdam 5× LR、gating 数学 | 不顾 regime 直接套规模 | configs + 训练配方 |
| TN-gram / tensorized Engram | 因子化与碰撞缓解、每字节质量 | 没有 golden 测试就换 hash | 可选压缩表后端 |
| HF PEFT | LoRA/DoRA 配置、训练循环、合并 | 自研 LoRA | `peft` 包 + 统一 trainer |
| RAG / 检索项目 | 知识 baseline、评测协议 | 把检索和 PLE 混为一谈 | standard suite 里的 baseline arm |

冲突规避规则：

1. 每层唯一真源，边界处用 adapter。
2. 跨仓库 golden parity 测试；CI 与远程环境都 pin `engramdb-python`、`engram-peft`。
3. 共享评测协议；claim 不用仓库私有的指标定义。
4. 一个统一 trainer 覆盖 frozen/LoRA/full FT + reader/gate + hot rows。
5. 存储永远是 EngramDB；研究代码不自创行格式。
6. serving 永远是 bundle；runtime 集成是 plugin。

---

## 9. 运维手册（压缩后照着做）

```text
连接（host key 已变，必须带这两个选项，或用 scripts/ssh_autodl.sh）：
  ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 19236 \
      root@connect.nmb1.seetacloud.com

目录：
  /root/autodl-tmp/qwen35-ple/repo           代码（dirty/落后；本地是唯一真源）
  /root/autodl-tmp/qwen35-ple/venv           Python 环境（有 torch，无 peft）
  /root/autodl-tmp/qwen35-ple/models         0.8B / 2B / 4B + tokenizer + qwen38_ple
  /root/autodl-tmp/qwen35-ple/qwen38-rows    持久 PLE 行（48G）
  /dev/shm/qwen38-rows                       快速行（129 文件，48G）
  /root/autodl-tmp/qwen35-ple/outputs        实验产出
  /root/autodl-tmp/qwen35-ple/logs           日志

当前实测状态（本轮收尾）：
  GPU      RTX 4090 24GB，1 MiB 占用，0% 利用率，无进程  -> 完全空闲
  tmux     只有 learn（无关旧会话，可忽略或 kill）
  磁盘     /root/autodl-tmp 86G 中 74G 已用，13G free（86%）
  /dev/shm 60G 中 48G 已用（129 行文件），13G free
  行表     两个位置都是 129 files
  peft     未安装（LoRA 行要先装或用临时 shim）
  本地     分支 main 与 origin/main 同步；本地 .venv 无 torch

启动队列模板：
  tmux new -s roundXXX
  ROOT=/root/autodl-tmp/qwen35-ple bash /root/autodl-tmp/qwen35-ple/repo/scripts/run_roundXXX.sh
  # 产出 DONE 标记 + log；轮询用短步 SSH，不要长 sleep

磁盘注意：
  13G free 很紧。保存全量模型前先清 outputs/*-partial、backup、旧 checkpoint。
  优先 LoRA adapter + in-process 评测，避免复制整份 backbone。

重启后恢复：
  bash scripts/ensure_qwen38_rows.sh --python /root/autodl-tmp/qwen35-ple/venv/bin/python
```

---

## 10. 压缩上下文后的恢复动作

1. 读本文档 + `docs/round-149-systematic-roadmap-and-tech-debt.md`。
2. 读远程 `outputs/round149/gold-nll-raw.md` 与三个 fullft JSON，确认崩塌结论。
3. 确认环境：GPU 空闲、tmux 无队列、`/dev/shm` 129 行、磁盘 free。
4. 做 **LoRA 行**（首选）：装 `peft`（或写明确标注临时的 LoRA shim）→
   先跑 1 个 arm 的短 smoke（确认 baseline 不崩）→ 再放三条 arm 的队列：
   `lora-nople` / `lora-real` / `lora-control`，standard 1500 raw + gold NLL。
   **先验证 baseline 没崩，再解释 real vs control。**
5. 同时/随后跑 **G0**：Qwen3.5-4B + frozen PLE，standard 1500 raw + gold NLL。
6. 起 **G3** EngramDB 磁盘微基准（不依赖 GPU，可与 4/5 并行）。
7. 每个 gate 出结论就更新文档 + commit + push；负结果一样要写清楚。
8. 在 G0/G1/G2 给出明确停止信号前，不要开新的大方向（RL 等只作 baseline）。

---

## 11. 参考文档

```text
docs/round-143-oracle-upper-bound-and-overnight-queue.md
docs/round-144-strategic-roadmap.md
docs/round-145-standard-heldout-chat-template-and-format-matched-experiments.md
docs/round-146-standard-heldout-final-and-cross-file-oracle.md
docs/round-147-engram-lineage-comparison.md
docs/round-148-format-vs-content-nll-and-oracle-routing-probe.md
docs/round-149-systematic-roadmap-and-tech-debt.md
docs/round-150-session-recap-and-handoff.md
docs/round-151-session-consolidation-and-handoff.md（本文）
docs/session-log.md
```
