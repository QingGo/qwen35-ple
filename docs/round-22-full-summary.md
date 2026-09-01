# Round 22 全量总结：1M QA、扩大评测集、serving 探索与工程收口

> 日期：2026-09-01
> 范围：从“1M QA exact-match”到“扩大 QA 集”“vLLM/SGLang 可行性”“qwen35-ple 后续开发规划”
> 状态：科学正信号继续；工程债仍集中在 reader checkpoint / serving / 大样本 QA

---

## 1. 本轮目标

1. 完成 1M token 的 QA exact-match（TriviaQA / NQ / BoolQ、real / control / no-reader、3 seeds）。
2. 把 QA 集扩大到每任务至少 50 题。
3. 梳理 vLLM / SGLang 是否可用于加速推理。
4. 明确 qwen35-ple 与 EngramDB / engram-peft 的分工。
5. 制定后续更稳、更接近终极目标的开发计划。

---

## 2. 终极目标（回顾）

> 证明“冻结的 Qwen3.8-Flash-Next PLE 记忆表能否通过 target-side reader 嫁接到更小的 Qwen3.5 模型上”。

如果成立，最终交付：

```text
更强的 Qwen3.5 知识/长上下文模型
+ 可复现训练/评测体系
+ CPU 100 tok/s 推理闭环
```

---

## 3. 本轮关键发现

### 3.1 1M PPL 结果保持强正信号

| 线 | val loss | PPL |
|---|---:|---:|
| no-reader | 2.9896 | 19.88 |
| control | 2.8738 | 17.70 |
| **real** | **2.8167** | **16.72** |

```text
real − control = −0.0571
real − no-reader = −0.1729
```

3 seeds 全部 real > control。

### 3.2 1M QA exact-match（9题）

| 线 | QA EM mean | 3 seeds |
|---|---:|---|
| no-reader | 44.44% | 44.4 / 44.4 / 44.4 |
| control | 48.15% | 44.4 / 33.3 / 66.7 |
| **real** | **51.85%** | 66.7 / 44.4 / 44.4 |

```text
real − control = +3.70pp
real − no-reader = +7.41pp
```

分项：

| task | no-reader | control | real |
|---|---:|---:|---:|
| TriviaQA | 100% | 77.8% | 100% |
| NQ | 33.3% | 55.6% | 55.6% |
| BoolQ | 0% | 11.1% | 0% |

解读：

- QA 也出现 real > control > no-reader 的平均排序。
- 但 9 题样本太小，种子波动大。
- seed 2 上 control 反超 real。
- 不能单独作为决定性证据。

### 3.3 扩大 QA 集已生成

```text
assets/qa-expanded-150.json
50 TriviaQA-style + 50 NQ + 50 BoolQ
```

- NQ 来自 NQ-open dev 子集。
- BoolQ 来自 BoolQ dev 子集，带 passage 上下文。
- TriviaQA 当前为人工整理的 TriviaQA-style 集合。

### 3.4 推理现状

- 当前 exact-match 使用 Transformers 手动逐 token forward。
- 未使用 vLLM / SGLang。
- 未使用 KV cache / continuous batching。
- 150 题规模会非常慢。

### 3.5 EngramDB 已有 vLLM / SGLang 适配

EngramDB 仓库已存在：

```text
engramdb/vllm_plugin.py
engramdb/sglang.py
engramdb/vllm.py
```

但它们主要面向：

```text
源模型 PLE embedding 表 → DiskPleEmbedding 替换
```

不能直接覆盖我们的 target-side reader：

```text
Qwen3.5 backbone + OfficialSourceQwenReader + 每步 rowids→e_t → layer 8 注入
```

---

## 4. 本轮做的尝试

### 4.1 1M QA exact-match

- 在 `run_phase0.py` 中新增：
  - `--qa-exact-match`
  - `--qa-max-new-tokens`
  - `--qa-file`
- 实现：
  - real/control：每步实时从 EngramDB 取当前 token 序列的 `e_t`
  - no-reader：同一条贪心生成循环，不注入 PLE
- 增加 normalized exact-match 评分。

### 4.2 WSL 实机运行

- 通过内网访问：
  ```text
  Mac → 192.168.31.108（Windows DESKTOP-VI1IC4Q）→ WSL Ubuntu
  ```
- 使用 Windows scheduled task 在后台运行长任务。
- 完成 9 题 1M QA。
- 启动 150 题扩大版 QA。
- 150 题任务因 Windows 关机中断。

### 4.3 扩大 QA 数据构建

- 从 GitHub raw 获取：
  - NQ-open.dev.jsonl
  - BoolQ dev.jsonl
- 手工整理 50 条 TriviaQA-style。

### 4.4 vLLM / SGLang 可行性调研

- 确认 vLLM、SGLang 均已支持 Qwen3.5。
- 确认 EngramDB 已有 vLLM / SGLang 插件原型。
- 确认 target-side reader 需要自定义 serving 适配，不能直接用现有 source-side 插件。

---

## 5. 踩过的坑

| # | 坑 | 后果/解决 |
|---|---|---|
| 1 | 当前环境不能直接访问 WSL | 实际可通过内网 `192.168.31.108` 访问 |
| 2 | Windows SSH 默认 shell 不是 bash | 需要通过 `wsl -d Ubuntu -- bash -lc` |
| 3 | `nohup` 后台任务在 SSH 断开后可能不保留 | 改用 Windows scheduled task |
| 4 | 调度任务运行后 `/tmp` 日志会丢失 | 日志应写到仓库 `logs/` 或持久路径 |
| 5 | WSL `run_phase0.py` 可能被其他工作覆盖 | 需要重新同步文件并保留备份 |
| 6 | 150 题扩大版 QA 中途 Windows 关机 | 任务中断，暂无结果 |
| 7 | 当前没有训练后 reader checkpoint | 换 QA 集需重训 1M |
| 8 | stdout 重定向后 Python 块缓冲 | 日志更新滞后 |
| 9 | `--qa-file` 只影响 exact-match，不影响 log-likelihood | 需要明确两套 QA 口径 |
| 10 | EngramDB 的 vLLM/SGLang 插件不是 target-side reader 插件 | 需要 qwen35-ple 侧自建适配 |

---

## 6. 已完成内容

- [x] `run_phase0.py` 支持 QA exact-match 生成式评测。
- [x] 1M token 9题 QA exact-match 三线结果。
- [x] 修复 `_split` 在非 live-store 模式下误判 numpy ndarray 的问题。
- [x] 修复新版 Transformers 下模型 dtype 不一致问题。
- [x] 扩大 QA 集：
  - `assets/qa-expanded-150.json`
  - 50 TriviaQA-style + 50 NQ + 50 BoolQ。
- [x] 探索 vLLM/SGLang 与 EngramDB 适配现状。
- [x] 明确 qwen35-ple / EngramDB / engram-peft 分工。
- [x] README 增加“推理 / Serving 现状与规划”。
- [x] 提交并推送 qwen35-ple 当前变更。

---

## 7. 未完成内容 / 技术债

| # | 技术债 | 影响 |
|---|---|---|
| 1 | 没有 reader checkpoint 保存/加载 | 换 QA 集、做 serving 都要重训 |
| 2 | 150 题扩大版 QA 未跑完 | 缺少放大样本证据 |
| 3 | 无 reader version registry | 结构演进会破坏加载 |
| 4 | 无统一 serving bundle | vLLM/SGLang/LLM-CompileForge 无法统一消费 |
| 5 | target-side reader 的 vLLM/SGLang 插件未开发 | QA 推理慢 |
| 6 | 5M token 正式实验未跑 | 无法最终 Go/No-Go |
| 7 | reader + LoRA / partial unfreeze 未测 | 可能错过信号 |
| 8 | Qwen3.5-4B 未测 | 尚不能判断规模扩展性 |
| 9 | CPU 100 tok/s 未开始 | 产品目标未验证 |
| 10 | WSL 任务可恢复性不足 | 中断后需重跑 |
| 11 | corpus/manifest/checksum 不完整 | 复现和审计不足 |
| 12 | TriviaQA 不是官方标准集 | 后续需替换或补充为官方子集 |

---

## 8. 借鉴矩阵

| 项目 | 借什么 | 不拿什么 |
|---|---|---|
| XMemTransfer | 5M–20M 训练预算、target-side reader、dual-layer | 不拿记忆表/模型 |
| Qwen / Flash-Next | 官方 PLE 结构、FP8 scale、key/value/norm/conv | 不重训 51B 表 |
| DeepSeek Engram / engram-peft | EngramConfig、hash mapping、adapter 保存/加载、incremental token buffer | 不引入第二套存储 |
| Memory Grafting | 冻结离线记忆、轻量 projection/gating | 不复制记忆提取 |
| Prometheus Mind | 冻结模型忽略信号、stage-wise 解锁 | 不复制记忆构建 |
| EngramDB | Store-I/Store-P、磁盘 embedding、缓存/预取、C ABI、manifest | 不改存储核心 |
| vLLM / SGLang | 批量推理、KV cache、continuous batching | 不成为科学验证核心 |
| LLM-CompileForge | CPU 100 tok/s 编译/运行时 | 不替代训练 |
| 标准评测 | TriviaQA / NQ / BoolQ / OpenBookQA / WikiText | 不追榜单 |

**分工原则：**

```text
EngramDB        → 固定存储
engram-peft     → 训练/适配设施
qwen35-ple      → 科学验证、reader 设计、编排
vLLM/SGLang    → GPU 推理加速/迭代
LLM-CompileForge → CPU 产品推理
```

---

## 9. 后续开发计划

### 第一阶段：工程收口（最高优先）

1. `run_phase0.py` 增加：
   - `--save-reader`
   - `--load-reader`
2. 建立：
   ```text
   qwen35_ple/reader_registry.py
   qwen35_ple/serving/bundle.py
   ```
3. 支持从已训练 reader 直接跑 QA，不再重训。
4. 150 题 QA 正式跑。
5. WSL 任务可恢复：
   - 日志写到 `logs/`
   - 记录启动命令、seed、环境、manifest。

### 第二阶段：科学确认

1. 150 题 QA 三线结果。
2. reader 变体：
   - dual-layer
   - multi-layer
   - reader + LoRA
   - reader + partial unfreeze
3. 5M 正式矩阵：
   ```text
   5M tokens
   3 seeds
   real / control / no-reader
   PPL + 150题 QA
   ```
4. Go / No-Go：
   ```text
   Go：
     3 seeds real > control
     且 PPL 或 QA 至少一项 real > no-reader

   No-Go：
     5M 下 real 无法稳定超过 control
     或两者都未超过 no-reader
   ```

### 第三阶段：产品化

1. Qwen3.5-4B。
2. target-side reader 的 vLLM / SGLang 自定义适配。
3. LLM-CompileForge CPU 100 tok/s。
4. SFT / RL。

---

## 10. 关键提交

```text
c365151 docs: add serving/vLLM/SGLang findings and expanded QA status
0e600fa data: add expanded QA set (50 TriviaQA-style / 50 NQ / 50 BoolQ)
cb89b0e docs: add 1M QA exact-match results from WSL run
a2d0609 docs: add serving/vLLM/SGLang findings and expanded QA status
65b1024 data: add expanded QA set (50 TriviaQA-style / 50 NQ / 50 BoolQ)
181fc0a docs: add 1M QA exact-match results from WSL run
```

---

## 11. 当前结论

```text
PPL 强正信号：real < control < no-reader，稳定
QA 方向性正信号：real > control > no-reader，但样本不足
150 题结果：未完成（Windows 关机中断）
5M 正式判定：未开始

下一步最高优先：
  保存/加载 reader checkpoint
  → 跑完 150 题 QA
  → 决定是否进入 5M
```
