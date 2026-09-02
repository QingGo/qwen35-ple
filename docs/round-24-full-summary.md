# Round 24 全量总结：系统性思考、上游升级、三线 QA、语料混比与后续路线

> 日期：2026-09-02
> 范围：engram-peft 1.2.7 / EngramDB 0.2.12 升级、reader/bundle/serving 工程落地、1M 150 题三线 QA、bad case 分析、语料混比调研
> 状态：工程闭环已打通；科学结论仍为“PLE 有信号但当前未跑赢无 PLE 基线”

---

## 1. 本轮系统性思考：终极目标是什么

我们的北极星没有变：

> **用最小可复现实验证明“Qwen3.5 主干 + 冻结 Flash-Next PLE 记忆表”的嫁接是否成立；**
> 若成立，交付 0.8B 后训练模型 + CPU 100 tok/s 可复现推理闭环；
> 若不成立，留下可审计的负结果。

展开为四条验收轴：

| 轴 | 当前状态 |
|---|---|
| 科学 | ⚠️ PLE 有正信号（real > control），但 150 QA 上未跑赢 no-reader |
| 工程 | ✅ reader checkpoint / bundle / serving adapter / 三线评测闭环已打通 |
| 产品 | ⏳ 真实 vLLM/SGLang 与 CPU 100 tok/s 尚未验证 |
| 过程 | ⚠️ 单 seed、无严格语料污染审计，仍需加固 |

本轮最重要的判断：

```text
val loss：real < control < no-reader
150 QA：no-reader > real > control
```

这说明：

- PLE 确实降低了语言建模 loss；
- 但当前 reader 还没有把这种能力转化为问答/生成任务上的净收益；
- 最可能的原因不是“PLE 没有用”，而是：
  1. reader 没有在 Qwen3.5 任务格式 / 对话 / QA / CoT / tool / agent 分布上训练；
  2. 当前 1M 语料是杂乱网页文本，与评测分布不匹配；
  3. exact-match 对生成格式、数字、截断过于敏感，放大了格式干扰。

---

## 2. 本轮计划

1. 确认 engram-peft 1.2.7 / EngramDB 0.2.12 是否需要升级。
2. 完成升级收口：pyproject / uv.lock / CI / WSL 脚本。
3. 把 reader registry、bundle、serving adapter、reader checkpoint 工程落地。
4. 在 WSL 保存 1M real reader，并用 `--load-reader` 跑 150 题 QA。
5. 补跑 150 题 no-reader / control，完成三线对比。
6. 分析各自 good case / bad case 与语料重叠。
7. 调研可收集的高质量语料和 1M token 混比实验方案。

---

## 3. 关键发现

### 3.1 三线 150 QA 结果

| 线 | QA EM | TriviaQA | NQ | BoolQ | val loss | PPL |
|---|---:|---:|---:|---:|---:|---:|
| no-reader | **53.3%** | 70.0% | 0.0% | **90.0%** | 2.9895 | 19.88 |
| real | 42.0% | 66.0% | 0.0% | 60.0% | **2.7892** | **16.27** |
| control | 30.7% | 62.0% | 2.0% | 28.0% | 2.9159 | 18.46 |

### 3.2 逐题 new correct / new wrong

- real vs no-reader：
  - real 新做对 9 题；
  - real 新做错 26 题。
- real vs control：
  - real 新做对 26 题；
  - real 新做错 9 题。

### 3.3 no-reader 为什么做对更多

主要原因：

1. **回答格式 / 截断**
   - real 生成更“解释性/思考性”，答案词常被推迟到 16 token 外；
   - no-reader 更直接，容易命中严格 exact-match。
2. **数字 / 同义词差异**
   - `six` vs `6` 这类差异导致 real 被误判。
3. **BoolQ 稳定性下降**
   - no-reader BoolQ 90%，real 降到 60%；
   - PLE 注入在简单 passage 判断上产生干扰。
4. **少量真实知识错误**
   - 例如 fastest land animal 被 real 给成 Giant Panda。
5. **训练语料与评测格式不匹配**
   - 当前 1M 语料基本是网页/直播/新闻等杂乱混合文本；
   - 没有 Qwen3.5 chat template、CoT、tool call、agent trajectory。

### 3.4 语料重叠检查

- real 相对 no-reader 新做对的 TriviaQA 中：
  - Newton、Shakespeare、Saturn 的答案 **不在 1M 语料中**；
  - 这是最接近“语义对齐而非记忆新知识”的正信号。
- 但简单答案字符串重叠噪声较大，需要更严格的“完整事实句”审计。

### 3.5 语料混比调研结论

- 对“外部记忆表 + target-side reader”场景，训练语料的作用不是注入新知识，而是：
  - 教 reader 如何在目标任务格式下访问 PLE；
  - 保持基座通用能力；
  - 提供高质量实体 / 知识表达做语义对齐。
- 建议混入：
  - 通用高质量文本；
  - Qwen3.5 chat / QA / instruction；
  - Wikipedia / 百科；
  - CoT / tool / agent 轨迹。
- 不能只堆 Wikipedia，否则无法区分“PLE 对齐”和“模型背题”。
- 必须做评测污染审计和 held-out 控制。

---

## 4. 这一轮做了什么

### 4.1 版本与工程收口

- `engram-peft>=1.2.7`
- `engramdb-python>=0.2.12`
- `uv.lock`、CI tag、WSL 脚本同步
- 新增：
  - `src/qwen35_ple/reader_registry.py`
  - `src/qwen35_ple/serving/bundle.py`
  - `src/qwen35_ple/serving/adapter.py`
  - `scripts/analyze_qa_lines.py`

### 4.2 reader / bundle / serving 能力

- `TargetReaderRegistry` 薄封装，注册 official / engram / simple reader
- `save_reader` / `load_reader` / `load_reader_with_extra`
- `ShortConv` 状态并入 checkpoint
- Bundle manifest 生成 / 加载
- `QwenReaderServingAdapter` + vLLM/SGLang 风格别名
- `run_phase0.py` 新增：
  - `--save-reader`
  - `--load-reader`
  - `--save-bundle`

### 4.3 WSL 实际操作

- 通过 Tailscale 直连 Windows：`100.78.250.122`
- 长任务使用 Windows Scheduled Task，避免 SSH 断开被杀
- 保存：
  - `outputs/reader-real-seed0.pt`
  - `outputs/bundle-real-seed0.json`
  - `outputs/reader-control-seed0.pt`
  - `outputs/bundle-control-seed0.json`
- 跑出：
  - `outputs/phase0-live1m-qa150-loaded.json`
  - `outputs/phase0-live1m-qa150-noreader.json`
  - `outputs/phase0-live1m-qa150-control.json`

### 4.4 分析文档

- `docs/round-23-upgrade-assessment.md`
- `docs/phase0-live1m-qa150-analysis.md`
- `docs/round-24-full-summary.md`

---

## 5. 踩过的坑

| # | 坑 | 解决 / 状态 |
|---|---|---|
| 1 | 本地 Mac transformers 不认识 `qwen3_5` | 使用 WSL 上的 transformers 5.16.1 |
| 2 | SSH 直连 `192.168.31.108` 不通 | 改用 Tailscale IP `100.78.250.122` |
| 3 | Windows 默认 shell 是 cmd，不是 bash | 用 `wsl -d Ubuntu -- bash -lc` 包一层 |
| 4 | 直接 `nohup ... &` 在 wsl 会话退出后进程被杀 | 改用 Windows Scheduled Task 托管 |
| 5 | 远端 engramdb 只有 0.2.9，缺少 TargetReaderRegistry | 本地增加 `_LocalRegistry` fallback |
| 6 | scp 到 `C:/Users/minam/` 后还需手动复制进 WSL | 已用 `wsl cp /mnt/c/...` 同步 |
| 7 | Windows cmd / wsl 嵌套引号经常出错 | 长命令写成 `.sh` 脚本再交给 wsl 执行 |
| 8 | QA 长任务没有逐题 progress | 尚未加日志；后续应在 `_qa_exact_match` 加 per-question print |
| 9 | exact-match 对 `six` / `6`、截断、格式敏感 | 后续要改进归一化，或改用更鲁棒的打分 |
| 10 | 科学结论只有单 seed | 需要补 3 seeds |

---

## 6. 本轮完成 / 未完成

### 已完成后

- [x] 依赖与版本收口
- [x] reader registry / bundle / serving adapter
- [x] reader checkpoint 保存 / 加载 / bundle 生成
- [x] WSL 1M real reader 保存
- [x] 150 题三线 QA（real / no-reader / control）
- [x] good case / bad case / 语料重叠分析
- [x] 语料混比调研
- [x] 分析脚本与文档

### 未完成

- [ ] 3 seeds 三线复跑
- [ ] 严格语料污染审计
- [ ] Qwen3.5 chat / CoT / tool / agent 格式语料构建
- [ ] 1M token 混比矩阵实验（M1–M5）
- [ ] 改进 exact-match 归一化 / 增加逐题日志
- [ ] 真实 vLLM / SGLang serving A/B
- [ ] 直接 WSL SSH 环境配置
- [ ] 5M–20M 正式实验
- [ ] CPU 100 tok/s 闭环

---

## 7. 技术债清单

| 编号 | 技术债 | 影响 | 处置方向 |
|---|---|---|---|
| V166 | 150 QA 上 no-reader > real | 当前科学结论不能 Go | 修任务格式语料 / reader 干扰 / 生成格式 |
| V167 | 当前语料不是 Qwen3.5 任务格式 | reader 没有在目标分布上被激活 | 构建 chat/CoT/tool/agent 混合语料 |
| V168 | exact-match 对格式/数字/截断敏感 | 可能低估 real | 改进归一化 / 增加宽松匹配 |
| V169 | 只有 single seed | 结论不稳定 | 最优 mix 跑 3 seeds |
| V170 | 无受控混比实验 | 不知道最优语料配比 | M1–M5 1M token 混比矩阵 |
| V171 | 无严格污染审计 | 无法区分 PLE 对齐与背题 | 建设 n-gram/答案剔除与审计 |
| V172 | 缺 CoT/tool/agent 评测集 | 无法验证目标能力 | 增加小规模评估集 |
| V173 | 长任务依赖 Windows Scheduled Task | 复现/自动化弱 | 写 WSL 帮助脚本 + 可复现 manifest |
| V174 | 未配置直接 WSL SSH | 操作链路仍绕 Windows | 安装 sshd + portproxy 或 Tailscale |
| V175 | serving adapter 未在真实 vLLM/SGLang 验证 | 产品目标未闭环 | 真实引擎 A/B |
| V176 | `_LocalRegistry` 与 canonical 并存 | 长期双实现风险 | 远端升级 engramdb 0.2.12 后收敛到 canonical |
| V177 | 没有统一 1M mix builder | 实验不可复现 | 写 `build_mix.py` + manifest |
| V178 | NQ 三线接近 0 | 当前方法对开放域无效 | 探索 reader 结构 / 更大训练量 / NQ 专用数据 |
| V179 | val PPL 正收益与 QA 负收益不一致 | 需要归因 | 分开评估 PPL 与生成式能力 |
| V180 | 缺少逐题进度日志 | 长任务不可观测 | `_qa_exact_match` 增加 per-item 输出 |

---

## 8. 借鉴矩阵

| 来源 | 借什么 | 不拿什么 | 为什么能共存 |
|---|---|---|---|
| **XMemTransfer** | 5M–20M 目标侧训练量；target-side reader；先做 5M 再判断 | 不照搬表/模型 | 验证量级与训练预算 |
| **Memory Grafting** | 外部冻结记忆与训练语料解耦；轻量 projection/gating | 不复制记忆提取 | 支持“语义对齐而非背题” |
| **DeepSeek Engram** | 条件记忆、稀疏检索、门控/ShortConv、训练基础设施 | 不引入第二套存储 | 直接复用 engram-peft |
| **EngramDB** | 证据库、位级一致、manifest、可重建资产 | 不改存储核心 | 作为底层事实源 |
| **数据混合定律 / CMR** | 最优混比可建模；持续预训练需通用数据防遗忘 | 不照搬单一比例 | 指导 M1–M5 实验设计 |
| **ModelScope / SWIFT 生态** | 国内快速下载、现成 instruction/agent 数据、统一处理管线 | 不绑定某个训练框架 | 快速构建语料 |
| **RAG / 检索增强** | 训练时使用知识型文本、评测时隔离知识源 | 不把它变成外部检索系统 | 我们本质是“记忆层 + reader” |
| **Benchmark 污染研究** | 报告 train-test overlap、retro-holdout、n-gram 审计 | 不因此不做知识评测 | 保证科学结论可信 |
| **vLLM / SGLang** | serving 薄适配、批量推理、KV cache | 不复制推理引擎 | 用于产品化验证 |
| **LLM-CompileForge** | CPU 100 tok/s 编译/runtime、C ABI 对接 | 不替代科学验证 | 产品目标 |

---

## 9. 下一阶段开发计划

### Phase A：语料与混比实验（最高优先）

1. 从 ModelScope 下载分片：
   - `AI-ModelScope/fineweb-edu` / `chinese-fineweb-edu`
   - `AI-ModelScope/Magpie-Qwen2-Pro-200K-English` 或 `wyj123456/instruct`
   - 中文维基 / `wikitext` 分片
   - `open-r1/OpenR1-Math-220k` 小采样
   - `damo/MSAgent-Bench` 或 `mlabonne/orca-agentinstruct-1M-v1-cleaned` 小采样
2. 写 `build_mix.py`：
   - 每类 tokenize；
   - 按比例采样；
   - 输出 1M token 实验包；
   - 生成 manifest + 污染审计。
3. 跑 M1–M5：
   - M1：50% 通用 / 20% chat / 20% wiki / 10% CoT+tool
   - M2：40% 通用 / 30% chat / 20% wiki / 10% CoT+tool
   - M3：30% 通用 / 40% chat / 20% wiki / 10% CoT+tool
   - M4：30% 通用 / 30% chat / 30% wiki / 10% CoT+tool
   - M5：20% 通用 / 40% chat / 20% wiki / 20% CoT+tool
4. 每个 mix 跑：
   - real / no-reader / control；
   - 150 QA；val loss；BoolQ 分项。

### Phase B：工程加固

1. `_qa_exact_match` 增加 per-question progress。
2. 改进 exact-match：
   - 数字归一化（`6` ↔ `six`）；
   - 可选宽松匹配；
   - 保留严格版本做对照。
3. 长任务脚本化：
   - `scripts/wsl_task.sh`
   - `scripts/build_mix.py`
   - `scripts/run_mix_batch.sh`
4. 将远端 engramdb 升级到 0.2.12，统一 canonical registry。

### Phase C：科学确认

1. 如果某一 mix 的 real 稳定超过 no-reader 且超过 control：
   - 跑 3 seeds；
   - 检查新做对题目是否仍在 held-out 外；
   - 再做 5M–20M。
2. 如果所有 mix 都未超过 no-reader：
   - 记录系统性负结果；
   - 停止放大；
   - 保留文档与资产。

### Phase D：产品化

1. 真实 vLLM / SGLang 接入 `QwenReaderServingAdapter`。
2. serving A/B。
3. 与 LLM-CompileForge 的 CPU 100 tok/s 目标对接。

---

## 10. 本轮纪律

1. **科学结论必须三线齐全**：real / control / no-reader。
2. **必须审计污染**：训练语料不得包含评测题/答案/passage 原文。
3. **不因 val PPL 正收益就宣称为成功**：必须看任务级净收益。
4. **不因 single seed 就下结论**：至少 3 seeds。
5. **所有实验保留 manifest**：语料来源、比例、token 数、污染检查、命令。
6. **长任务必须可恢复**：使用 Scheduled Task/脚本 + 日志 + pid。
7. **路线优先级**：先证明科学，再产品化；不要抢跑 100 tok/s。
