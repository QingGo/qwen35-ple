# Round 156: G0-nople 结果 —— graft 的主导效应是「格式」而非「知识」，外加两个指标 bug

> 承接 `docs/round-155-g0-contrast-is-a-decoding-artifact.md` §7 的预注册预测。

---

## 0. 摘要

1. `g0-nople`（同 real reader + `--ple-off`，注入恒为零）已跑完，**预注册的方向性预测成立**：
   它**不**崩塌，而 control 崩塌 —— 证实 G0 那 +10 点 EM 测的是 control 的损伤，不是 PLE 的知识。
2. 但它同时揭示了一个更深的事实：**注入 PLE（无论 real 还是乱序行）的主导效应是把 4B 拉进
   raw 续写格式；不注入时 4B 根本不在那个格式里** —— 它吐 chat/thinking 脚手架。
3. 因此 G0 的三个臂**输出格式互不相同**，而现有两个指标都无法跨格式比较：
   * **子串匹配 EM 会虚高脚手架臂最多 33 个点**（`Nouser` 含子串 `no`）；
   * **gold NLL 在测一个模型不会吐出的 token**（`'yes'`=9405 vs 模型实际吐 `' yes'`=9542）。
4. 结论：**G0 不构成"冻结跨模型 PLE 迁移带来通用能力提升"的证据**；它测到的是格式/协议效应。
5. **自动关机没有生效**，根因是 tmux session 拆除时 SIGHUP 杀掉后台 finisher；已修复（§5）。

---

## 1. 三个臂的完整结果（standard 1500, raw protocol）

| Run | BoolQ | TriviaQA | NQ | Mean EM（报告值） | Gold NLL |
|---|---:|---:|---:|---:|---:|
| g0-real | 0.874 | 0.390 | 0.112 | 0.4587 | 4.1624 |
| g0-control | 0.814 | 0.178 | 0.084 | 0.3587 | 4.5569 |
| **g0-nople** | **0.880** | **0.404** | **0.152** | **0.4787** | **8.4737** |
| frozen-0.8b-no-reader（参照） | — | — | — | — | 8.7429 |

配对（n=1500）：
* g0-real vs g0-nople：EM **−0.0200 ± 0.0100**（real 更差）；NLL **−4.3113**（real 更好）
* g0-real vs g0-control：EM +0.1003 ± 0.0098；NLL +0.3945

**两个指标的排序完全相反**，这是必须先解决矛盾再谈结论的信号。

---

## 2. 决定性观察：三个臂的输出格式根本不同

`g0-nople` 的 BoolQ 高频输出（原始字符串）：

```text
'\n\n<think>\n\n</think>\n\nNoUU\nU\nU\nU\nU\nU\nU\n'          x21
'\n\n<think>\n\n</think>\n\nNodata\n</think>\n\nNodata\n</think>\n\nNo'  x19
'\n\n<think>\n\n</think>\n\nYesuser\nQuestion: ...\nAnswer with one word, Yes or No:\n\n'  x2
```

而 `g0-real` / `g0-control` 直接吐 `' yes'` / `' no'`。

即：**零注入的 4B 不遵守 raw "Question: … Answer:" 协议**，它退回对话先验，输出
`<think>` 脚手架并开始幻觉新的对话轮次。**注入 PLE —— 无论内容对不对 —— 都会把它压回 raw 续写格式。**

这解释了 NLL 的反常：nople 的 gold NLL（8.47）几乎等于 0.8B 的（8.74），
一个在生成上远强于 0.8B 的 4B 不该如此。答案是这两个数都在测"模型是否处在 raw 续写模式"，
而这个模式本身主要由**有无注入**决定，与注入内容的真假关系不大。

> 这与 round-148 的 "format vs content 判决" 是同一主题的再现：**PLE 首先是一个格式先验。**

---

## 3. 指标 bug #1：子串匹配 EM 会虚高脚手架臂

`scripts/run_phase0.py:1274`：

```python
hit = _normalize_answer(item["answer"]) in _normalize_answer(generated_text)
```

`_normalize_answer` 去掉标点后，"Yes" 紧跟 chat 模板的 "user" 解码成 `Yesuser`，
归一化后 `yesuser` **包含子串 `yes`** → 判为答对。`Nouser` 同理。

重算（同一批 1500 条）：

| arm | 报告值 | 子串匹配 | 词序列匹配 | 严格完全匹配 | 虚高 |
|---|---:|---:|---:|---:|---:|
| g0-real | 0.4587 | 0.4580 | 0.3620 | 0.3860 | +0.096 |
| g0-control | 0.3587 | 0.3580 | 0.3147 | 0.3460 | +0.043 |
| **g0-nople** | **0.4787** | 0.4767 | **0.1427** | **0.0000** | **+0.334** |

逐任务词序列 EM：

| arm | BoolQ | TriviaQA | NQ | Mean |
|---|---:|---:|---:|---:|
| g0-real | 0.8540 | 0.2140 | 0.0180 | 0.3620 |
| g0-control | 0.8140 | 0.0920 | 0.0380 | 0.3147 |
| g0-nople | 0.1300 | 0.2360 | 0.0620 | 0.1427 |

**注意：这两种修法都不公平。** 子串匹配高估脚手架臂（+33 点），词序列匹配又低估它
（nople 确实吐出了正确的 `No` token，只是与 `user` 粘连）。**任何字符串级规则都无法跨格式比较。**

> **正确做法（待实现）**：按 **token 级**评分 —— 取生成序列的首个内容 token 与 gold 答案的
> token id 比较。字符串级评测在"不同臂输出不同格式"时必然失真。

---

## 4. 指标 bug #2：gold NLL 评的是模型不会吐的 token

`_qa_gold_nll` 的 docstring（`run_phase0.py:606-608`）明确写道：continuation 一律用
**原始答案分词**，"避免臂间的空格分词不一致"。

但 tokenizer 实测：

```text
'yes' -> [9405]      ' yes' -> [9542]
 'no' -> [2083]      ' no' -> [874]
```

**是两个不同的 token。** 模型在 `Answer:` 之后自然吐带空格的那个；评测却在给不带空格的
那个打 NLL。于是 BoolQ 这种**二选一**任务的 NLL 高达 6–14 nat（p≈10⁻³～10⁻⁶），
而同一批数据上报告的 EM 却有 0.88 —— **一个 88% 答对的模型不可能给正确答案 10⁻⁵ 的概率。**

结论：`--qa-gold-nll` 在 raw 协议下**主要测量空格 token 约定与格式模式，不能单独作为
知识或能力探针**。它跨臂仍可比（同一目标），但**绝对值无意义，且与 EM 冲突时不能默认信它**
（round-155 §6 第 3 条需要修正为："两者冲突时必须先排查分词/格式，而不是直接以 NLL 为准"）。

**修复方向**：把 continuation 用 `" " + answer` 编码（即模型实际会生成的形态）作为主口径，
并保留无空格口径作为对照；两者差异本身就是格式敏感度的度量。

---

## 5. 自动关机为什么没生效（用户提问的核对）

### 5.1 实验是否跑完：**是，全部跑完**

| 队列 | DONE 标记 | 臂 | 摘要 |
|---|---|---|---|
| `outputs/round152`（LoRA 行） | 01:15 | lora-nople/real/control | ✅ |
| `outputs/round152chat`（chat 复评） | 08:49 | lora-nople/real/control | ✅ |
| `outputs/round152g0`（4B + PLE） | 14:27 | g0-real/control/**nople** | ✅ |

`logs/round152-g0.log` 结尾：`=== [g0] G0 DONE 14:27:46 ===` / `started finisher 14:27:46` / `EXIT=0`。
无 `outputs/chain/FINISH_BLOCKED`。

### 5.2 自动关机是否生效：**没有**

证据：`logs/round152-finish.log` 的最后一行停在 **08:07:13**，
新 finisher 在 14:27:46 被启动，**却一行日志都没写**。

根因：g0 脚本用

```bash
bash "$ROOT/round152_finish.sh" >/dev/null 2>&1 &
```

启动 finisher。**该脚本本身运行在 tmux 里**；脚本退出 → tmux 关闭 session →
同 session 的后台 finisher 立刻收到 **SIGHUP 被杀**，来不及写第一行日志。
（finisher 的第一个 `log` 调用在等待循环开头，DONE 已存在，本应立即输出。）

于是实例一直运行到**账号欠费被断**（14:35 后 GPU 空闲，14:42 起端口拒绝连接）。

**修复**（`scripts/run_round152_g0_4b.sh` 尾部）：

```bash
setsid nohup bash "$ROOT/round152_finish.sh" >>"$LOG" 2>&1 </dev/null &
log "started finisher (detached via setsid+nohup, pid=$!)"
```

`setsid` 让 finisher 进入新 session，tmux 拆除够不到它；`nohup` 再加一层保险；
stdout/stderr 也并入日志，避免再有"静默死亡"。

> **通用教训**：在 tmux/screen 里跑的脚本，其后台子进程与脚本**同 session**。
> 凡是"脚本退出后还要继续活着"的工作，必须 `setsid`/`nohup`/`disown` 明确脱离，
> 否则它会在脚本成功退出的那一刻被静默杀死 —— 而且因为它死得比写日志还早，**日志上看不出任何异常**。

---

## 6. G0 的最终结论

1. **G0 不成立为"内容迁移"的证据。** real vs control 的 +10 点 EM 是 control 的损伤
   （TriviaQA 264/500 空串、NQ 混入 `yes`、BoolQ 偏向 `Yes`），nople 不崩，round-155 的诊断成立。
2. **graft 有强因果影响，但主导是格式/协议层面**：注入把 4B 从对话先验压回 raw 续写格式；
   这个效应 real 与乱序行**都有**，因此不是知识效应。
3. **通用能力没有提升**：在最公平的口径下（词序列 EM）real 0.3620 vs nople 0.1427 看似 real 赢，
   但 nople 的 0.1427 是**字符串评测对脚手架输出的系统性低估**造成的假象，不是能力差距。
   在 token 级评测实现之前，**这个对比没有结论**。

   → 因此本轮**不能**声称"real PLE 提升 4B"，也**不能**声称"real PLE 无用"。
   能确定的是：**在现有指标下，任何关于 PLE 内容效应的结论都不可靠**，必须先修指标。

---

## 7. 技术债（新增，按优先级）

| 优先级 | 项 | 说明 |
|---|---|---|
| P0 | token 级 EM | 字符串级评测在跨格式时必然失真；改为首个内容 token 与 gold token id 比较 |
| P0 | gold NLL 的空格口径 | 主口径改为 `" " + answer`；保留无空格口径，差值作为格式敏感度指标 |
| P0 | 每个 arm 记录输出格式指纹 | 脚手架率（含 `<think>`/角色 token 的比例）、空串率、distinct 数、标签分布偏移；任一异常即拒绝跨臂比较 |
| P1 | 协议一致性断言 | 训练与评测必须同协议；臂间输出格式不一致时禁止直接比 EM |
| P1 | finisher 脱离 session | 已修；其余队列脚本（`run_round152_lora.sh` 等）需同样检查 |
