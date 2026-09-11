# Round 163：读出实现的对拍 —— 补齐唯一没有检查的一环

> 问题（第三轮审计的收尾）：**我们所有"嫁接无法注入内容"的结论，是关于
> `OfficialSourceQwenReader` 这个类的。它的 `forward` 从来没有和它声称复用的
> 官方数学比对过。** 如果它不忠实，那么"这个设计不能注入内容"就必须降级为
> "我们的实现不能注入内容"。
>
> 新增：`tests/test_official_reader_forward_golden.py`（16 项，全部必跑）
> 固化：`tests/golden/qwen38_flash_next_text_config.json`（官方 config 摘录 + sha256）

---

## 0. 判决

**读出实现是忠实的。审计的第三个缺口是唯一一个关掉之后结论不变反而更强的缺口。**

| 主张 | 审计前 | 审计后 |
|---|---|---|
| "**这个设计**无法注入内容" | 只有"**我们的实现**"的证据 | **实测：我们的实现 ≡ 官方数学**（位级） |

同时**新发现一处保真度偏差**（§4）：官方把 PLE 挂在 **0-based 第 1 层**，
我们的主线挂在**第 2 层**。

---

## 1. 缺口的确切形状

在此之前，`OfficialSourceQwenReader` 在全仓库只在 `test_reader_registry.py`
被**实例化**过。三处提到官方 PLE 的测试，按各自覆盖的对象都不是它：

| 文件 | 实际对象 | 几何 |
|---|---|---|
| `test_official_ple_reference.py` | 快照能否复现自己的生成器 | `hc=1, k=2, d=2` |
| `test_ple_forward_golden.py` | engram-peft `EngramLayer` vs `ple_reference` | `hc_mult=1` |
| `test_phase_b_official_loader.py` | 磁盘表适配器 | — |
| `tests/golden/official_ple_forward_4096.npz` | 官方快照的前向 | **`hc=1, k=2, d=2`** |

**没有一处**跑过嫁接真正使用的多分支（`hc_count=4`）、膨胀卷积（kernel 4,
dilation 3）读出。所以"设计不能注入内容"有证据，"我们对设计的实现是忠实的"
一条都没有 —— 而这正是支撑最强结论的那一环。

---

## 2. 对拍设计

比较**共享读出**：从 `(e_t, h)` 到返回张量的全部。`e_t` 在生产中由调用方给出
（`model._current_ple_e_t`），所以哈希/查表**故意不在范围内**（已由跨仓库 golden
单独固定）；增量解码也不在范围内（我们的 reader 看不到 `past_kv`）。

把 `query_bridge` 与 `out_proj` 换成 `Identity` 后，两边必须是**同一个函数**。
这两处适配是我们加的、不是官方代码的，测试**显式固定**它们而不是藏起来：

- `query_bridge`：官方对**已经过 hyper-connection 展开**的 `[B,T,hc*d]` 隐状态做
  `norm_query`。单流目标模型没有这个张量，所以插一个学习到的线性映射。
- 分支求和 + `out_proj`：官方返回完整 `[B,T,hc*d]`；我们把 `hc` 个分支求和后投到目标宽度。

断言用 `atol=0, rtol=0`（**位级**），不是"接近"。

---

## 3. 结果

```
test_shared_readout_matches_official_small                    PASS  (hc=4, k=4, d=3)
test_shared_readout_matches_official_single_branch            PASS  (hc=1)
test_shared_readout_matches_official_short_sequence           PASS  (T=3 < 感受野)
test_shared_readout_matches_official_production_geometry      PASS  (2560/4/4/3)
test_branch_sum_plus_out_proj_is_the_whole_adaptation         PASS  (真实适配器)
test_shared_readout_matches_official_with_real_weights        PASS  (官方真实权重，位级)
test_every_shared_tensor_is_actually_exercised                PASS  (6/6 变异可检出)
```

其中 `..._with_real_weights` 是决定性的一条：**生产几何 + `data/official_ple_reader.pt`
的真实冻结张量 + 位级相等**。它不需要 47.684 GiB 的行表 —— 读出只由
`key_proj`/`value_proj`/三个 norm/`conv1d` 决定，表是另存的。

### 3.1 反空转守卫

`test_every_shared_tensor_is_actually_exercised` 逐个扰动 6 个共享张量，
要求输出**必须改变**。没有它，一个悄悄丢掉某一项的"对拍"（或比较两个零）
也会通过。这是本轮唯一一个"让检查可被检查"的机制。

---

## 4. 顺带查出的两件事

### 4.1 官方 config 的权威值（已固化）

第一次跑是**失败**的，bisect 定位到 `norm_key` 开始分叉（`key_proj`/`value_proj`
位级相等）。原因**在我的测试里，不在 reader 里**：我给测试配置写了
`rms_norm_eps=1e-5`（抄了 `test_phase_b_official_loader.py`），
而官方 Qwen3.8-Flash-Next 的 `text_config` 是 **`rms_norm_eps = 1e-06`** ——
**正好等于 `OfficialSourceQwenReader` 的默认值**。reader 是对的，我的测试是错的。

> 这是**族 B 的第一次**（见 §5.1）：检查跑了，第一遍就抓到，bisect 一次定位。
> 错误本身是新形状 —— 不是"没跑检查"，而是**从别处抄了一个常数而没有验证它的出处**
> （`test_phase_b_official_loader.py` 用 `1e-5`，官方是 `1e-06`）。

官方 `text_config` 的权威值（现在从 `tests/golden/qwen38_flash_next_text_config.json`
读取并断言，不再靠记忆）：

| 字段 | 值 | 作用 |
|---|---|---|
| `hidden_size` | 2560 | `key_proj` (10240, 2560) |
| `hc_count` | 4 | 10240 的来源 |
| `ple_embed_dim` | 2560 | `value_proj` (2560, 2560) |
| `ple_conv_kernel_size` | 4 | `conv1d` (10240, 1, 4) |
| `heads_per_ngram` | 8 | — |
| **`ngram_size`** | **3** | **膨胀 = 3；头数 (3−1)×8 = 16** |
| `rms_norm_eps` | 1e-06 | reader 默认值 |
| `split_ngram_parts` | 128 | 128 个分片 |
| `ngram_vocab_size_base` | 20,000,000 | 行表规模 |

`ngram_size` 是**最值得固定**的一个：它同时决定卷积膨胀和头数，而
**`conv1d.weight` 的形状 `(10240, 1, 4)` 对任何 dilation 都一样** ——
膨胀是唯一无法从权重形状反推的读出参数，也是 round-157 那个 12-token 窗口的来源。

### 4.2 【新】官方在第 1 层，我们在第 2 层

`refs/qwen4_exp_modeling.py:1276`：

```python
ple_layer_index = config.ple_layer_ids.index(layer_idx + 1) if layer_idx + 1 in config.ple_layer_ids else None
```

`ple_layer_ids` 是 **1-based**。官方 config 是 `[2]` → `layer_idx = 1`（0-based），
与 checkpoint 前缀 `model.language_model.layers.**1**.ple.*` 完全一致。

而我们的主线（round 149 / 152 / 156 / 162）**都是 `--layer 2`**，即 0-based 第 2 层
（第三层）。已核对产物：`outputs/round162/arm-wiki.json`、`round152chat/lora-real.json`、
`round152g0/train-real.json` 全部 `layer = 2`。

**影响范围**：

- round-157 的界 `I(future; e_t | h_t) ≤ I(future; window | h_t)` 是**读出输入→输出的
  确定性函数**的性质，**与挂在哪一层无关**，不受影响。
- 全部嫁接否证（149/152/156/162）**未受影响但未被排除在第 1 层的版本**：
  "第 1 层会不会不同"是一个**未测**的问题，不是被顺带否定的问题。
- 这一条**不改变任何已记录的结论**，但它改变了"我们复现了官方嫁接"这句话的措辞：
  我们复现了官方的**读出**，不是官方的**挂载点**。

---

## 5. 这一轮真正的产出

三次审计里，前两次是"结论其实是稳的"（好消息），这一次是**唯一一个真的没有检查过的地方**，
而它是**唯一一个支撑最强主张的地方**。

按本项目已经记录九次（族 A）的模式，这一轮值得记的是它的**形状**：

> 我验逻辑很勤（不等式反复推、预注册认真写），但**默认"我读到的代码就是跑的代码"**。
> **族 A 的第十次**是同一个形状：我**以为**有测试覆盖 reader，因为有三个文件提到官方 PLE。
> 真正起作用的修复不是"再读一遍代码"，而是**让覆盖本身可被查询** ——
> 一个会在缺件时 fail 的 golden、一个固定死常数的 config 断言、
> 一个会因扰动而失败的变异守卫。

### 5.1 族 B：机制开始自己抓错（本轮两次）

> **先把计数说准**（我第一版写错了）。两类错误形状不同，不该合并计数：
>
> | 族 | 形状 | 次数 | 谁发现的 |
> |---|---|---|---|
> | **A** | 检查**不存在**或**静默跳过**，于是主张未经检验地立住了 | 九次 + 本轮的 reader 缺口 = **十次** | 审计 |
> | **B** | 检查**跑了**，并且**当场抓到** | 本轮两次 | 机制本身 |
>
> 下面这条属于**族 B**。把它记成"第十一次检查没跑"是错的 —— 恰恰相反，
> 它跑得又快又准。

第一次提交 **CI 红了**（`Test` 步失败；`Lint` 与论文编译都通过）。原因不是测试写错，
而是**我把"缺件要不要 fail"这条语义只修了一半**：

CI 的 Test 步给整个 job 设了 `QWEN35_REQUIRE_GOLDEN=1`，意思是
"**本该在仓库里的** golden 不见了 → 这次运行未经检查 → 这是失败不是跳过"。
我却把这个"致命"助手用在了 65 MB 的 `data/official_ple_reader.pt` 上 ——
而它被 `.gitignore:8` 忽略、**根本不在 CI 里**。于是 CI 上必然硬失败。

我在**同一个文件里、相隔一个函数**的地方，刚刚才为 `config.json` 修好这条语义，
然后没有回头把同一个判断用到旁边的 checkpoint 上。

修复不是把断言删掉，而是**把两类资产在代码里分开**：

| 助手 | 用于 | 缺件时（`QWEN35_REQUIRE_GOLDEN=1`） |
|---|---|---|
| `_require_or_skip` | **已提交**的 golden（`tests/golden/*`） | **fail** —— 运行确实未经检查 |
| `_skip_external` | gitignore 的大资产（`.pt`、49 GB checkpoint） | **skip** —— 它在不在与检查是否接好无关 |

四种组合都实测过（远端）：

| 条件 | 结果 |
|---|---|
| 无 `.pt` + flag（= CI） | `13 passed, 3 skipped` —— **不再红** |
| 有 `.pt` + flag | `15 passed, 1 skipped` |
| 摘录缺失 + flag | **`1 failed`** —— 提交的 golden 仍然硬失败 |
| 摘录缺失 + 无 flag | `14 passed, 2 skipped` |

> 这一条本身就是本轮论点的第二个例证：**"我知道该怎么做"不等于"我在每一处都做了"。**
> 唯一能兜住它的是**在 CI 的真实条件下跑一遍**，而不是再读一遍代码。

---

## 6. 复现

```bash
# 全量（torch + 官方权重存在时）
PYTHONPATH=src QWEN35_REQUIRE_GOLDEN=1 pytest tests/test_official_reader_forward_golden.py -q

# 模拟 CI（权重缺席，两个外部资产测试应当 skip 而非 fail）
PYTHONPATH=src QWEN35_REQUIRE_GOLDEN=1 pytest tests/test_official_reader_forward_golden.py -q

# 额外用真实 config.json 校验固化摘录（作者机器）
QWEN35_OFFICIAL_CONFIG="/Volumes/My Passport/qwen38-ple/config.json" \
  PYTHONPATH=src pytest tests/test_official_reader_forward_golden.py -q
```

远端实测：有权重 `15 passed, 1 skipped`（含真实权重位级对拍）；
模拟 CI（无权重）`13 passed, 3 skipped`；全量在两种条件下均**零失败**
（无权重 `187 passed, 7 skipped`）。

---

## 7. 未决

- 【已记录，不在本轮范围】官方挂载点（第 1 层）的版本未测。属于
  "原版验证过的配置"清单，与"可训练的表 / 学过路由的主干 / 长尾语言建模目标"同级，
  **明确标注为未验证**，而不是被否证。
- 【新增】`outputs/v41-flash/config.json` 的 `rms_norm_eps = 1e-20`（DeepSeek V4.1-Flash）
  与 Qwen3.8 的 `1e-06` 差 14 个数量级。**V4.1 已移除卷积**，故对我们无直接影响，
  但如果将来做跨系统对拍，这是一个必须显式传参而不能吃默认值的字段。
