# Round 169:方向 A 的 infra 计划(先量后建)

> 日期:2026-09-14
> 目的:在下载 10B 级语料、开小规模从零共训配对之前,把**能省时间的都先量掉**。
> 本文所有数字都是本机实测,不是估计。凡未实测的都标了"待测"。

---

## 0. 机器(实测)

| 项 | 值 |
|---|---|
| CPU | **128 核**(B1 训练期间 93% idle,load 10–22) |
| 内存 | 1007 GB 总,787 GB available |
| GPU | 1 × RTX 4090,**24564 MiB** |
| 可写磁盘 | `/root/autodl-tmp` 124 G,**当前余 20 G** |
| 只读挂载 | `/autodl-pub`(7 TB)与 `/autodl-pub/data`(4 TB)都是 AutoDL 公共数据集库,**只读** |
| 非我们的盘 | `/dev/sda2` 那 397 GB 是**宿主机**的盘(以单文件 bind 进来),不可用 |

**结论:可写磁盘是唯一的硬约束。**

---

## 1. 网络(实测):用 hf-mirror,**不要**用 network_turbo

```
hf-mirror.com                     285 MB / 19.2 s  =  16.4 MB/s   ← 用这个
huggingface.co(开 network_turbo)  40 MB / 90 s   =   0.45 MB/s   ← 慢 36 倍
```

`/etc/network_turbo` 自己都提示"开启后访问其他资源会更慢"。pypi 直连 3.5 MB/s 正常。

⇒ **FineWeb-Edu 28.5 GB 约 30 分钟下完**。下载从来不是瓶颈。

---

## 2. 语料(实测)

`HuggingFaceFW/fineweb-edu` 的 `sample/10BT`:**14 个 parquet,共 28.52 GB**
(13 × 2.15 GB + 1 × 0.54 GB)。

**全量实测一个 shard 后的真实换算(已跑完,不是估计):**

| 量 | 实测 |
|---|---|
| 1 个 shard | 726,000 docs → **755,998,000 tokens**(2.153 GB 原始文本) |
| tokens / 原始字节 | **0.3512**(外推 28.52 GB × 0.3512 = **10.02B tokens**,`10BT` 名副其实) |
| tokens / 字符 | 0.2190 |
| token 产物 | **3.02 GB / shard**(uint32) |
| **磁盘比** | **1.40×**(tokenize 会**放大**占用,因为 uint32 = 4 B/token) |

**A 的规模必须从 10B 降到 ~2B**,理由是纯算术(uint32 是必需的:Qwen 词表 248320 > 65535,
uint16 装不下 token id):

```
10B tokens → 40 GB tokens(13.3 shards)+ 28.5 GB 原始
2B  tokens → 9.1 GB tokens(2.65 → 3 shards)+ 峰值 2.15 GB 原始 ≈ 11.2 GB 峰值
可写磁盘:                                                17 GB(清理前)
```

**2B tokens 已经是现有 5M 的 400 倍,且对 20M 参数模型是 100 tok/param —— 高于 Chinchilla 的 20×。**
拿 10B 换来的边际信息远不如"能真的跑起来"重要。

---

## 3. Tokenization 管线(已实现 + 已实测)

`scripts/round169_a_datapipe.py`:parquet → 文本 → Qwen fast tokenizer → `.npy` (uint32)。
文档间插一个 `<eos>`(建模决定,不是细节)。

**实测吞吐(单进程,2.15 GB parquet 的前 120k docs):**

| threads | tok/s |
|---|---|
| 8 | **1,402,908** |
| 32 | 1,318,984 |
| 64 | **1,869,567** |

**关键发现:线程伸缩是平的。** 8 → 32 线程**没有任何提升**(反而略降),64 线程只有 1.33×。

⇒ **串行段不在 tokenizer 上,而在 parquet 解码 + `to_pylist()`**(单线程 Python)。
这正是 rung ③ 那个教训的同一族:**先找串行段,再谈优化**。

**修法(尚未实施,已量化)**:按 row group **跨进程**并行解码,而不是多线程喂一个解码器。
预期能把 ~2.1 M tok/s 提到接近核数量级的倍数。

**全量单 shard 的实测(空闲时,不在争抢)**:726,000 docs / 756 M tokens / **353.6 s**
= **2,137,968 tok/s**。所以 2B tokens 单进程只需 **~16 分钟**。
⇒ 并行解码是"锦上添花",**不是阻塞项**;A 的瓶颈从头到尾都是磁盘。

---

## 4. 训练侧 requirements(这才是省时间的地方)

### 4.1 数据加载
* **memmap `.npy` uint32**,随机取连续窗口。**禁止在任何热路径上出现 `.tolist()`** ——
  本项目已经在两个地方被它钉在单核上(rung ③ 的行抓取、parquet 解码)。
* 一个 epoch 的窗口顺序用固定种子洗牌,并把种子写进 checkpoint。

### 4.2 从第 0 步就挂吞吐仪表
playbook 的规矩:稳态跑起来先探 30 秒,记 **tokens/s、MFU、显存占比**;
收尾再探一次,两次都写进结果文档。没有这张表的加速不许进主流程。

**MFU 定义(小模型必须写清)**:

```
MFU = 6 · N_params · tokens_per_sec / peak_flops
```

4090 上 bf16 + tensor core 的**实测可达**约 60–100 TFLOPS(小模型受 launch/带宽限制,
达不到标称 330)。**用实测值算 MFU,不要用标称值自欺。**

### 4.3 配对臂必须**双匹配**
有记忆与无记忆两臂要在 **token 数**和 **FLOPs** 上分别匹配一次并分别报告 ——
记忆臂多了查表与 reader 的算力,只匹配 token 会让对比偏袒某一侧。

### 4.4 checkpoint / resume
每 N 步存(模型 + 优化器 + 数据窗口种子 + 步数)。A 要跑十几个 arm,
任何一次中断都不能从头再来。

---

## 5. 预算(算术,不是估计)

```
单臂 FLOPs = 6 · N · T
20M 参数 × 2B tokens = 2.4e17 FLOPs
按实测 60 TFLOPS      ≈ 67 分钟/臂

规模阶梯 {5M, 20M, 80M} × 2 臂 × 3 种子 = 18 次
总 FLOPs ≈ 7.6e18      ≈ 35 GPU·小时
```

⇒ 2B tokens + 三个规模是**两三天**的量级,不是"跑不起",而是"要排好队"。
若时间紧,第一步只跑 **{5M, 20M} × 2 臂 × 2 种子 = 8 次 ≈ 10 小时**,
先把"收益是否随规模增长"的方向判出来。

---

## 6. 不要做的事(violates 本项目已经付过学费的教训)

```text
✗ 在热路径上用 Python list / to_pylist / reshape(-1).tolist()
✗ 用 bf16 存任何后面要「相减」的量(见 B1:量化底噪 3e-5 > 信号 1e-7..2e-5)
✗ 用标称 TFLOPS 算 MFU
✗ 只匹配 token 不匹配 FLOPs
✗ 没有预注册就跑 verdict-bearing 的实验
✗ 挂 watcher/supervisor 而不写「终局失败即中止 + 未达成非零退出 + 日志记原因」(playbook §2.6)
```

---

## 7. 待办(按顺序)

1. ~~等 tokenization 全量跑完 → 得到真实 tokens/byte 与单 shard 的 token 数~~
   **已完成**:0.3512 tokens/字节,0.756B tokens/shard,2.14 M tok/s,磁盘比 1.40×。
2. **磁盘**:A 需要 ~11.2 GB 峰值。清理方案见下(当前 17 GB 可用,清完约 32 GB)。
3. **修并行解码**(按 row group 跨进程)→ 可选,非阻塞(单进程 2B 只要 ~16 分钟)。
4. 再下 2 个 shard → 凑够 2B tokens,写进 `a-corpus/tokens/`。
5. **写 A 的预注册**(规模阶梯、双匹配、判定规则)——在任何 A 的数字产生之前。
6. 建训练 harness(4.1–4.4),先跑 5M 规模的冒烟,把 MFU 表立起来。
7. 再开 8 次配对跑。

---

## 8. 磁盘地图(去重后实测)

⚠️ **`repo-old/` 不是旧备份**:`repo/outputs` 是指向 `repo-old/outputs` 的**符号链接**
(inode 相同)。删掉 `repo-old` 等于删掉全部产物。`du` 也会因此**重复计数 12 G**。

| 路径 | 大小 | 处置 |
|---|---|---|
| `qwen38-rows` | 48 G | 🔒 行表,一切的基础 |
| `models` | 15 G | 🔒 0.8B/2B/4B/qwen38_ple |
| `outputs`(顶层) | 13 G | 部分可清,见分级 |
| `repo-old/` | 12 G | 🔒 **其中 11 G 就是活的 outputs** |
| `a-corpus` | ~10 G | 🔒 A 的语料 |
| `venv` | 5.5 G | 🔒 → miniconda3 |

`/`(overlay 30 G):已清出 7.7 G(删掉自己的 `/tmp/E-smoke.npy` 6.83 G + pip 残渣),
现余 **11 G**。最大的一块是 `/root/engram-serve` **17 G**(另一项目的两个 venv)。

