# Round 52：P1 记忆接口原型（代码与实验入口）

> 日期：2026-09-04
> 状态：P1 第一阶段代码已完成，尚未在 WSL 真表上跑门禁
> 目的：先把“exact longest-match PLE bank + TokenMem cross-attention + distribution-level memory/router”做成可复现原型，再决定是否进入 MoRA/GaLore。

---

## 1. 本轮完成内容

### 1.1 新增模块

| 文件 | 作用 |
|---|---|
| `src/qwen35_ple/memory/bank.py` | 纯 NumPy exact n-gram bank，支持 2/3/4-gram，longest-match 查询，control shuffle，保存/加载 |
| `src/qwen35_ple/memory/token_mem.py` | TokenMem 式独立 cross-attention、distribution memory head、router、P1MemoryModule |
| `src/qwen35_ple/memory/__init__.py` | 内存包导出；PyTorch 模块懒加载，bank 可在无 torch 环境使用 |

### 1.2 新增脚本

| 脚本 | 用途 |
|---|---|
| `scripts/build_exact_ple_bank.py` | 从预计算 `tokens.npy` / `e_t.npy` 构建 real 与 control bank |
| `scripts/train_p1_memory.py` | 冻结 backbone，只训练 P1 记忆模块 |
| `scripts/eval_p1_memory.py` | rare/common QA 上对比 no-memory / real / control |

### 1.3 测试

- `tests/test_memory_bank.py`：longest-match、fallback、control、save/load、multi-slot；
- `tests/test_memory_token_mem.py`：cross-attention、memory head、router/fusion、P1 module shape（有 torch 时运行）。

---

## 2. P1 架构

```text
输入 token 序列
    │
    ├── 原始 PLE e_t（可选 fallback）
    │
    └── ExactNgramBank
          ├─ 2-gram exact
          ├─ 3-gram exact
          └─ 4-gram exact（外部补足 Qwen PLE 原生缺失）
               │
               ▼
        per-token memory slots [T, K, d_mem]
               │
               ▼
   TokenMemCrossAttention（独立通道，不碰 backbone self-attention）
               │
               ▼
   P1MemoryModule
     ├─ MemoryLogitHead：memory representation -> vocab distribution
     └─ MemoryRouter：backbone logits 与 memory logits 的逐 token 融合
```

关键点：

- backbone 冻结；
- memory 是独立信息通道；
- real/control 只差 bank 的 key->value 对应关系；
- 4-gram 为外部 exact bank 能力，不依赖 Qwen PLE 原生 4-gram。

---

## 3. 运行方式

### 3.1 构建 bank

```bash
python scripts/build_exact_ple_bank.py \
  --feature-dir data/ple-books-160k \
  --output data/exact-ple-bank.npz \
  --control-output data/exact-ple-bank-control.npz \
  --max-order 4
```

可重复 `--feature-dir` 合并多个预计算语料。

### 3.2 训练

```bash
python scripts/train_p1_memory.py \
  --model data/models/Qwen3.5-0.8B \
  --features data/ple-books-160k \
  --bank data/exact-ple-bank.npz \
  --steps 200 \
  --seq-len 64 \
  --layer 8 \
  --output outputs/p1-memory-real.pt
```

### 3.3 评测

```bash
python scripts/eval_p1_memory.py \
  --model data/models/Qwen3.5-0.8B \
  --checkpoint outputs/p1-memory-real.pt \
  --bank-real data/exact-ple-bank.npz \
  --bank-control data/exact-ple-bank-control.npz \
  --qa-file data/rare-kb-v1.json \
  --rows-dir /home/zeng/qwen38-rows \
  --scale 0.00019931793212890625 \
  --output outputs/p1-memory-eval.json
```

---

## 4. 门禁口径

P1 的第一道门禁：

```text
rare knowledge:  real answer-logprob > control answer-logprob
通用能力:        no-memory 与 real 不显著退化
```

建议同时记录：

- rare/common 分层；
- real−control logprob；
- first-token hit (弱 exact-match)；
- mean router alpha；
- 3 seeds 重复；
- bank 是否包含 QA 泄漏（应使用独立记忆语料构建 bank）。

---

## 5. 待办 / 风险

| 待办 | 说明 |
|---|---|
| WSL 真表跑通 bank 构建 | 当前本地只有预计算特征，尚未在 WSL 生成 real/control bank |
| 训练规模与超参 | 当前脚本是最小可用版本，未做 LR/层数/head 数扫描 |
| 4-gram 价值单独验证 | 需要分别测 2/3-only bank 与 2/3/4 bank |
| 防泄漏 | bank 不得包含评测 QA；否则 real 优势是记忆泄漏不是 PLE 信号 |
| 若门禁失败 | 按 round-50 停止条件转向 RAG / 蒸馏 / 更语义化记忆，不做大规模 RL |

---

## 6. 下一步

1. 在 WSL 上构建 bank；
2. 跑 `train_p1_memory.py`（real 与 control 各一轮）；
3. 跑 `eval_p1_memory.py`，看 rare real−control；
4. 如果 positive，进入 Phase P2 MoRA/GaLore；
5. 如果 negative 或不显著，记录审计结果并转 RAG/蒸馏。
