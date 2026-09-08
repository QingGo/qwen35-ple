# Round 135：Phase 2 六语料 pilot（3 步 / 1 seed / 无 QA）初步结果

> 日期：2026-09-08
> 实验：PURE_WIKI / PURE_FINEWEB / PURE_STEM / PURE_CODE / FW_CODE / FW_STEM
> 条件：1M token、seq_len=128、3 训练步、1 seed、reader=official、live Store-I、无 QA
> 输出：`outputs/phase2-pilot/phase1-*.json`

## 1. 结果

| Corpus | Real PPL | Control PPL | No-reader PPL | Real < Control | Real < No-reader |
|---|---:|---:|---:|---:|---:|
| PURE_WIKI | 44.405 | 44.806 | 46.324 | ✅ | ✅ |
| PURE_FINEWEB | 32.541 | 32.582 | 32.768 | ✅ | ✅ |
| PURE_STEM | 7.810 | 7.938 | 8.029 | ✅ | ✅ |
| PURE_CODE | 11.039 | 11.158 | 11.251 | ✅ | ✅ |
| FW_CODE | 14.054 | 15.038 | 15.267 | ✅ | ✅ |
| FW_STEM | 5.788 | 5.792 | 5.804 | ✅ | ✅ |

**所有六个语料都满足：real PPL < control PPL < no-reader PPL。**

这是 Phase 2 全量跑之前的强正向信号：三条线在 3 步内就出现了与 Phase 0 一致的
“真实 PLE 读取带来语言建模增益”模式，且不在只依赖 M1 混合语料。

## 2. 已知局限

- 只有 3 步训练，不是预注册的 500 步；
- 只有 1 个 seed，未做 3-seed 稳定性；
- 没有 QA / Code / Math 生成式评测，所以尚不能判断通用任务收益；
- 未做 unseen-KB；
- 仍在 CPU/本机环境下运行，未上 4090 做 5M/20M。

## 3. 下一步

1. 在 WSL/4090 跑正式 Phase 2：1M × 500 步 × 3 seeds × 150 QA（96 token 生成）；
2. 用 `scripts/summarize_phase1_matrix.py` 聚合 PPL + contains + extracted EM；
3. 用 `scripts/run_unseen_kb_experiment.sh` 跑 unseen-KB；
4. 若正 → 5M/20M 规模曲线。

## 4. 本轮新增工具

- `scripts/summarize_phase1_matrix.py`：Phase 1/2 矩阵聚合；
- `scripts/build_kb_token_streams.py`：KB split → train/eval token 流；
- `scripts/run_unseen_kb_experiment.sh`：seen-KB 训练 + unseen-KB 评测；
- `scripts/run_phase1_matrix.sh --skip-qa`：快速 pilot 模式。
