# 脚本目录说明

脚本进入前请同步更新本文件与 `docs/qwen35-ple-design.md` / `docs/roadmap.md`。

## 已有脚本

| 脚本 | 作用 |
|---|---|
| `build_table_assets.sh` | 调用 EngramDB CLI 构建/校验 Store-P 视图，并检查 manifest 字段 |
| `run_m0_smoke.py` | M0 冒烟：磁盘版 MultiHeadEmbedding 自检 + 合成表/真表模型 e2e |
| `run_ablation_eval.py` | 最小知识召回评测执行器，产出 A0/A1 兼容 JSON |
| `run_eval.py` | A0/A1 评测结果 JSON 对比报告入口 |
| `table_assets.py`（src 内） | 查找 EngramDB CLI、读取/校验视图 manifest 的 Python 编排层 |

## 预期脚本（与设计文档里程碑对应）

- `prepare_corpus.py`        语料构建（复用 EngramDB scripts/corpus_build.py 产物）
- `run_cpt.py / run_sft.py`  训练入口（包装 engram-peft CLI）
- 完整评测扩展：长上下文、基础 reasoning（当前只有最小知识召回）
