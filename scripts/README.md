# 脚本目录说明

脚本进入前请同步更新本文件与 `docs/qwen35-ple-design.md` / `docs/roadmap.md`。

## 已有脚本

| 脚本 | 作用 |
|---|---|
| `build_table_assets.sh` | 调用 EngramDB CLI 构建/校验 Store-P 视图，并检查 manifest 字段 |
| `run_m0_smoke.py` | M0 冒烟：磁盘版 MultiHeadEmbedding 自检 + 合成表/真表模型 e2e |
| `run_ablation_eval.py` | 最小知识召回/长上下文/推理评测执行器，产出 A0/A1 兼容 JSON |
| `run_cpt_smoke.py` | M2 CPT 训练冒烟：A0 基线 / A1 PLE 处理可训练性验证 |
| `run_eval.py` | A0/A1 评测结果 JSON 对比报告入口 |
| `table_assets.py`（src 内） | 查找 EngramDB CLI、读取/校验视图 manifest 的 Python 编排层 |
| `generate_official_ple_snapshot.py` | 从固定官方 `refs/qwen4_exp_modeling.py` AST 抽取 PLE 参考快照，支持 `--check` |
| `generate_official_ple_forward_golden.py` | 用官方 PLE 快照生成 4096 token 前向 golden（`tests/golden/`） |
| `run_qwen35_e2e.py` | Qwen3.5-0.8B + engram-peft PLE-lite 的 CPU forward/generate e2e（模型放 `data/`，已 gitignore） |
| `run_qwen35_ablation.py` | Qwen3.5-0.8B A0/A1 极小消融：同语料同 step 数，记录 held-out loss + 迷你知识/推理 probe |
| `precompute_real_ple_features.py` | 用真实 FP8 PLE 表 + EngramDB Store 预计算 `e_t` 特征 |
| `run_ple_knowledge_probe.py` | 真实 PLE `e_t` 线性知识探针：语义类别可分性 |
| `run_ple_adapter.py` | 冻结 Qwen3.5 + 预计算 `e_t` 的 Engram-style reader，支持 layer/branches/short_conv |
| `run_full_matrix.sh` | 完整实验矩阵批处理：layer × branches × short_conv × real/control |
| `run_engramdb_v028_smoke.py` | EngramDB v0.2.8 消费冒烟：rowid / discover_ple / weight_scale / Store e_t / View |

## 预期脚本（与设计文档里程碑对应）

- `prepare_corpus.py`        语料构建（复用 EngramDB scripts/corpus_build.py 产物）
- `run_cpt.py / run_sft.py`  训练入口（包装 engram-peft CLI）
- 完整评测扩展：长上下文、基础 reasoning（当前只有最小知识召回）
