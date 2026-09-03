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
| `run_phase0.py` | Phase 0 三线实验基座：train/val 分割、多 seed、no-reader/real/control、QA log-likelihood + exact-match 生成式评测 |
| `run_phase0.sh` | Phase 0 一条命令 wrapper（自动处理本机兼容 PYTHONPATH）|
| `run_live_vs_precomputed.py` | Phase 1 gate：live DiskPleNGramEmbedding 与当前 fetch_e_t 数值一致性 |
| `qwen4_ple_custom_loader.py` | Phase B：官方 Qwen4Exp 模型加载时跳过 ngram shard，并安装磁盘 PLE adapter（支持 dry-run） |
| `run_real_fp8_e2e.py` | 真实 FP8 Store-I + 配置驱动 engram-peft 自动注入的 CPU forward/generate e2e |
| `download_mix_sources.py` | 可复现下载 M1–M5 使用的 ModelScope 语料来源（alpaca/wiki/cot/msagent） |
| `build_mix.py` | 可复现 1M token 混合语料构建：general/chat/wiki/cot/tool 按比例采样、Qwen tokenizer 产出 `tokens.npy`、manifest + 污染过滤 |
| `audit_contamination.py` | 严格 QA 污染审计：答案/问题/QA n-gram 重叠，输出逐题和汇总报告 |
| `run_mix_batch.sh` | WSL 批量跑 M1–M5：`run_phase0.py --live-store` + 150 QA exact-match 三线 |
| `run_mix_one_wrapper.sh` | 单 mix 后台 wrapper：配合 Windows Scheduled Task 长任务托管 |
| `summarize_mix_results.py` | 汇总多份 Phase 0 JSON：各 mix 的 real/control/no-reader EM、val loss、分任务 EM |
| `export_phase0_metrics.py` | 将 Phase 0 JSON 导出为文章可用的 train_loss / summary / per_question CSV |
| `plot_phase0_metrics.py` | 从 CSV 生成 QA EM、val loss、train loss 曲线等论文图 |
| `analyze_qa_lines.py` | 三线 QA JSON 逐题对比：new correct/wrong、语料命中 |
| `mechanism_alignment.py` | 机制验证：PLE e_t 与 Qwen hidden 的 CKA / Procrustes / kNN overlap / intrinsic dimension，以及 reader 参数和 gate 统计 |
| `mechanism_logit_patch.py` | 快速 logit-level activation patching：no-reader / real / control / random / zero 五条件；支持 `--inject-scale` 做注入强度扫描 |
| `mechanism_patching.py` | 逐 token 生成的 activation patching / 条件生成对比脚本 |

## 预期脚本（与设计文档里程碑对应）

- `prepare_corpus.py`        语料构建（复用 EngramDB scripts/corpus_build.py 产物）
- `run_cpt.py / run_sft.py`  训练入口（包装 engram-peft CLI）
- 完整评测扩展：长上下文、基础 reasoning（当前只有最小知识召回）
