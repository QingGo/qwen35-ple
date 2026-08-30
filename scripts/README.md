# 本文档是脚本目录的占位说明；脚本进入前请先更新本文件与设计文档 §7。
#
# 预期脚本（与设计文档里程碑对应）：
#  - `build_table_assets.sh`   重建/校验 Store-I 行表与 Store-P 视图（engramdb view）
#  - `prepare_corpus.py`        语料构建（复用 EngramDB scripts/corpus_build.py 产物）
#  - `run_cpt.py / run_sft.py`  训练入口（包装 engram-peft CLI）
#  - `run_eval.py`              知识/长上下文/推理评测
