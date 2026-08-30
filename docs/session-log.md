# qwen35-ple Session Log

> 短期复盘，供空白上下文 agent 快速恢复。详细战略见 `docs/roadmap.md`。

## 2026-08-30：初始化后第二轮开发

### 完成

- 新增 `docs/roadmap.md`：北极星、技术债、借鉴矩阵、阶段计划。
- 新增 `src/qwen35_ple/ple_hash.py`：纯 Python `PLE_QWEN_V1` 参考实现，
  与 EngramDB golden 对拍通过（注意需用 64 位 wrapping 乘/异或）。
- 新增 `tests/test_hash_golden.py`、`tests/test_table_assets.py`、`tests/test_config.py`、
  `tests/test_cross_repo_hash_golden.py`、`tests/test_eval_protocol.py`。
- 新增 `src/qwen35_ple/table_assets.py`：定位 EngramDB CLI、读取视图 manifest、视图构建封装。
- 新增 `scripts/build_table_assets.sh`：调用 `engramdb view build --verify` 并校验 manifest。
- 新增 `scripts/run_m0_smoke.py`：M0 冒烟（quick 自检已跑通；e2e 需完整 engram-peft 依赖）。
- 新增 `scripts/run_eval.py`：A0/A1 结果 JSON 对比报告入口。
- 新增 `src/qwen35_ple/eval/protocol.py`：A0/A1 消融对比协议。
- 新增 `src/qwen35_ple/config.py`：YAML 配置加载 + 契约级校验。
- 初始化 `engine/`、`data/`、`train/`、`eval/`、`infer/` 包骨架。
- 修正 `.gitignore`：`data/`、`outputs/` 改为根目录锚定，避免忽略 `src/qwen35_ple/data/`。
- 修正配置样例中 `engram_vocab_size_per_ngram` 初值：
  `20000096` → `160000000`（每个 n-gram 的 8 头总桶数 ≈ 8×20M）。

### DONE（engram-peft 侧，按契约 C2 只增字段）

- `EngramConfig` 新增 `engine` / `table_spec` / `table_source` 字段。
- `engram_peft/hashing.py` 新增 `QwenPleHashMapping` 与 `create_hash_mapping`。
- `EngramModel` / `EngramLayer` / `EngramDataCollator` / `weight_transfer` 接入工厂，
  `qwen_ple` 自动关闭 tokenizer compression 并使用原始 Qwen token 哈希。
- 跨仓 golden：`tests/test_cross_repo_hash_golden.py` 已验证
  `QwenPleHashMapping` 与 EngramDB golden 逐位一致。

### 发现的债

1. EngramDB `engramdb prep` CLI 存在参数顺序/输出路径疑似 bug
   （输出写到了第一个参数路径），后续接入数据管线时需要绕开或修复上游。
2. 本仓 e2e 仍需要安装完整 engram-peft 依赖（peft/transformers）才能真正跑模型前向。
3. 多 PLE 层（A3）在 `qwen_ple` 引擎中尚未支持；当前 M1 严格限定单层。
4. 真表 Store-I / Store-P 资产尚未在本仓正式生成并纳入 M0 门禁。

### 下一步

- 在具备 peft 的完整环境运行 `scripts/run_m0_smoke.py --e2e`。
- 用真表或更大合成表跑 TinyLlama/Qwen3.5 + EngramDB 磁盘注入的前向/生成。
- 补真实评测执行器，产出 A0/A1 JSON，再用 `scripts/run_eval.py` 生成报告。
