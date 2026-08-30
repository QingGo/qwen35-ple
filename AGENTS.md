# AGENTS.md — qwen35-ple 协作约定

## 仓库角色

| 仓库 | 角色 | 依赖 |
|---|---|---|
| EngramDB | 行表存储 + 视图 + C ABI（底层，无内部依赖） | — |
| engram-peft | 记忆层引擎（DeepSeek 对齐实现）+ 训练/TRL 基础设施 | EngramDB（可选，磁盘注入） |
| **qwen35-ple（本仓库）** | 嫁接实验编排：配置/数据/CPT/后训练/消融/评测 | engram-peft + EngramDB |
| LLM-CompileForge | 推理编译器 + Rust runtime | EngramDB（C ABI / 视图文件，**不依赖 Python 侧**） |

依赖方向严格无环：`LLM-CompileForge → EngramDB`；`qwen35-ple → {engram-peft, EngramDB}`。

## 交互契约（必须遵守）

- 契约唯一权威：`docs/integration-contract.md`（版本 v1，冻结）。
- 任何跨仓库接口变化：先改契约文档 → 再改实现；**只允许新增**（新字段/新符号/新枚举值），
  禁止改变既有项语义或删除；ABI 演进用 `_v2` 新符号，旧符号永不改变行为。
- 数值一致性：与 EnGramDB/官方参考位级一致（bit-exact），任何捷径必须带 golden 对拍与说明。

## 惯例

- 文档用中文（与兄弟仓库一致）；代码导出符号/标识符用英文。
- 命令面：`make sync/lint/test`；测试跑在 `tests/`，禁止在 src 里写 IO 副作用测试。
- 实验资产（模型、表视图、语料）一律放 `data/` 与 `outputs/`（已被 .gitignore），不进版本库。
