# Round 118：论文图表自动绘制工具调研与落地

> 目标：为 System Architecture / Inference Pipeline 选择“agent 可自动绘制、不需要人工 GUI”的方案，并在本仓库落地。

---

## 1. 结论

**选型：Typst + Fletcher/CeTZ。**

理由：

- 纯文本/代码绘图，agent 只需生成 `.typ` 源码；
- 直接输出 SVG，矢量、可缩放、论文排版质量高；
- 与现有 Typst 论文链路无缝集成；
- 无需人工拖拽、无需外部 GUI；
- 图源文件可提交到仓库，具备可复现性；
- Fletcher 自带 ML architecture gallery，适合画神经网络/系统架构图。

---

## 2. 备选方案对比

| 方案 | 优势 | 劣势 | 是否适合 agent 自动绘制 |
|---|---|---|---|
| Typst + Fletcher/CeTZ | 矢量、代码化、与论文同链路、可复现 | 需要本地 vendored package | **首选** |
| Mermaid | 语法简单、生态好 | 需要 mermaid-cli / puppeteer 或在线服务 | 可用，但依赖外部渲染 |
| Graphviz | DAG/流程图成熟 | 需要 dot 二进制，样式偏工程图 | 可用，但不是首选 |
| PlantUML | UML/流程图 | 需要 Java/PlantUML 服务 | 可用，但重 |
| matplotlib | 已安装、可控 | 需要手写坐标，代码量大 | 可作为 fallback |
| draw.io XML | 适合人工、有丰富形状 | 自动生成/渲染链路较重 | 不够轻量 |

---

## 3. 落地内容

### 3.1 已新增文件

- `paper/figures/architecture_diagram.typ`
- `paper/figures/architecture_diagram.svg`
- `paper/figures/inference_pipeline.typ`
- `paper/figures/inference_pipeline.svg`
- `scripts/make_paper_diagrams.sh`
- `paper/vendor/typst/packages/`（vendored Fletcher/CeTZ/oxifmt）

### 3.2 重新生成图表

```bash
bash scripts/make_paper_diagrams.sh
```

或：

```bash
make paper-diagrams
```

### 3.3 重新编译论文

```bash
make paper
```

或者直接：

```bash
/usr/local/bin/typst compile paper/paper.typ paper.pdf
```

图纸使用 `paper/vendor/typst/packages` 作为 Typst package cache，避免网络依赖：

```bash
TYPST_PACKAGE_CACHE_PATH="$PWD/paper/vendor/typst/packages" \
  typst compile paper/figures/architecture_diagram.typ paper/figures/architecture_diagram.svg
```

---

## 4. 当前论文新增图

1. **Figure 1：System Architecture**
   - 主线：Query → Task Router → PLE Projector → Token Policy → Logit Fusion → Output
   - 三通道：BM25/RAG、PLE/Engram、Backbone+MoRA

2. **Figure 2：Inference Pipeline**
   - Query → Task Classifier → BM25+PLE Retrieval → Memory Feature Extraction
   - Token Policy 决策：yes 走 PLE Projector，no 走 Base Logits
   - 最终 Logit Fusion → Decode/Generated Text

---

## 5. 后续可扩展

- 增加 Algorithm 1/2 伪代码；
- 增加 memory size / n-gram order / data size 敏感性图；
- 增加 case study 示意图；
- 在 CI 中加入 Typst 图表与 PDF 构建检查。
