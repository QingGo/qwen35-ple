# Round 109：Phase A PLE Projector v0

> 日期：2026-09-06
> 状态：完成 v0 实现与 seed0 首轮实验
> 目标：把“是否用 PLE”升级为“如何把 PLE 映射到 backbone logit 空间”。

---

## 1. 新增组件

### `src/qwen35_ple/projector.py`

- `PleProjector`：小 MLP，输入为：

  ```text
  冻结 backbone 的当前 hidden state (1024)
  + n-gram order / entropy / density ratio / agreement 等 7 个 memory 特征
  ```

  输出为：

  ```text
  scale
  bias
  ```

  融合方式：

  ```text
  fused_logits = base_logits + scale * log p_memory + bias
  ```

- 初始化 final head 为零，因此未训练时等价于 `base`，不会主动开启 PLE。
- 支持 `save_projector` / `load_projector` 持久化为 JSON。
- `compute_memory_features` 提供统一的特征提取。

### 训练脚本 `scripts/train_ple_projector.py`

- 冻结 backbone，只训练 projector；
- 优化目标：next-token cross-entropy；
- 自动对比三个条件：
  - `base`：不加 PLE；
  - `fixed`：全局 calibrated scale/bias；
  - `projector`：learned per-token scale/bias。
- 输出：
  - `outputs/ple-projector-v0-seed0.json` 实验报告；
  - `outputs/ple-projector-v0.json` 训练好的 projector artifact。

### Serving 集成

- `TaskConditionedNgramLogitProcessor` 新增可选 `projector`；
- `__call__` 支持传入 `hidden_state`；
- `RAGServingAdapter._generate` 现在会取最后一层 hidden state 并传给 logit processor；
- 配置可通过 `router.projector_path` 加载 projector。

---

## 2. Seed0 实验结果

配置：

```text
max_samples=100
steps=100
batch_size=4
hidden_dim=64
num_layers=1
device=cuda
```

### 总体 teacher-forced NLL / hit

| 条件 | NLL | hit |
|---|---:|---:|
| base | 2.5109 | 0.480 |
| fixed calibration | 2.0941 | 0.547 |
| learned projector | 2.1297 | 0.600 |

### 分任务

| 任务 | fixed NLL | projector NLL | fixed hit | projector hit | 结论 |
|---|---:|---:|---:|---:|---|
| code | 1.6321 | **1.4801** | 0.667 | **0.729** | projector 明显改进 |
| number | 1.5454 | 1.5855 | 0.438 | 0.438 | 基本持平 |
| name | **4.9082** | 5.7558 | 0.182 | 0.273 | projector 在 name 上变差 |

---

## 3. 解读

1. **Projector 已经学会条件化使用 PLE**：
   - code 上强于固定标定（NLL 1.48 vs 1.63，hit 0.73 vs 0.67）；
   - 总体 hit 也高于 fixed（0.60 vs 0.55）。
2. **当前 v0 还不是全面优于固定标定**：
   - 总体 NLL 略差于 fixed（2.13 vs 2.09）；
   - 主要拖累是 name 类：projector 在 name 上为了提升 hit 牺牲了概率校准。
3. **符合早期预期**：
   - 这是 v0，数据量只有 100 样本、100 步；
   - 固定标定是对当前小样本调出来的强基线；
   - 要宣布 projector 胜出，需要扩大数据、增加 seed、并加入生成质量指标。

---

## 4. 下一步

1. **3 seed + paired test**：运行 seed 0/1/2，计算 bootstrap/配对置信区间；
2. **扩大局部续写数据**：从 100 样本 → 1k/10k；
3. **加入任务条件/one-hot**：显式告诉 projector 当前是 code/name/number；
4. **分离任务评测**：不要在 name 上强求统一 projector，优先确认 code 局部续写；
5. **开放生成保护**：用现有 token policy + semantic task 路由测试 projector 不伤害开放生成；
6. **M2 规模放大**：1M tokens + LoRA/部分解冻。

---

## 5. 产物

```text
src/qwen35_ple/projector.py
scripts/train_ple_projector.py
tests/test_projector.py
docs/round-109-ple-projector-v0.md
outputs/ple-projector-v0-seed0.json
outputs/ple-projector-v0.json
```
