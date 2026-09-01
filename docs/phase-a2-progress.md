# Phase A2 进度：Store-P + access-order 1M 三臂复跑

> 状态：**运行中**（2026-09-01 22:24 UTC+8 WSL 启动）。
> 本文件记录当前进度、产物与后续收尾，正式结果完成后合并到
> `docs/phase-a-1m-result.md` 或单独结果文档。

## 目标

用 Store-P 物化视图 + 通用 slot-index + access-order 自动调度复跑
Phase A 的 1M real/control/no-reader 三臂实验，确认磁盘路径不改变
科学结论，并在同一 JSON 中记录每个 arm 的 fetch timing。

## 已完成

### 1. Store-P access-order 视图构建

- 输入：`data/wet-1m-tokens.npy`（1,540,336 token）
- 输出：
  - `outputs/phase-a2-1m.view`（3.7 GB，2560B slot）
  - `outputs/phase-a2-1m.keys`（227 MB，16 rowid/token）
  - `outputs/phase-a2-1m.slot_indices.npy`
  - `outputs/phase-a2-1m.slot_index.npz`
- 耗时：`engramdb view build --keys-stream --verify` 约 346s。
- 校验：view verify 抽样 1000 grams 全部匹配。

### 2. 实验代码增强

- `LiveETViewStore.reset_stats()` 改为原地重置共享 `FetchStats`，
  避免 view 子对象继续写旧 stats。
- `run_phase0.py` 每个 arm 输出：
  - `fetch_stats`
  - `fetch_ms_per_window`
  - `fetch_ms_per_token`
- 相关提交：
  - `c1b1d09` fetch timing 记录
  - `6ec6f37` 修复 reset_stats 原地重置
  - `9a21f38` Store-P 构建器原生/回退磁盘索引

## 运行命令

```bash
cd /home/zeng/qwen35-ple
.venv/bin/python -u scripts/run_phase0.py \
  --live-store \
  --store-p-view outputs/phase-a2-1m.view \
  --store-p-slot-index outputs/phase-a2-1m.slot_index.npz \
  --access-order \
  --tokens-npy data/wet-1m-first.npy \
  --rows-dir /home/zeng/qwen38-rows \
  --model-dir data/models/Qwen3.5-0.8B \
  --reader official \
  --official-reader-path data/official_ple_reader.pt \
  --steps 500 --seq-len 128 --lr 1e-4 \
  --seeds 0 1 2 \
  --modes real control no-reader \
  --output outputs/phase-a2-storep-accessorder-3seed.json
```

## 产物

- 结果 JSON：`outputs/phase-a2-storep-accessorder-3seed.json`（尚未生成）
- 日志：`/home/zeng/phase_a2_run.log`
- 小规模冒烟：`outputs/phase-a2-smoke2.json`（已确认 fetch_stats 非零）

## 下一步

1. 等待完整运行结束。
2. 汇总 real/control/no-reader 的 val_loss、PPL、fetch timing。
3. 与 Phase A Store-I 结果对比。
4. 将正式结论写入 `docs/phase-a-1m-result.md` / roadmap。
