# EngramDB v0.2.9 重新测试指南（给 qwen35-ple Agent）

> 目标：验证 EngramDB v0.2.9 的快速 `Store.fetch` / `fetch_e_t_tensor` 路径，
> 并确认 qwen35-ple 的 live-store 与预计算读取不再走慢的 `PleDiskGather` Python 字节展开。

> **验收结果（2026-xx-xx，Mac + WSL）**
>
> - EngramDB v0.2.9 快速门禁通过：`python_wheel_smoke.py` ✅
> - qwen35-ple 正确性全部通过：official smoke、bit-exact、sparse oracle、phase B tests ✅
> - Mac 1000 tokens precompute：`fetch+dequant 0.67s` ✅
> - WSL live-store Phase 0：`live fetch 1.05s e_t=(1000,2560)`，模型加载并完成 no-reader ✅
> - 旧 `PleDiskGather(store` 在 qwen35-ple 主路径已无引用 ✅
>
> 注意：Mac 上 20k 级首次调用约 6–9s，热态约 0.55s；正式结论需以冷热分离后的 CSV 为准。


## 0. 版本确认

```bash
cd EngramDB
git log --oneline -1
# 应包含 release: bump v0.2.9
git tag --list 'v0.2.9'

cd qwen35-ple
git log --oneline -1
# 应包含 perf(precompute): use Store.fetch tensor fast path and add live-store
```

如果本地 EngramDB 不是 v0.2.9：

```bash
cd EngramDB && git fetch origin && git checkout v0.2.9
```

如果 qwen35-ple 不是最新：

```bash
cd qwen35-ple && git fetch origin && git checkout origin/main
```

## 1. 使用本地 EngramDB Python 包

不要依赖旧安装。直接用工作区源码：

```bash
cd EngramDB
PATH="$HOME/.cargo/bin:$PATH" CARGO_HOME="${CARGO_HOME:-/tmp/cargo-home}" \
  bash scripts/build_pyo3.sh
```

测试时设置：

```bash
export PYTHONPATH="src:/path/to/EngramDB/python"
```

## 2. 快速门禁（应该全绿）

```bash
cd EngramDB
cargo test --workspace

PYTHONPATH=python python scripts/python_wheel_smoke.py
```

其中应看到：

```text
PleDiskGather OK
DiskPleEmbedding prefetch OK
fetch_e_t_tensor OK
python wheel smoke OK
```

## 3. qwen35-ple 正确性测试

```bash
cd qwen35-ple
export PYTHONPATH="src:/path/to/EngramDB/python"

python scripts/qwen4_ple_official_loader_smoke.py
python scripts/qwen4_ple_bit_exact_small.py
python scripts/sparse_real_row_oracle.py \
    --checkpoint "/Volumes/My Passport/qwen38-ple" \
    --store "/Volumes/My Passport/qwen38-rows"
python tests/test_phase_b_official_loader.py -q
```

预期：

```text
OFFICIAL_SNAPSHOT_DISK_PLE_STRUCTURE_OK
OFFICIAL_DISK_PLE_BIT_EXACT_SMALL_OK
SPARSE_REAL_ROW_ORACLE_OK
3 passed
```

## 4. 20k 级预计算 / live 路径复测

### 4.1 预计算小样本

```bash
python - <<'PY'
import numpy as np
np.save('/tmp/retest_tokens.npy', np.arange(100, 1100, dtype=np.int64))
PY

PYTHONPATH=src:/path/to/EngramDB/python \
python scripts/precompute_real_ple_features.py \
    --tokens-npy /tmp/retest_tokens.npy \
    --rows-dir "/Volumes/My Passport/qwen38-rows" \
    --model-dir "/Volumes/My Passport/qwen38-ple" \
    --output /tmp/retest-ple-features
```

应看到：

```text
[precompute] fetch+dequant ...
[precompute] e_t shape=(1000, 2560) finite=True
```

### 4.2 直接速度对比

```bash
PYTHONPATH=src:/path/to/EngramDB/python python - <<'PY'
import time
import engramdb
from engramdb.vllm import PleDiskGather, fetch_e_t_tensor

store = engramdb.Store('/Volumes/My Passport/qwen38-rows', 128, 2_500_012, 160)
try:
    tokens = list(range(100, 20100))
    rowids = engramdb.rowids_for_seq(tokens)
    flat = [r for row in rowids for r in row]

    t = time.perf_counter()
    raw = PleDiskGather(store, 160).fetch(flat)
    t1 = time.perf_counter() - t

    t = time.perf_counter()
    arr = fetch_e_t_tensor(store, flat, num_heads=16, head_dim=160)
    t2 = time.perf_counter() - t

    print(f'PleDiskGather.fetch      {t1:.3f}s bytes={len(raw)}')
    print(f'fetch_e_t_tensor         {t2:.3f}s shape={tuple(arr.shape)}')
finally:
    store.close()
PY
```

注意：**结果会受数据库冷/热、USB/NVMe、系统分页缓存影响**。请记录介质与冷热状态。

### 4.3 live-store Phase 0

```bash
PYTHONPATH=src:/path/to/EngramDB/python \
python scripts/run_phase0.py --live-store \
    --tokens-npy /tmp/retest_tokens.npy \
    --rows-dir "/Volumes/My Passport/qwen38-rows" \
    --model-dir "/Volumes/My Passport/qwen38-ple" \
    --model data/models/Qwen3.5-0.8B \
    --steps 1 --seeds 0 --modes no-reader \
    --output /tmp/retest-live.json
```

预期先出现：

```text
[phase0] live-store: ... tokens, reading PLE rows from ...
[phase0] live fetch ... e_t=(..., 2560)
```

## 5. 旧慢路径确认已不再默认使用

在 qwen35-ple 中搜索：

```bash
grep -R "PleDiskGather(store" -n src scripts | grep -v "fetch_tensor" || true
```

`real_ple.fetch_e_t` 和 `precompute_real_ple_features.py` 现在都应使用
`engramdb.fetch_e_t_tensor`，而不是旧 `PleDiskGather.fetch` 的 Python 字节展开。
