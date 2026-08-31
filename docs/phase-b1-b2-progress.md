# Phase B1/B2 Progress

## Phase B1: official model loading without allocating the giant PLE table

Landed in `EngramDB`/`qwen35-ple`:

- `engramdb.official_loader.patch_official_ngram_embedding_for_disk_load`
  - Context manager that patches the official `Qwen4ExpTextNGramEmbedding`
    constructor so `nn.Embedding` is created with a one-row placeholder instead
    of the full multi-hundred-GB PLE table.
- `engramdb.official_loader.load_official_checkpoint_without_ngram_shards`
  - Streams safetensors shards with `safe_open`, skips every
    `ngram_embedding.*` tensor, and loads only non-PLE weights.
- `scripts/qwen4_ple_custom_loader.py --load-model`
  - Uses the placeholder patch before `AutoModelForCausalLM.from_config`,
    loads the filtered checkpoint, then calls
    `install_disk_ple_in_official_model`.

Structural smoke without requiring a Qwen4Exp-enabled Transformers build:

```bash
PYTHONPATH=src:../EngramDB/python \
python scripts/qwen4_ple_official_loader_smoke.py
# OFFICIAL_SNAPSHOT_DISK_PLE_STRUCTURE_OK
```

## Phase B2: small-table bit-exact with the frozen official PLE class

`DiskPleNGramEmbedding` now supports:

- custom prime sizes / head offsets / padded vocabulary,
- per-batch n-gram context,
- batched forward with output shape `[batch, seq, embedding_dim]`.

The frozen official `Qwen4ExpTextNGramEmbedding` and the EngramDB disk adapter
are bit-exact on a small synthetic table:

```bash
PYTHONPATH=src:../EngramDB/python \
python scripts/qwen4_ple_bit_exact_small.py
# OFFICIAL_DISK_PLE_BIT_EXACT_SMALL_OK
```

Covered: batched input, an EOS token inside a sequence, and chunked/streaming
calls using per-batch internal context.

## Low-resource real-row oracle

Without loading the full 48GB Store or PLE embedding, the sparse oracle reads
only the rows touched by a small fixed token sequence from the original
checkpoint and compares them with EngramDB Store-I:

```bash
PYTHONPATH=src:../EngramDB/python \
python scripts/sparse_real_row_oracle.py \
    --checkpoint "/Volumes/My Passport/qwen38-ple" \
    --store "/Volumes/My Passport/qwen38-rows"
```

Result:

```text
[oracle] tokens=9 unique_rows=144
[oracle] all 144 real rows byte-identical
[oracle] DiskPle real-Store maxdiff vs checkpoint rows: 0.0
SPARSE_REAL_ROW_ORACLE_OK
```

This proves:
- real checkpoint FP8 rows match EngramDB Store-I byte-for-byte;
- `DiskPleNGramEmbedding` reading the real Store produces the same dequantized
  values as the same rows read directly from the checkpoint.

## Prefetch A/B smoke

```bash
PYTHONPATH=src:../EngramDB/python \
python scripts/prefetch_real_ab.py --tokens 64 --compute-ms 30
```

Result:

```text
[sync]     total=192.390ms fetch_s=188.817ms
[prefetch] total=34.117ms fetch_s=1.434ms wait_s=0.028ms issued=1024
[delta]    sync - prefetch = 158.273ms
PREFETCH_AB_SMOKE_OK
```

This shows the new prefetch pipeline can hide disk fetch behind a simulated
earlier-layer compute window.

## Mini official-model prefetch A/B (real Store + frozen official PLE layer)

This step replaces the pure sleep-based microbench with an actual nn.Module
forward path.  It uses:

- the frozen official `Qwen4ExpTextPLELayer` as the real PLE layer,
- `DiskPleNGramEmbedding` backed by the real EngramDB Store,
- synthetic dense blocks before/after PLE to represent earlier/later compute,
- the same model-level prefetch hook used by the full loader.

```bash
PYTHONPATH=src:../EngramDB/python \
python scripts/mini_official_prefetch_ab.py \
    --checkpoint "/Volumes/My Passport/qwen38-ple" \
    --store "/Volumes/My Passport/qwen38-rows" \
    --tokens 8 --hidden 64 --pre-layers 2 --post-layers 1 --reps 2 \
    --csv /tmp/mini_ab.csv
# MINI_OFFICIAL_PREFETCH_AB_OK
```

The script records:

- total forward wall time,
- pre-PLE compute time and actual PLE-layer time,
- prefetch issued rows, prefetch wait time, synchronous fetch time,
- whether the prefetch had already completed when the PLE layer was entered,
- optional CSV for later regression thresholds.

This is still a low-resource smoke, not a full Qwen4Exp end-to-end tok/s result.
It also exposed/fixed two production details:

- the model-level pre-hook now tolerates both `hook(module, args)` and
  `hook(module, args, kwargs)` calling conventions across PyTorch versions;
- `DiskPleEmbedding.close()` / `DiskPleNGramEmbedding.close()` now shut down the
  prefetch executor and can be used in benchmark scripts and long-running
  services without leaking worker threads.

## Fast e_t fetch path (Store.fetch + torch tensor)

The 20k-token precompute benchmark showed that the old `PleDiskGather.fetch`
Python byte-expansion path was the bottleneck:

- `PleDiskGather.fetch` (old): 16.857s for 320k rows
- `Store.fetch` direct: 0.562s for 320k rows
- `Store.fetch` + torch conversion: 0.808s end-to-end

This has now been addressed in EngramDB:

- `PleDiskGather.fetch` no longer does Python per-row dedup/slice/join; it
  returns the contiguous `Store.fetch` buffer directly.
- New `engramdb.fetch_e_t_tensor(store, rowids, ...)` returns a torch tensor
  with shape `[T, 16, 160]` from one `Store.fetch` call, optionally scaled.
- `PleDiskGather.fetch_tensor(...)` is the same fast path through the gather
  helper.
- `qwen35_ple.real_ple.fetch_e_t` and
  `scripts/precompute_real_ple_features.py` now use the direct tensor path.
- `scripts/run_phase0.py --live-store` can read PLE rows live from EngramDB
  instead of loading a precomputed `e_t.npy`.

```bash
PYTHONPATH=src:../EngramDB/python \
python scripts/run_phase0.py --live-store \
    --tokens-npy /tmp/tokens.npy \
    --rows-dir "/Volumes/My Passport/qwen38-rows" \
    --model-dir "/Volumes/My Passport/qwen38-ple"
```


Still pending:
- Full official Qwen4Exp model load/memory verification on a Transformers build
  that ships Qwen4-Exp.
- MTP / Transformers `Cache` streaming integration.
- Official-class-forward over real rows without a full in-memory PLE table
  (currently covered indirectly by small-table official bit-exact + real-row
  DiskPle dequant oracle).
