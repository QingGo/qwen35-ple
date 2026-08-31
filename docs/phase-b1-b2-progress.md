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

Still pending:
- Full official Qwen4Exp model load/memory verification on a Transformers build
  that ships Qwen4-Exp.
- MTP / Transformers `Cache` streaming integration.
- Official-class-forward over real rows without a full in-memory PLE table
  (currently covered indirectly by small-table official bit-exact + real-row
  DiskPle dequant oracle).
