# Official Qwen4 PLE reference snapshot

This directory pins the upstream Qwen4-Exp PLE modeling source used to build
the M1 official-forward golden in this repository.

## Snapshot

| File | SHA-256 | Source commit |
|---|---|---|
| `qwen4_exp_modeling.py` | `91e9b1e9c74efe373cd989fe1974a8fa305f4aad43628dbcbd03dac20437814f` | EngramDB `13aaa3773bcb7fba2c820277d7ca9d9af3aa0c89` |

- Upstream file: `refs/qwen4_exp_modeling.py` in the EngramDB repository.
- Upstream location in transformers: generated from
  `src/transformers/models/qwen4_exp/modular_qwen4_exp.py`.
- License: Apache-2.0. This copy is reference-only and is not imported as a
  production dependency.

## Local extracted snapshot

- `src/qwen35_ple/official_ple_snapshot.py` is automatically generated from
  `qwen4_exp_modeling.py` by `scripts/generate_official_ple_snapshot.py`.
- It contains only the classes/functions required for the PLE forward golden:
  `Qwen4ExpTextRMSNorm`, `Qwen4ExpTextNGramEmbedding`,
  `Qwen4ExpTextPLELayer`, and the hash/multiplier helpers.

## Regeneration

```bash
python scripts/generate_official_ple_snapshot.py           # write snapshot
python scripts/generate_official_ple_snapshot.py --check   # verify snapshot
```

When the upstream file changes intentionally, update the snapshot and the
checksum in this README at the same commit.
