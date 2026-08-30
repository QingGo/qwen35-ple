# Official PLE forward golden

- `official_ple_forward_4096.npz` — 4096-token official Qwen4-Exp PLE forward fixture.
- `official_ple_forward_4096.meta.json` — hyperparameters, primes, multipliers, source.

Regenerate with:

```bash
source <conda env with torch>
PYTHONPATH=src python scripts/generate_official_ple_forward_golden.py
```

The fixture is generated from the frozen upstream snapshot
`src/qwen35_ple/official_ple_snapshot.py`, which in turn is extracted from
`refs/qwen4_exp_modeling.py` by `scripts/generate_official_ple_snapshot.py`.
