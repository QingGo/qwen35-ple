#!/usr/bin/env python3
"""Offline harness self-test for ``scripts/probe_table_next_token.py``.

The real experiment needs the remote row table, but that is not where the bugs
live: the risky parts are the streaming memory discipline, the ridge probe, the
count-model indexing and -- above all -- whether the shuffled-row controls
actually collapse.  This script stubs out transformers / engramdb / torch and
injects a *synthetic* row table, then runs the whole of ``main()`` locally.

Two modes:

    python3 scripts/selftest_probe_table.py signal
        the synthetic rows ARE linearly decodable from the target
        -> the probe MUST beat the floor and the verdict must NOT be
           NO_RECOVERABLE_SIGNAL

    python3 scripts/selftest_probe_table.py noise
        the synthetic rows are pure noise
        -> the probe MUST collapse to the floor and the verdict MUST be
           NO_RECOVERABLE_SIGNAL

Exit code is non-zero if the expected behaviour is not observed, so this can be
used as a gate before spending a real run.  Extra CLI flags can be appended via
the ``EXTRA`` environment variable, e.g.
``EXTRA='--state-file /tmp/s.npz' python3 scripts/selftest_probe_table.py signal``.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SCRATCH = REPO / "outputs" / "_probe_selftest"
SCRATCH.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))

# ---- stub transformers (imported inside main()) ---------------------------- #
fake_tf = types.ModuleType("transformers")


class FakeTok:
    vocab_size = 0
    eos_token_id = 0
    _doc: list[int] = []

    def encode(self, text, add_special_tokens=False):
        return list(FakeTok._doc)

    def decode(self, ids, **kw):
        return "<tok>"


class _AutoTok:
    @staticmethod
    def from_pretrained(*a, **k):
        return FakeTok()


fake_tf.AutoTokenizer = _AutoTok
sys.modules["transformers"] = fake_tf

# ---- stub the torch-bearing qwen35_ple package, keep the dependency-free spec #
ph_spec = importlib.util.spec_from_file_location("_ph", REPO / "src/qwen35_ple/ple_hash.py")
ph = importlib.util.module_from_spec(ph_spec)
sys.modules["_ph"] = ph
ph_spec.loader.exec_module(ph)

pkg = types.ModuleType("qwen35_ple")
pkg.__path__ = []
mod_hash = types.ModuleType("qwen35_ple.ple_hash")
mod_hash.real_spec = ph.real_spec
mod_real = types.ModuleType("qwen35_ple.real_ple")
mod_real.real_spec = ph.real_spec
mod_real.rowids_from_tokens = lambda toks: np.asarray(
    ph.real_spec().rowids_for_seq(np.asarray(toks).reshape(-1).tolist()), dtype=np.int64
)
pkg.ple_hash, pkg.real_ple = mod_hash, mod_real
sys.modules["qwen35_ple"] = pkg
sys.modules["qwen35_ple.ple_hash"] = mod_hash
sys.modules["qwen35_ple.real_ple"] = mod_real

spec = importlib.util.spec_from_file_location(
    "pbt", str(HERE / "probe_table_next_token.py")
)
pbt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pbt)

# ---- synthetic world ------------------------------------------------------- #
MODE = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "signal"
V, T = 40, 120_000
rng = np.random.default_rng(7)
toks = np.zeros(T, dtype=np.int64)
for i in range(2, T):
    toks[i] = (toks[i - 1] * 7 + toks[i - 2] * 3 + (i % 5)) % V
FakeTok._doc = toks[T // 2 :].tolist()
FakeTok.vocab_size = V
np.save(SCRATCH / "tokens.npy", toks)
(SCRATCH / "dummy.jsonl").write_text('{"text": "x"}\n', encoding="utf-8")

Z = rng.normal(size=(V, 160)).astype(np.float32)


def rows_for(tokens, positions):
    a = Z[tokens[positions]]
    b = Z[tokens[np.maximum(positions - 1, 0)]]
    c = Z[tokens[np.maximum(positions - 2, 0)]]
    return np.tile(np.concatenate([a, b, c], axis=1), (1, 6))[:, :2560].astype(np.float32)


# main() re-encodes each held-out record and appends eos (=0) after each one
eval_tokens = np.concatenate([np.append(toks[T // 2 :], 0) for _ in range(8)])
if MODE == "signal":
    train_rows = rows_for(toks, np.arange(T))
    eval_rows = rows_for(eval_tokens, np.arange(len(eval_tokens)))
else:
    train_rows = rng.normal(size=(T, 2560)).astype(np.float32)
    eval_rows = rng.normal(size=(len(eval_tokens), 2560)).astype(np.float32)

# ---- patch the IO boundaries ---------------------------------------------- #
pbt.build_heldout_indices = lambda **kw: (["d"] * 8, np.arange(8), {
    "records_total": 8, "records_selected_reproduced": 0, "records_heldout": 8,
    "tokens_selected_reproduced": 0, "qa_filtered_records": 0,
    "reproduction_matches_manifest": True,
    "manifest_expected_records": 0, "manifest_expected_tokens": 0,
})
pbt.containment_check = lambda *a, **k: {"skipped": True}
pbt.verify_rows_dir = lambda *a, **k: {
    "dir": "stub", "shard_files_found": 128, "expected_shards": 128,
    "expected_shard_bytes": 400_001_920, "n_shards_wrong_size": 0,
    "shards_wrong_size_examples": [], "total_files_in_dir": 129,
    "unrecognised_bin_files": [], "total_bytes": 128 * 400_001_920,
    "allow_partial": False, "problems": [],
}


def patched_make_fetcher(rows_dir, scale, block):
    stats = {"rows": 0, "zero_rows": 0, "finite": True, "seconds": 0.0}

    def fetch(rd):
        rd = np.asarray(rd, dtype=np.int64)
        pos, tag = rd[:, 0], rd[:, 1]
        out = np.empty((len(pos), 2560), dtype=np.float32)
        m = tag == 0
        out[m] = train_rows[pos[m]]
        out[~m] = eval_rows[pos[~m]]
        stats["rows"] += len(pos)
        return out

    return fetch, types.SimpleNamespace(close=lambda: None), stats


pbt.make_fetcher = patched_make_fetcher

_orig_rowids = pbt.rowids_at_positions


def patched_rowids(tokens, positions, **kw):
    # the function verifies itself by recursing; let that recursion see the real
    # implementation so the column-0 position stash does not corrupt the check
    pbt.rowids_at_positions = _orig_rowids
    try:
        out = _orig_rowids(tokens, positions, **kw).astype(np.int64).copy()
    finally:
        pbt.rowids_at_positions = patched_rowids
    out[:, 0] = positions                      # stash the position
    out[:, 1] = 0 if len(tokens) == T else 1   # stash the stream id
    return out


pbt.rowids_at_positions = patched_rowids

sys.argv = [
    "probe", "--train-tokens", str(SCRATCH / "tokens.npy"),
    "--wikitext", str(SCRATCH / "dummy.jsonl"),
    "--pure-wiki-corpus", str(SCRATCH / "nope.txt"),
    "--manifest", str(SCRATCH / "nope.json"),
    "--tokenizer", str(REPO), "--qa-exclude", str(SCRATCH / "no-qa.json"),
    "--skip-value-proj", "--topk", str(V),
    "--n-probe-train", "3000", "--n-probe-val", "800", "--n-eval", "1500",
    "--n-dev", "600", "--chunk", "500", "--rowid-window", "4096",
    "--dev-tokens", "4000", "--out", str(SCRATCH / f"out-{MODE}.json"),
]
EXTRA = os.environ.get("EXTRA", "").split()
sys.argv += EXTRA
print(f"[selftest] mode={MODE} extra={EXTRA}", flush=True)

try:
    rc = pbt.main()
except SystemExit as exc:
    print(f"[selftest] SystemExit: {exc}")
    sys.exit(2)

import json

report = json.loads((SCRATCH / f"out-{MODE}.json").read_text(encoding="utf-8"))
verdict = report["verdict"]["label"]
ev = report["results"]
probe = ev["probe_raw_rows"]["eval"]["top1"]
tri = ev["count_trigram"]["eval"]["top1"]
floor = report["control_floor"]
collapsed = report["controls_collapsed"]
print("\n[selftest] ---------------------------------------------")
print(f"[selftest] probe={probe:.4f} trigram={tri:.4f} control_floor={floor:.4f} "
      f"collapsed={collapsed} verdict={verdict} peak_rss_mb={report['peak_rss_mb']:.0f}")

ok = True
if MODE == "signal":
    if verdict == "NO_RECOVERABLE_SIGNAL":
        print("[selftest] FAIL: decodable rows were reported as no-signal")
        ok = False
    if probe <= floor + 0.02:
        print("[selftest] FAIL: probe did not clear the floor on decodable rows")
        ok = False
else:
    if verdict != "NO_RECOVERABLE_SIGNAL":
        print("[selftest] FAIL: noise rows were reported as carrying signal")
        ok = False
if not collapsed:
    print("[selftest] FAIL: shuffled controls did not collapse -- pipeline leaks")
    ok = False

print("[selftest] PASS" if ok else "[selftest] FAILED")
sys.exit(0 if ok else 1)
