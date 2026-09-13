#!/usr/bin/env python3
"""Round 168 Stage 1.5f: what does the *addressing* step destroy?

The capability ladder has four rungs (the trigram ceiling; the addressing from a
trigram to the 16 row ids; the content of the frozen rows; the read-out).
Rung 2 was never isolated, and the natural guess -- 6.15e10 bigram keys and
1.53e16 trigram keys collapsed into 3.200014e8 rows -- is that it must lose a
lot.  This script measures it exactly, with no reader and no model:

* **resolution**: distinct trigrams vs distinct 16-tuples on a real stream.  The
  tuple is a function of the trigram, so equality means the addressing is
  injective on that stream;
* **continuation top-1**: answer held-out positions from the modal continuation
  of the trigram key, then from the modal continuation of the tuple key, both
  fitted on the same training half.  A lossless addressing makes these
  *identical*; the gap is what the addressing threw away.

And the control that makes the reading meaningful: the same measurements on
deliberately crushed specs (``crushed_spec``), on the same corpus.  A statistic
that cannot see the crushed case cannot certify the real one.

CPU only, no GPU, no training.  Pre-registration: the measurement and its
two-way validation are stated in ``docs/round-168-stage1.5f-preregistration.md``.

Usage::

    python scripts/round168_addressing_resolution.py --tag wiki \
        --eval-npy data/phase1/wikitext-heldout-decon/tokens.npy \
        --workdir outputs/round168/addressing
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen35_ple.addressing import (
    crushed_spec,
    evaluate_addressing,
    structured_stream,
)
from qwen35_ple.ple_hash import real_spec

#: Crush factors for the control arm.  The head modulus is ~2.0e7; a factor of
#: 2e6 leaves 10 rows per head, which is small enough to collide on any real
#: stream and is what makes the "lossless" reading on the real spec non-vacuous.
CRUSH_FACTORS: tuple[int, ...] = (1000, 100_000, 2_000_000)


def log(msg: str) -> None:
    print(f"[r168-1.5f] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--eval-npy", required=True)
    ap.add_argument("--workdir", default="outputs/round168/addressing")
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="0 = the whole stream")
    ap.add_argument("--train-fraction", type=float, default=0.6)
    args = ap.parse_args()

    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)

    tokens = np.load(args.eval_npy).astype(np.int64)
    if args.max_tokens and tokens.shape[0] > args.max_tokens:
        tokens = tokens[: args.max_tokens]
    log(f"{args.tag}: {tokens.shape[0]:,} tokens, "
        f"{int(tokens.max())} max token id")

    specs: list[tuple[str, Any]] = [("real", real_spec())]
    for factor in CRUSH_FACTORS:
        specs.append((f"crushed_{factor}", crushed_spec(factor)))

    arms: list[dict[str, Any]] = []
    for name, spec in specs:
        t0 = time.time()
        res = evaluate_addressing(tokens, spec, train_fraction=args.train_fraction)
        res["arm"] = name
        res["total_rows"] = int(spec.total)
        arms.append(res)
        log(f"{name}: rows {spec.total:,} | distinct trigrams "
            f"{res['resolution']['n_distinct_trigrams']:,} -> tuples "
            f"{res['resolution']['n_distinct_tuples']:,} | injective "
            f"{res['resolution']['injective']} | top1 tri "
            f"{res['trigram_top1']['accuracy']:.4f} vs tuple "
            f"{res['row_tuple_top1']['accuracy']:.4f} | gap "
            f"{res['top1_gap']:+.4f} ({time.time() - t0:.1f}s)")

    real = arms[0]
    worst = arms[-1]
    # The top-1 statistic needs a corpus whose continuation is genuinely
    # context-determined; WikiText is not one (measured trigram top-1 = 8.4%,
    # i.e. the modal continuation is the same token almost everywhere).  Run the
    # same real/crushed pair on a phrase stream to demonstrate the top-1 arm
    # works where it applies, instead of silently reporting a null.
    phrase = structured_stream(seed=0)
    applic = {
        "n_tokens": int(phrase.shape[0]),
        "real": evaluate_addressing(phrase, real_spec()),
        "crushed": evaluate_addressing(phrase, crushed_spec(2_000_000)),
    }
    log(f"applicability (phrase stream, {applic['n_tokens']:,} tokens): real gap "
        f"{applic['real']['top1_gap']:+.4f} vs crushed gap "
        f"{applic['crushed']['top1_gap']:+.4f}")

    out: dict[str, Any] = {
        "tag": args.tag,
        "eval_npy": args.eval_npy,
        "n_tokens": int(tokens.shape[0]),
        "train_fraction": args.train_fraction,
        "crush_factors": list(CRUSH_FACTORS),
        "arms": arms,
        "applicability": {
            "note": (
                "top-1 sensitivity is corpus-dependent; demonstrated on a "
                "phrase stream because this corpus is Zipf-like"
            ),
            "n_tokens": applic["n_tokens"],
            "real_top1_gap": applic["real"]["top1_gap"],
            "real_top1_accuracy": applic["real"]["trigram_top1"]["accuracy"],
            "crushed_top1_gap": applic["crushed"]["top1_gap"],
            "crushed_lossy": not applic["crushed"]["resolution"]["injective"],
        },
        "validation": {
            # Validity of the RESOLUTION statistic.  It is an exact enumeration,
            # so its "sensitivity" is not a statistical property: the control
            # arms simply show that the same enumeration registers a loss when a
            # loss exists.
            "real_injective": real["resolution"]["injective"],
            "real_top1_gap": real["top1_gap"],
            "control_lossy": not worst["resolution"]["injective"],
            "control_lost_tuples": worst["resolution"]["lost_tuples"],
            "control_top1_gap": worst["top1_gap"],
            "resolution_sensitive": (
                (not worst["resolution"]["injective"])
                and worst["resolution"]["lost_tuples"] > 0
            ),
            "top1_sensitive_on_this_corpus": worst["top1_gap"] > 0.05,
        },
        "conclusion": None,
    }
    v = out["validation"]
    if v["real_injective"] and v["resolution_sensitive"]:
        out["conclusion"] = (
            "ADDRESSING_IS_LOSSLESS: on this stream the 16-head tuple is "
            "injective on distinct trigrams (exact enumeration, no estimator), "
            "and the same enumeration registers a large loss on deliberately "
            "crushed specs over the identical corpus.  The gap between the "
            "trigram ceiling and what the rows deliver is therefore NOT an "
            "addressing loss."
        )
    elif v["resolution_sensitive"]:
        out["conclusion"] = (
            "ADDRESSING_LOSES_INFORMATION: resolution is not injective on this "
            "stream; see the per-arm table for how much."
        )
    else:
        out["conclusion"] = (
            "CONTROL_FAILED: the enumeration did not register the crushed spec, "
            "so no claim about the real spec is licensed."
        )
    log(out["conclusion"])

    json_path = work / f"{args.tag}-addressing.json"
    json_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    md_path = work / f"{args.tag}-addressing.md"
    md_path.write_text(render_markdown(out), encoding="utf-8")
    log(f"wrote {json_path} and {md_path}")
    return 0


def render_markdown(out: dict[str, Any]) -> str:
    lines = [f"# Stage 1.5f addressing resolution — {out['tag']}", ""]
    lines.append(f"* stream: `{out['eval_npy']}` ({out['n_tokens']:,} tokens)")
    lines.append(f"* train/test split: contiguous, {out['train_fraction']:.0%} train")
    lines.append("")
    lines.append("## Arms")
    lines.append("")
    lines.append("| arm | total rows | distinct trigrams | distinct tuples | injective | "
                 "top-1 from trigram | top-1 from tuple | gap |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for a in out["arms"]:
        r = a["resolution"]
        lines.append(
            f"| {a['arm']} | {a['total_rows']:,} | {r['n_distinct_trigrams']:,} | "
            f"{r['n_distinct_tuples']:,} | {r['injective']} | "
            f"{a['trigram_top1']['accuracy']:.4f} | "
            f"{a['row_tuple_top1']['accuracy']:.4f} | {a['top1_gap']:+.4f} |"
        )
    lines.append("")
    lines.append("## Two-way validation")
    lines.append("")
    v = out["validation"]
    lines.append(f"* real spec injective: **{v['real_injective']}** "
                 f"(top-1 gap {v['real_top1_gap']:+.4f})")
    lines.append(f"* crushed control lossy: **{v['control_lossy']}** "
                 f"({v['control_lost_tuples']:,} trigrams lost their own tuple; "
                 f"top-1 gap {v['control_top1_gap']:+.4f})")
    lines.append(f"* resolution statistic sensitive: "
                 f"**{v['resolution_sensitive']}**")
    lines.append(f"* top-1 statistic sensitive on THIS corpus: "
                 f"**{v['top1_sensitive_on_this_corpus']}**")
    lines.append("")
    lines.append("## Conclusion")
    lines.append("")
    lines.append(f"**{out['conclusion']}**")
    lines.append("")
    lines.append("## Honesty")
    lines.append("")
    lines.append("* Injection is decided on a *function of the key*: a tuple is "
                 "always a function of the trigram, so `distinct tuples <= "
                 "distinct trigrams` holds by construction and equality is the "
                 "only way to be injective.")
    ap = out.get("applicability")
    if ap:
        lines.append(
            f"* **Top-1 needs a structured corpus.** On this stream the crushed "
            f"spec keeps its top-1 "
            f"(gap {out['validation']['control_top1_gap']:+.4f}) because the "
            f"modal continuation is the same token almost everywhere. The "
            f"applicability arm runs the same real/crushed pair on a phrase "
            f"stream, where the crushed gap is {ap['crushed_top1_gap']:+.4f} "
            f"against a real gap of {ap['real_top1_gap']:+.4f}. The resolution "
            f"statistic has no such limitation and is what the verdict rests on."
        )
    lines.append("* This says nothing about whether a *row* learned the right "
                 "value: it is rung 2 only.  Rung 3 is round 161's table probe "
                 "(~59% of the trigram top-1) and rung 4 is the read-out.")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
