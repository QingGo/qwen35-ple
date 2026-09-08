# Phase 2 full matrix artifacts (AutoDL RTX 4090)

Run: 2026-09-08 23:05 CST -> 2026-09-09 03:44 CST

Remote paths:

```text
outputs: /root/autodl-tmp/qwen35-ple/outputs/phase2-full
log:     /root/autodl-tmp/qwen35-ple/logs/phase2-full.log
```

The instance was rebooted at ~07:20 CST on 2026-09-09. All six corpus JSONs
finished before the scheduled 04:00 shutdown; there was no running process and
no `Traceback`/CUDA OOM/error line in the log.

Files in this directory:

- `phase1-<CORPUS>.json.gz` — compressed raw result JSONs, including all
  per-seed training/validation curves and all 150 QA generations per mode.
- `phase2-full.log.gz` — full matrix log.
- `summary.json` / `summary.md` — output of
  `scripts/summarize_phase1_matrix.py --protocol v1` (legacy last-sentence
  extractor).
- `summary-v2.json` / `summary-v2.md` — output of
  `scripts/summarize_phase1_matrix.py --protocol v2` (answer-marker /
  first-sentence extractor).
- `sha256.txt` — SHA-256 of every committed artifact.

Decompress:

```bash
for f in phase1-*.json.gz; do gzip -dk "$f"; done
gzip -dk phase2-full.log.gz
```

Verify:

```bash
sha256sum -c sha256.txt
```

Important caveat: the QA protocol used raw completion prompts and the stored
`correct` field is a normalized *contains* match. The `contains` metric is not
a valid yes/no metric because a generation that lists both options can be
counted as correct. Treat the QA numbers as diagnostic only until the answer
extraction protocol is fixed.
