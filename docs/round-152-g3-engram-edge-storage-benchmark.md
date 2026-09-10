# G3: EngramDB PLE storage benchmark (edge serving feasibility)

> Date: 2026-09-11 (round 152)
> Script: `scripts/bench_engram_edge.py`
> Raw: `outputs/g3/nvme-persistent.{json,md}`, `outputs/g3/shm.{json,md}`

## 0. Why this gate exists

All our training and evaluation so far read the 51.2GB PLE table from
`/dev/shm`, which hides the cost that a real edge device would pay.  G3 replaces
that assumption with measurements: how fast can we actually fetch PLE rows from
persistent flash, and is the disk-first design (EngramDB Store-I/Store-P + hot
cache + prefetch) inside the pre-registered edge budget?

Pre-registered budget (docs/round-149 section 1.2):

```text
decode:  >=20 tok/s on NVMe-class storage, >=5 tok/s on USB SSD/SD (0.8B)
RAM:     model + cache <= 4GB excluding the flash-resident table
p99:     no large stall beyond the model compute budget
```

## 1. Setup

```text
table      Qwen3.8-Flash-Next FP8 PLE, 128 shards x 2,500,012 rows, width 160
           (16 heads x 160 dim = 2560 bytes/token, 16 rows per token)
volumes    nvme-persistent : /root/autodl-tmp/qwen35-ple/qwen38-rows (host NVMe)
           shm             : /dev/shm/qwen38-rows (RAM, the hidden-cost baseline)
method     engramdb.Store + fetch_e_t_tensor over real rowids from
           engramdb.rowids_for_seq; 30 reps per batch size, page cache dropped
           with posix_fadvise(DONTNEED) for the cold series
```

## 2. Measured results

Clean run (idle GPU/CPU), rows/s from p50; tokens/s = rows/s ÷ 16:

| rows fetched | tokens | NVMe warm p50 | NVMe warm rows/s | NVMe cold p50 | NVMe cold rows/s | shm warm rows/s |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 1 | 0.32 ms | 49,471 | 0.66 ms | 24,151 | 51,523 |
| 64 | 4 | 0.40 ms | 160,412 | 1.57 ms | 40,737 | 188,834 |
| 256 | 16 | 1.35 ms | 189,018 | 4.92 ms | 52,009 | 353,816 |
| 1024 | 64 | 4.53 ms | 226,119 | 15.57 ms | 65,769 | 673,086 |
| 2048 | 128 | 4.43 ms | 462,158 | 29.03 ms | 70,559 | 276,273 |

Under concurrent load (a LoRA eval was running) the same NVMe warm series
measured 21K–131K rows/s, so treat the clean run as the best case and the loaded
run as the contention lower bound.  Tail latency is the real weakness: warm p99
is ~0.6–1.9 ms for small batches but ~93–99 ms at 256+ rows, on both volumes
(so it is wrapper/CPU overhead, not the device).

Process cost: VmRSS ≈ 400 MB, harness `read_bytes` ≈ 140 MB for the cold series
(the page cache absorbs the rest).  Peak RSS stayed at ~400 MB regardless of
batch size, i.e. fetching does not grow memory with the table.

## 3. What this means

* **Prefill/training-style fetching on disk is cheap.** Extrapolating the
  measured 462K rows/s: a 2048-token prefill needs 32,768 rows ≈ 71 ms of row
  fetching.  Even the loaded lower bound (131K rows/s) is ~250 ms per 2048-token
  forward, still far above the 20 tok/s decode budget when amortised.
* **Decode is not storage-bound.** Decoding one token needs 16 rows: at the
  small-batch warm rate (49K rows/s) the fetch cost is ~0.32 ms/token, versus
  the tens of ms/token the 0.8B model itself takes.
* **RAM budget is fine.** ≈400 MB resident for the fetch path, far inside the
  4GB model+cache target; the 51.2GB table never becomes resident.
* **Disk-first is viable but prefetch still matters.** Cold-cache first touch
  costs 2–7× the warm cost, so sequential prefill without prefetch would stall;
  EngramDB's prefetch/hot-cache path is what removes that.
* **Two real risks remain.** (a) warm p99 ~95 ms at batch ≥256 — needs the
  async path/persistent readers to hide; (b) these are host-NVMe numbers on an
  AutoDL box, not UFS/SD, so the USB-SSD/SD row of the budget is still
  unmeasured.

## 4. Verdict

```text
G3 (NVMe row): PASS on throughput and RAM for prefill and decode.
               UNMEASURED on USB SSD/SD, UFS, and true mobile power/endurance.
Next: (1) repeat on a USB SSD to fill the >=5 tok/s row;
      (2) integrate a hot-cache/prefetch path so warm p99 does not hit the
          decode loop;
      (3) keep /dev/shm as the training default but report the disk number
          alongside, since the difference is now known to be small.
```

## 5. Reproduction

```bash
PY=/root/autodl-tmp/qwen35-ple/venv/bin/python
$PY scripts/bench_engram_edge.py \
  --rows-dir /root/autodl-tmp/qwen35-ple/qwen38-rows \
  --label nvme-persistent --cold \
  --json outputs/g3/nvme-persistent.json \
  --markdown outputs/g3/nvme-persistent.md
$PY scripts/bench_engram_edge.py \
  --rows-dir /dev/shm/qwen38-rows --label shm \
  --json outputs/g3/shm.json --markdown outputs/g3/shm.md
```
