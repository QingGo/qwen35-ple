# Round 152: LoRA backbone co-adaptation row, G0 4B scale/space test, G3 storage benchmark

> Date: 2026-09-11 (overnight autonomous run)
> Status: LoRA row and G0 results are filled in below; G3 has its own document
> (`docs/round-152-g3-engram-edge-storage-benchmark.md`).
> Raw artifacts: remote `outputs/round152/`, `outputs/round152g0/`, `outputs/g3/`.

## 0. Why these three experiments

Round 149's full-FT row collapsed open QA in all three arms, so it could not test
whether backbone co-adaptation unlocks the frozen PLE table.  This round runs the
experiments that the falsification of frozen grafting left open:

```text
G1 (LoRA row)  {frozen, LoRA} x {no PLE, real, control} ->
               does a *gentle* adaptation keep the base model alive and create a
               real-vs-control content effect?
G0 (4B)        frozen PLE graft on Qwen3.5-4B (hidden 2560 == PLE source hidden)
               -> is the 0.8B failure a scale/space mismatch?
G3 (storage)   EngramDB row fetching on NVMe vs /dev/shm -> is disk-first edge
               serving inside the pre-registered budget?  (separate doc)
```

Pre-registered thresholds (unchanged): content effect `real − control` ≥2
accuracy points or ≥0.05 nat; both directions must beat the control arm.

## 1. G1: LoRA row of the 2×2

### 1.1 Design

```text
arms        lora-nople   : LoRA backbone + reader present, PLE contribution off
            lora-real    : LoRA backbone + real frozen PLE table
            lora-control : LoRA backbone + shuffled PLE rows (placebo)
recipe      500 steps, mixed50 QA SFT (6000 items, weight 0.5), lr 1e-4,
            seq-len 128, LoRA r=16 alpha=32 dropout 0, targets
            q/k/v/o/gate/up/down_proj (6.39M trainable), base weights frozen
eval        standard 1500 raw prompt: generation (exact match) + gold-answer NLL
```

Three bugs had to be fixed before this row was valid:

1. `mode=no-reader` returns before training, so the first version of the no-PLE
   cell was an *untrained* backbone; the cell now uses `--ple-off` (reader
   present, contribution suppressed) so all three cells train.
2. Suppressing the reader store also skipped the QA SFT cache, which silently
   trained the no-PLE cell on corpus loss only.  The SFT cache now builds from
   the tokenizer whenever a QA SFT file is given.
3. `_run_mode` referenced `main()`'s `lora_meta`, crashing every LoRA run that
   saved a checkpoint.

### 1.2 Results

### 1.2 Results (standard 1500, raw prompt)

| Arm | BoolQ | TriviaQA | NQ | Mean EM | Gold NLL |
|---|---:|---:|---:|---:|---:|
| frozen no-reader (R148-A ref) | 0.672 | 0.128 | 0.062 | 0.2873 | 8.7429 |
| lora-nople (PLE off) | 0.770 | 0.144 | 0.052 | **0.3220** | 2.4025 |
| lora-real | 0.764 | 0.128 | 0.054 | 0.3153 | 2.4025 |
| lora-control | 0.758 | 0.140 | 0.048 | 0.3153 | **2.3845** |

Paired item-level (positive = first run better):

```text
lora-real vs lora-control   EM  +0.0000 ± 0.0049   NLL −0.0180 ± 0.0068
lora-real vs lora-nople     EM  −0.0074 ± 0.0055   NLL +0.0008 ± 0.0088
```

Reading:

* **LoRA co-adaptation works**: +2.8 to +3.5 points over the frozen reader row
  (0.2873 → 0.3153–0.3220) and, unlike the full-FT recipe, it does not collapse
  open QA. This makes LoRA the usable adaptation path for the edge story.
* **The PLE content effect is still absent**: `lora-real` and `lora-control`
  generate *identical* task accuracies, gold NLL slightly favours the shuffled
  control (−0.018 nat), and the no-PLE arm is numerically the best of the three.
  Against the pre-registered threshold (≥2 points or ≥0.05 nat) **the LoRA row of
  G1 fails**.
* Consequence: this removes the last "the reader/adaptation just is not good
  enough" explanation. See `docs/round-153-...` section 2 for the consolidated
  falsification list.

### 1.3 Secondary protocol (chat template) — caveat

The chat re-evaluation reuses the **raw-trained** readers and adapters
(`--load-lora-adapter`) rather than retraining per protocol, because adapters
are cheap to keep but a second full training row costs hours.  Consequence: the
chat numbers are **protocol-mismatched** (trained on the raw prompt, evaluated
with the chat template), so only the `real` vs `control` **delta** is
interpretable; absolute chat values are not comparable to R148-A's chat row,
which did train per protocol.

`lora-nople` (chat, mismatched): BoolQ 0.802 / TriviaQA 0.188 / NQ 0.088 /
mean EM 0.3593; gold NLL overall 5.0249 (vs 2.4025 raw).  The gap between raw
and chat gold NLL (2.40 vs 5.02) quantifies the mismatch cost, and is a useful
caution for the next session: **always train and evaluate with the same prompt
protocol**.

## 2. G0: frozen 4B graft (scale/space test)

### 2.1 Design

```text
backbone   Qwen3.5-4B, hidden 2560 == 16 PLE heads x 160 dim (exact source space)
dtype      bfloat16: the released checkpoint is stored bf16 (9.3GB / 4.2B params),
           so this is lossless w.r.t. the published weights and 2x cheaper than
           the fp32 conversion the 0.8B path uses.  fp32 OOM'd next to concurrent
           work on the 24GB card.
arms       g0-real, g0-control: each trains its own frozen reader (500 steps,
           mixed50) on the 4B backbone, table frozen either way
eval       standard 1500 raw prompt: generation + gold NLL
reference  frozen 0.8B no-reader (round148a) for the same prompt protocol
```

The frozen no-reader arm is not rerun: it does not depend on backbone width and
is already measured (overall gold NLL 8.7429).

### 2.2 Results

<!-- RESULTS:G0 -->

## 3. What this changes

<!-- RESULTS:CONCLUSION -->

## 4. Engineering notes for the next session

* The remote venv now has `peft 0.20.0` installed; LoRA runs need no shim.
* `--lora` writes adapter-only checkpoints (`<save-reader>.lora/adapter.pt`,
  `get_peft_model_state_dict`), so a LoRA row never risks the 13GB disk budget
  with full backbone copies.
* `--qa-max-items N` truncates the eval set for smoke tests; never use it for
  reported numbers.
* The AutoDL instance regenerates its SSH host key on restart; use
  `scripts/ssh_autodl.sh` instead of raw `ssh` + `BatchMode`.
* The 4B backbone needs its own reader checkpoint (the official source reader
  projects into 2560, so 0.8B readers are not reusable).
* 4B evaluation must not share the GPU with another training job: fp32 needed
  16.7GB and OOM'd with 5GB of concurrent work.
