#!/usr/bin/env python
"""Round 169 Stage B1, step 2: train the rows, everything else frozen.

The pre-registration is ``docs/round-169-stageB1-row-retraining-preregistration.md``.
Only ``E`` (the row bank, indexed by trigram) receives gradient; the 0.8B backbone
and the official reader are frozen and in eval mode.

Two implementation decisions worth stating, because both are load-bearing:

1. **Row ids are computed once over the whole stream**, never over a gathered
   subsequence.  A row id is a function of ``tokens[t-2..t]``, so gathering
   destroys the very context that defines it.  The snapshot script's bound check
   caught exactly this mistake on its first run (19,996/20,000 spurious
   "disagreements"), which is why the check exists.

2. **The batch's rows are the only leaf that requires grad.**  Making ``E``
   itself a leaf would allocate a 6.8 GB ``.grad`` buffer and zero it every step.
   Instead the batch's rows are detached-and-regathered as a small leaf, and the
   gradient is scattered back into a sparse Adam whose state lives in CPU RAM
   (``E``, ``m``, ``v`` = 3 x 6.8 GB, against 1 TB of host memory).

The reader contains a causal convolution over time, so batches are contiguous
windows of the stream rather than shuffled positions: the convolution must see
real neighbours.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from round169_row_snapshot import trigram_codes


def log(msg: str) -> None:
    print(f"[r169-train] {msg}", flush=True)


def window_row_indices(pos: np.ndarray, s: int, batch_tokens: int) -> np.ndarray:
    """Snapshot row indices for the ``batch_tokens + 1`` stream positions from ``s``.

    ``pos[t - 2]`` is the snapshot row for stream position ``t``, because the
    injection at ``t`` is addressed by the trigram ENDING at ``t`` and
    ``codes[i]`` describes the trigram ending at token ``i + 2``.  The window has
    one more position than it scores, because the hook requires ``e_t`` to match
    the hidden-state length.
    """
    return pos[s - 2 : s + batch_tokens - 1]


class SparseAdam:
    """Adam over a row bank, updating only the rows a step actually touched.

    A dense ``torch.optim.AdamW`` over 667,404 x 2560 fp32 would touch 6.8 GB of
    parameters and 13.7 GB of state *every step*; with ~500 steps per epoch that
    dominates everything else.  Only a few thousand distinct rows appear in any
    batch, so the update is done on those rows alone.
    """

    def __init__(self, E, lr: float, betas=(0.9, 0.999), eps: float = 1e-8) -> None:
        import torch

        self.E = E
        self.lr = float(lr)
        self.b1, self.b2 = betas
        self.eps = eps
        self.m = torch.zeros_like(E)
        self.v = torch.zeros_like(E)
        self.t = 0

    def step(self, idx, grad) -> float:
        """``idx`` [N] int64 (CPU), ``grad`` [N, D] float32 (CPU).  Returns grad rms."""
        import torch

        self.t += 1
        uniq, inv = torch.unique(idx, return_inverse=True)
        g = torch.zeros((uniq.numel(), grad.shape[1]), dtype=torch.float32)
        g.index_add_(0, inv, grad)
        # Adam's moments assume the *mean* gradient; index_add_ summed, so divide
        # by how many times each row appeared.
        counts = torch.zeros(uniq.numel(), dtype=torch.float32)
        counts.index_add_(0, inv, torch.ones_like(inv, dtype=torch.float32))
        g /= counts.unsqueeze(1)

        m = self.m[uniq]
        v = self.v[uniq]
        m.mul_(self.b1).add_(g, alpha=1.0 - self.b1)
        v.mul_(self.b2).addcmul_(g, g, value=1.0 - self.b2)
        bc1 = 1.0 - self.b1**self.t
        bc2 = 1.0 - self.b2**self.t
        denom = (v / bc2).sqrt_().add_(self.eps)
        self.E[uniq] = self.E[uniq] - self.lr * (m / bc1) / denom
        self.m[uniq] = m
        self.v[uniq] = v
        return float(g.pow(2).mean().sqrt())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--snapshot-dir", required=True, help="dir with E0.npy, trigram-codes.npy")
    ap.add_argument("--train-tokens", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--reader", required=True)
    ap.add_argument("--reader-kind", choices=("official", "saved"), default="official",
                    help="official = raw extracted checkpoint (production path); "
                         "saved = a registry-format checkpoint from save_reader")
    ap.add_argument("--out", required=True, help="output .npy for the trained E")
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--scale", type=float, default=0.00019931793212890625)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-tokens", type=int, default=512,
                    help="small on purpose: the backward through a frozen 0.8B "
                         "backbone is the memory limit (2048 OOMs a 24 GiB card), "
                         "and smaller batches give each row MORE Adam steps for the "
                         "same exposure")
    ap.add_argument("--arm", choices=("real", "shuf"), default="real",
                    help="shuf trains against permuted next-token targets (the null)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--backbone-dtype", default="float32")
    ap.add_argument("--short-conv", action="store_true",
                    help="wrap the reader output in the official short conv")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument(
        "--keep-index", default="",
        help="(.npy of int64 row indices) save ONLY these rows of the trained bank. "
             "The full bank is 6.83 GB fp32; the eval-relevant rows are ~1.2 GB. "
             "bf16 is NOT an option: expected row movement is 1e-7..2e-5 against a "
             "bf16 quantisation floor of ~3e-5 on a bank of rms 8e-3, so bf16 noise "
             "would exceed the signal in the trained arm only.",
    )
    ap.add_argument("--max-steps", type=int, default=0, help="smoke only; 0 = all")
    args = ap.parse_args()

    import torch

    from qwen35_ple.reader import install_reader_hook
    from qwen35_ple.real_ple import resolve_ple_weight_scale

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import run_phase0 as p0

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    snap = Path(args.snapshot_dir)
    E0 = torch.from_numpy(np.load(snap / "E0.npy"))          # [n_tri, 2560] fp32 CPU
    codes_uniq = np.load(snap / "trigram-codes.npy")
    n_tri, dim = E0.shape
    log(f"row bank: {n_tri:,} trigrams x {dim} (E0 {E0.numel() * 4 / 1e9:.2f} GB, CPU)")

    tokens = np.load(args.train_tokens).astype(np.int64).reshape(-1)
    codes = trigram_codes(tokens)
    if codes.size != tokens.size - 2:
        raise SystemExit("trigram code count does not match the stream length")
    # codes[i] describes the trigram ending at token i+2.  Position t of the
    # stream needs the trigram ending at t, i.e. code index t-2.
    pos = np.searchsorted(codes_uniq, codes)
    if pos.min() < 0 or pos.max() >= n_tri or not np.array_equal(codes_uniq[pos], codes):
        raise SystemExit("the stream contains a trigram absent from the snapshot")
    pos = pos.astype(np.int64)
    log(f"stream: {tokens.size:,} tokens, all trigrams resolved to snapshot rows")

    scale = resolve_ple_weight_scale(model_dir=args.model, scale=args.scale)
    log(f"PLE weight scale: {scale:.6g}")
    _tokenizer, model = p0._load_model(args.model, args.device, args.backbone_dtype)
    for prm in model.parameters():
        prm.requires_grad_(False)
    model.eval()

    if args.reader_kind == "official":
        # ``data/official_ple_reader.pt`` is the RAW extracted official
        # checkpoint (keys like ``key_proj.weight``), not a registry-format one,
        # so ``load_reader_with_extra`` rejects it: it requires a "format"
        # marker.  This is the production path used by run_phase0 with
        # ``--reader official``, and it does not use a short conv.
        from qwen35_ple.reader import OfficialSourceQwenReader

        reader = OfficialSourceQwenReader.from_official_checkpoint(
            args.reader, d_target=model.config.hidden_size, freeze_source=True,
        )
    else:
        reader, _extra = p0.load_reader_with_extra(Path(args.reader), device=args.device)
    reader = reader.to(args.device)
    for prm in reader.parameters():
        prm.requires_grad_(False)
    reader.eval()

    short_conv = None
    if args.short_conv:
        from qwen35_ple.reader import QwenShortConv

        short_conv = QwenShortConv(dim).to(args.device).eval()

    install_reader_hook(model, args.layer, reader, short_conv)

    opt = SparseAdam(E0.clone(), lr=args.lr)
    B = int(args.batch_tokens)
    # A window of B+1 tokens scores B predictions, at stream positions s..s+B-1.
    # The injection at stream position t is addressed by the trigram ENDING at t,
    # which is ``codes[t-2]`` -- so the window needs ``pos[s-2 : s+B-1]``, length
    # B+1, matching the hidden states the hook checks against.  ``s`` starts at 2
    # because position t-2 must exist.
    starts = list(range(2, tokens.size - B - 2, B))
    if args.max_steps:
        starts = starts[: args.max_steps]
    log(f"{len(starts)} steps/epoch x {args.epochs} epochs, batch_tokens={B}")

    history: list[dict] = []
    t0 = time.time()
    for epoch in range(args.epochs):
        run_loss, run_g, n = 0.0, 0.0, 0
        for si, s in enumerate(starts):
            e = s + B + 1
            win = torch.from_numpy(tokens[s:e]).unsqueeze(0).to(args.device)
            idx_np = window_row_indices(pos, s, B)
            if idx_np.size != win.shape[1]:
                raise SystemExit(
                    f"row-window mismatch: {idx_np.size} rows for {win.shape[1]} positions"
                )
            idx = torch.from_numpy(np.ascontiguousarray(idx_np))
            # Gather from the CURRENT bank, not from E0: gathering from the frozen
            # snapshot would make every step train against the initialisation and
            # the bank would never actually move.
            rows = opt.E[idx].detach().to(args.device).requires_grad_(True)
            model._current_ple_e_t = rows.unsqueeze(0)

            labels = win.clone()
            if args.arm == "shuf":
                perm = torch.from_numpy(rng.permutation(labels.shape[1] - 1)).to(args.device)
                labels[0, 1:] = win[0, 1:][perm]

            out = model(input_ids=win, labels=labels)
            loss = out.loss
            loss.backward()
            if rows.grad is None:
                raise SystemExit(
                    "no gradient reached the rows: the reader/injection path is detached, "
                    "so B1 cannot answer its own question (this is the GRADIENT_BLOCKED case)"
                )
            g_rms = opt.step(idx, rows.grad.detach().to("cpu"))
            run_loss += float(loss)
            run_g += g_rms
            n += 1
            del out, loss, rows
            if si % args.log_every == 0:
                log(
                    f"  ep{epoch} step {si}/{len(starts)} loss={run_loss / n:.4f} "
                    f"grad_rms={run_g / n:.3e} ({time.time() - t0:.0f}s)"
                )
        history.append({"epoch": epoch, "mean_loss": run_loss / max(n, 1),
                        "mean_grad_rms": run_g / max(n, 1), "steps": n})
        log(f"epoch {epoch} done: loss={run_loss / max(n, 1):.4f} "
            f"grad_rms={run_g / max(n, 1):.3e}")

    delta = (opt.E - E0)
    meta = {
        "arm": args.arm, "epochs": args.epochs, "lr": args.lr, "seed": args.seed,
        "layer": args.layer, "batch_tokens": B, "steps_per_epoch": len(starts),
        "scale": scale, "model": args.model, "reader": args.reader,
        "snapshot_dir": str(snap), "train_tokens": args.train_tokens,
        "history": history,
        "changed_rows": int((delta.abs().sum(dim=1) > 0).sum()),
        "delta_rms": float(delta.pow(2).mean().sqrt()),
        "delta_absmax": float(delta.abs().max()),
        "e0_rms": float(E0.pow(2).mean().sqrt()),
        "elapsed_seconds": time.time() - t0,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keep = None
    if args.keep_index:
        keep = np.load(args.keep_index).astype(np.int64)
        if keep.min() < 0 or keep.max() >= n_tri:
            raise SystemExit("--keep-index contains an out-of-range row index")
        bank = opt.E.numpy()[keep]
        meta["keep_index"] = str(args.keep_index)
        meta["keep_rows"] = int(keep.size)
        meta["kept_delta_rms"] = float(delta[torch.from_numpy(keep)].pow(2).mean().sqrt())
    else:
        bank = opt.E.numpy()
    np.save(out_path, bank)
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    log(f"wrote {out_path} {bank.shape} ({bank.nbytes / 1e9:.2f} GB); "
        f"changed rows {meta['changed_rows']:,}/{n_tri:,}, "
        f"delta_rms {meta['delta_rms']:.3e} vs E0 rms {meta['e0_rms']:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
