"""Train the knots model: barista's architecture (untied head, PLE arm), a
new Batcher for the row-per-example dataset build_dataset.py writes.

Structurally mirrors research/tinystories/train.py's LR schedule, AdamW setup,
eval loop and checkpoint format (lines ~61-198), but:
  - no arm/core-matching solver (make_model): that machinery exists to keep
    ablation arms comparable at a fixed core budget, which does not apply to
    training one production config directly.
  - Batcher samples whole rows (each one already a complete, padded
    input/target sequence from build_dataset.py) instead of tinystories'
    sliding-window offsets into an open token stream.

  uv run python research/knots/train.py --steps 3000 --batch-size 32
"""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from model import Config, TinyLM

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "knots"
RUNS = ROOT / "runs"


def get_device() -> str:
    """The best available torch device: mps, then cuda, then cpu."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Batcher:
    """Samples whole rows: each row is already one complete padded example,
    unlike tinystories' Batcher which slices windows out of an open stream."""

    def __init__(self, split: str, manifest: dict, batch_size: int, device: str, seed: int = 0) -> None:
        seq_len = manifest["seq_len"]
        self.idx = np.memmap(DATA / f"{split}_input.bin", dtype=np.int16, mode="r").reshape(-1, seq_len)
        self.target = np.memmap(DATA / f"{split}_target.bin", dtype=np.int16, mode="r").reshape(-1, seq_len)
        self.bs, self.device = batch_size, device
        self.rng = np.random.default_rng(1234 if split == "val" else seed)

    def __call__(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample one random batch, returning (idx, targets) as int64 tensors on `device`."""
        ix = self.rng.integers(0, self.idx.shape[0], self.bs)
        x = self.idx[ix].astype(np.int64)
        y = self.target[ix].astype(np.int64)
        return torch.from_numpy(x).to(self.device), torch.from_numpy(y).to(self.device)


def lr_at(step: int, total: int, peak: float, warmup: int) -> float:
    """Linear warmup to `peak`, then cosine decay to 10% of `peak`."""
    if step < warmup:
        return peak * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return 0.1 * peak + 0.9 * peak * 0.5 * (1 + math.cos(math.pi * p))


@torch.no_grad()
def evaluate(model: TinyLM, batcher: Batcher, iters: int) -> float:
    """Mean validation loss over `iters` batches, on a fixed batch stream."""
    model.eval()
    batcher.rng = np.random.default_rng(1234)
    losses = [model(*batcher())[1].item() for _ in range(iters)]
    model.train()
    return sum(losses) / len(losses)


def main() -> None:
    """Parse training args, then train and checkpoint the knots model."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-iters", type=int, default=10)
    ap.add_argument("--ple-dim", type=int, default=128)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--ffn-hidden", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="pilot")
    args = ap.parse_args()

    manifest_path = DATA / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"{manifest_path} missing; run build_dataset.py first")
    manifest = json.loads(manifest_path.read_text())

    tok_path = ROOT / manifest["tokenizer"]
    tok_sha = hashlib.sha256(tok_path.read_bytes()).hexdigest()

    torch.manual_seed(args.seed)
    device = get_device()
    RUNS.mkdir(exist_ok=True)

    cfg = Config(
        arm="ple",
        vocab_size=manifest["vocab_size"],
        out_vocab_size=manifest["out_vocab_size"],
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ffn_hidden=args.ffn_hidden,
        ple_dim=args.ple_dim,
        seq_len=manifest["seq_len"],
    )
    model = TinyLM(cfg).to(device)
    budget = model.param_budget()
    print(
        f"knots pilot: core={budget['core']:,} table={budget['table']:,} "
        f"stream={budget['stream']:,} total={budget['total']:,}"
    )

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.ndim < 2 or "table" in n or "tok_emb" in n else decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr,
        betas=(0.9, 0.95),
    )

    train_b = Batcher("train", manifest, args.batch_size, device, seed=args.seed)
    val_b = Batcher("val", manifest, args.batch_size, device)

    name = f"knots-{args.tag}-s{args.seed}"
    history, best = [], float("inf")
    t0 = time.time()

    for step in range(args.steps):
        lr = lr_at(step, args.steps, args.lr, args.warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = train_b()
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.eval_every == 0 or step == args.steps - 1:
            vl = evaluate(model, val_b, args.eval_iters)
            best = min(best, vl)
            history.append({"step": step, "train": loss.item(), "val": vl})
            print(
                f"{name} step {step:5d} | train {loss.item():.4f} | "
                f"val {vl:.4f} | ppl {math.exp(vl):7.2f} | "
                f"{time.time() - t0:5.0f}s",
                flush=True,
            )

    result = {
        "name": name,
        "seed": args.seed,
        "tag": args.tag,
        "config": dict(cfg.__dict__),
        "manifest": manifest,
        "tokenizer_sha256": tok_sha,
        "params": budget,
        "final_val": history[-1]["val"],
        "best_val": best,
        "steps": args.steps,
        "wall_seconds": time.time() - t0,
        "history": history,
    }
    (RUNS / f"{name}.json").write_text(json.dumps(result, indent=2))
    torch.save(
        {
            "cfg": cfg.__dict__,
            "state": model.state_dict(),
            "tokenizer_sha256": tok_sha,
            "seed": args.seed,
            "tag": args.tag,
            "name": name,
        },
        RUNS / f"{name}.pt",
    )
    print(f"{name} DONE  best_val={best:.4f}  -> runs/{name}.pt")


if __name__ == "__main__":
    main()
