"""Encode qa_pairs.jsonl into fixed-length training rows for TinyLM.

Replicates exactly the sequence firmware/esp32_barista/esp32_barista.ino's
answer() loop builds at inference (lines ~216-242): question tokens, then
OUT2IN[BOS], then OUT2IN[class] for each sampled answer word in turn. Given a
K-word answer, the fed `idx` sequence is:

  q_ids + [out2in[BOS]] + [out2in[c] for c in answer_classes]

and the `targets` sequence (TinyLM.forward's second argument, ignore_index=-1
per src/model.py:257-260) is the same length, each position's target being
the class that position's forward pass should predict:

  [-1]*len(q_ids) + answer_classes + [EOS]

i.e. targets[i] is "the class produced by feeding idx[i]" -- targets is
idx's answer-side content shifted so the BOS-fed position predicts the first
answer word, each answer-word-fed position predicts the next one, and the
last answer word's position predicts EOS. Question-span positions carry no
output-class meaning (the model reads BPE there, it does not write it) and
are ignored.

Hard-fails (not silently drops) on any answer word missing from vocab.json:
qa_pairs.jsonl was generated against the same output_words.json this reads,
so a mismatch here means the two have drifted, not that data is merely dirty.

  uv run python research/knots/build_dataset.py --vocab 2048 --seq-len 128
"""

import argparse
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA = ROOT / "data" / "knots"
ARTIFACTS = ROOT / "artifacts" / "knots"

VAL_FRACTION = 0.15
SPLIT_SEED = 0


def encode_examples(
    tok: Tokenizer, words: list[str], out2in: list[int], seq_len: int
) -> tuple[NDArray[np.int16], NDArray[np.int16]]:
    """Encode every qa_pairs.jsonl example into one padded (idx, targets) row.

    Returns two `int16` arrays, each shape `(n_examples, seq_len)`.
    """
    word_to_class = {w: i for i, w in enumerate(words)}
    bos_class, eos_class, pad_class = 1, 2, 0

    rows_idx, rows_target = [], []
    qa_path = HERE / "qa_pairs.jsonl"
    for line in qa_path.read_text().splitlines():
        ex = json.loads(line)
        q_ids = tok.encode(ex["question"]).ids

        missing = [w for w in ex["answer_words"] if w not in word_to_class]
        if missing:
            raise SystemExit(
                f"answer word(s) {missing} not in vocab.json for question "
                f"{ex['question']!r}; output_words.json and vocab.json have "
                f"drifted from qa_pairs.jsonl -- rerun build_vocab.py or "
                f"regenerate the corpus"
            )
        answer_classes = [word_to_class[w] for w in ex["answer_words"]]

        idx = q_ids + [out2in[bos_class]] + [out2in[c] for c in answer_classes]
        targets = [-1] * len(q_ids) + answer_classes + [eos_class]
        if len(idx) > seq_len:
            raise SystemExit(f"example needs {len(idx)} positions, exceeds --seq-len {seq_len}: {ex['question']!r}")
        pad = seq_len - len(idx)
        idx = idx + [out2in[pad_class]] * pad
        targets = targets + [-1] * pad
        rows_idx.append(idx)
        rows_target.append(targets)

    return np.array(rows_idx, dtype=np.int16), np.array(rows_target, dtype=np.int16)


def main() -> None:
    """Parse --vocab/--seq-len, then write the train/val memmaps and manifest.json."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--vocab", type=int, default=2048, help="the --vocab value prepare.py/build_vocab.py were run with"
    )
    # 128 comfortably covers this corpus's longest example (a 12-knot category
    # listing, ~40 answer pieces) plus a full question and KNOTS_ANSWER_ROOM's
    # reserved headroom -- see firmware/esp32_knots/esp32_knots.ino. A smaller
    # value here previously caused live "(question too long)" rejections on
    # perfectly normal questions once paired with that fixed headroom.
    ap.add_argument("--seq-len", type=int, default=128)
    args = ap.parse_args()

    tok_path = DATA / f"bpe{args.vocab}.json"
    if not tok_path.is_file():
        raise SystemExit(
            f"tokenizer not found: {tok_path}\nrun: uv run python research/knots/prepare.py --vocab {args.vocab}"
        )
    tok = Tokenizer.from_file(str(tok_path))

    vocab = json.loads((ARTIFACTS / "vocab.json").read_text())
    layout = json.loads((ARTIFACTS / "layout.json").read_text())
    words = [t["token"] for t in vocab["tokens"]]
    out2in = layout["out2in"]

    idx, targets = encode_examples(tok, words, out2in, args.seq_len)

    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(len(idx))
    n_val = max(1, int(len(idx) * VAL_FRACTION))
    val_ix, train_ix = perm[:n_val], perm[n_val:]

    DATA.mkdir(parents=True, exist_ok=True)
    for name, ix in (("train", train_ix), ("val", val_ix)):
        idx[ix].tofile(DATA / f"{name}_input.bin")
        targets[ix].tofile(DATA / f"{name}_target.bin")

    manifest = {
        "seq_len": args.seq_len,
        "vocab_size": layout["total"],
        "out_vocab_size": len(words),
        "n_train": len(train_ix),
        "n_val": len(val_ix),
        "tokenizer": str(tok_path.relative_to(ROOT)),
    }
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"wrote {DATA}/{{train,val}}_{{input,target}}.bin  "
        f"({len(train_ix)} train / {len(val_ix)} val rows, seq_len={args.seq_len})"
    )
    print(
        "NOTE: this pilot corpus is too small and low-diversity for a "
        "held-out val split to mean much -- it validates the pipeline "
        "mechanics, not generalization. Re-split by held-out fact once "
        "the corpus covers more of knots.json."
    )


if __name__ == "__main__":
    main()
