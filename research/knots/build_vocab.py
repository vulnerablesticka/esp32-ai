"""Assemble artifacts/knots/vocab.json and layout.json from the curated word
list and the trained input tokenizer.

Barista's contract (validated by
firmware/esp32_barista/tools/generate_vocab_headers.py:load(), reused
verbatim here by pointing --vocab/--layout at these outputs):

  vocab.json  = {"total": N, "tokens": [{"token": str}, ...]}
  layout.json = {"bpe_vocab": int, "total": int, "n_words": int,
                 "out2in": [input_id per class]}

For each output-vocabulary word (output_words.json, specials first), this
reuses an existing input token id when the word, encoded standalone (no
leading space, no surrounding context -- matching exactly the check
firmware/esp32_knots/tools/verify_tokenizer.py runs: `tokenizer.encode(word)
== [out2in[k]]`), tokenizes to exactly one id. Everything else, including all
4 specials (they aren't natural-language text the tokenizer would ever see
standalone), gets a new appended row, in class order, starting right after
the tokenizer's own vocabulary.

  uv run python research/knots/build_vocab.py --vocab 2048
"""

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA = ROOT / "data" / "knots"
OUT = ROOT / "artifacts" / "knots"


def reused_id(tok: Tokenizer, word: str) -> int | None:
    """The single existing input id for `word`, or None if it needs >1 piece.

    Standalone encoding (no leading space), matching verify_tokenizer.py's
    check exactly -- that script is the actual gate, so this must agree with
    it rather than with any other plausible definition of "reusable".
    """
    if word.startswith("<"):
        return None  # specials are markers, not text the tokenizer ever sees
    enc = tok.encode(word, add_special_tokens=False)
    if len(enc.ids) != 1:
        return None
    if tok.decode(enc.ids) != word:
        return None  # a byte-level artifact would corrupt the word on decode
    return enc.ids[0]


def main() -> None:
    """Parse --vocab, then write artifacts/knots/vocab.json and layout.json."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=2048, help="the --vocab value prepare.py was run with")
    args = ap.parse_args()

    tok_path = DATA / f"bpe{args.vocab}.json"
    if not tok_path.is_file():
        raise SystemExit(
            f"tokenizer not found: {tok_path}\nrun: uv run python research/knots/prepare.py --vocab {args.vocab}"
        )
    tok = Tokenizer.from_file(str(tok_path))
    bpe_vocab = tok.get_vocab_size()

    words = json.loads((HERE / "output_words.json").read_text())["words"]

    out2in = []
    next_new = bpe_vocab
    reused, appended = 0, 0
    for word in words:
        rid = reused_id(tok, word)
        if rid is not None:
            out2in.append(rid)
            reused += 1
        else:
            out2in.append(next_new)
            next_new += 1
            appended += 1

    total = bpe_vocab + appended
    n_words = len(words)

    OUT.mkdir(parents=True, exist_ok=True)
    vocab_json = {
        "total": n_words,
        "tokens": [{"token": w} for w in words],
    }
    layout_json = {
        "bpe_vocab": bpe_vocab,
        "total": total,
        "n_words": n_words,
        "out2in": out2in,
    }
    (OUT / "vocab.json").write_text(json.dumps(vocab_json, indent=2) + "\n")
    (OUT / "layout.json").write_text(json.dumps(layout_json, indent=2) + "\n")

    print(f"wrote {OUT}/vocab.json and layout.json")
    print(
        f"  bpe_vocab={bpe_vocab}  n_words={n_words}  reused={reused}  appended={appended}  total_input_vocab={total}"
    )


if __name__ == "__main__":
    main()
