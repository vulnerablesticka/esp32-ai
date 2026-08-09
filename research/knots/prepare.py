"""Train the knots model's input BPE tokenizer.

Unlike research/tinystories/prepare.py, this does not also write train.bin/
val.bin: the knots corpus is a small set of individual Q&A examples, not an
open text stream, so encoding into fixed-length training rows (padded,
target-masked over the question span) is build_dataset.py's job, run after
this tokenizer exists. This script only trains and saves the tokenizer.

The training text is qa_pairs.jsonl's questions and answers, plus several
repetitions of output_words.json's word list on their own. The repetitions
bias BPE's merge learning toward forming a single token per curated output
word where possible: those are the words check_vocabulary's out2in can reuse
an existing input id for rather than append a new embedding row for
(mirroring barista's ~135-of-854-reused split, documented in
firmware/esp32_barista/tools/generate_vocab_headers.py).

  uv run python research/knots/prepare.py --vocab 2048
"""

import argparse
import json
import os
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

ROOT = Path(__file__).resolve().parents[2]
DATA = str(ROOT / "data" / "knots")
HERE = Path(__file__).resolve().parent

# Small default: this pilot corpus (hundreds of examples, ~280 output words)
# has far less text than tinystories' 300MB slice, so a barista-scale
# ~8000-token vocabulary would mostly be padding merges with no signal
# behind them. Grow this once the corpus grows -- it should scale with
# corpus size, not be fixed to match barista's number.
VOCAB_SIZE = 2048

# Bias weight: how many extra times the output word list is folded into the
# training text, on top of appearing naturally in the answers.
WORD_LIST_REPEATS = 8


def load_corpus_docs() -> list[str]:
    """Build the tokenizer's training text as a list of independent documents.

    barista's on-device encoder contract (checked by
    firmware/esp32_barista/tools/generate_tokenizer_header.py's
    check_encoding_contract) rejects a tokenizer with any "added tokens" --
    which is what a special_tokens=["<|endoftext|>"] BpeTrainer boundary
    marker becomes once saved. train_from_iterator treats each list element
    as its own sequence, so passing separate documents keeps merges from
    crossing document boundaries without needing that marker at all.
    """
    qa_path = HERE / "qa_pairs.jsonl"
    docs = []
    for line in qa_path.read_text().splitlines():
        ex = json.loads(line)
        answer_text = " ".join(ex["answer_words"])
        docs.append(ex["question"])
        docs.append(answer_text)

    words = json.loads((HERE / "output_words.json").read_text())["words"]
    word_line = " ".join(w for w in words if not w.startswith("<"))
    docs.extend([word_line] * WORD_LIST_REPEATS)

    return docs


def train_tokenizer(docs: list[str], vocab_size: int) -> Tokenizer:
    """Train (or load, if already cached) a byte-level BPE tokenizer over `docs`."""
    path = os.path.join(DATA, f"bpe{vocab_size}.json")
    if os.path.exists(path):
        print(f"already have {path}")
        return Tokenizer.from_file(path)
    print(f"training BPE vocab={vocab_size}...")
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train_from_iterator(docs, trainer=trainer)
    tok.save(path)
    return tok


def main() -> None:
    """Parse --vocab, then train and save the knots input tokenizer."""
    os.makedirs(DATA, exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=VOCAB_SIZE)
    args = ap.parse_args()
    if not 0 < args.vocab <= 65536:
        raise SystemExit("--vocab must be 1..65536; downstream bins are uint16")

    docs = load_corpus_docs()
    tok = train_tokenizer(docs, args.vocab)
    print(f"trained tokenizer: {tok.get_vocab_size()} ids, {sum(len(d) for d in docs):,} bytes of training text")


if __name__ == "__main__":
    main()
