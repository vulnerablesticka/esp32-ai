"""Check that the device encoder agrees with Hugging Face, and that the word
tables agree with the tokenizer.

Cloned from firmware/esp32_barista/tools/verify_tokenizer.py -- identical
checks, only the default paths and EXPECTED_REUSED (knots' own reuse count,
not barista's) differ.

Two things can silently disagree once this sketch is assembled, and neither
shows up as a crash:

  the encoder    a question typed on device must produce the same token ids the
                 Python tokenizer produced during training;
  the word map   the EXPECTED_REUSED output classes that reuse an existing BPE
                 id must reuse the id that actually encodes their own word.
                 The generator cannot check this, because it never sees the
                 tokenizer.

Probes are fixed and public. They are written here rather than read from the
corpus, so this runs from a clone with no private data, and so a corpus edit
cannot quietly change what was checked.

  uv run python firmware/esp32_knots/tools/verify_tokenizer.py
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from tokenizers import Tokenizer

TOOLS = Path(__file__).resolve().parent
SKETCH = TOOLS.parent
ROOT = SKETCH.parent.parent
DEFAULT_TOKENIZER = ROOT / "artifacts" / "knots" / "tokenizer.json"
DEFAULT_LAYOUT = ROOT / "artifacts" / "knots" / "layout.json"
DEFAULT_VOCAB = ROOT / "artifacts" / "knots" / "vocab.json"
CONFORMANCE_C = ROOT / "runtime" / "host_verify" / "tokenizer_conformance.c"

# The frozen knots release reuses exactly this many existing BPE ids. A change
# means the vocabulary moved, which invalidates the trained head, so it is a
# release-identity check rather than a soft expectation.
EXPECTED_REUSED = 63

# Fixed public probes. Whitespace, punctuation, contractions, casing and digits
# are where ByteLevel pre-tokenization is easiest to get wrong, so they are
# over-represented deliberately.
PROBES = [
    "",
    "hello world",
    "  hello",
    "hello  world",
    "hello   ",
    "a",
    "18 g, sour.",
    "I'm fine",
    "DON'T",
    "we'll see",
    "they're done",
    "i've been practicing",
    "a\tb",
    "line\nbreak",
    "what?! yes",
    "how do i tie a bowline",
    "which knot should i use for a tarp",
    "what grind size should i use",
    "why does my hitch keep slipping?",
    "i wrapped it three times and it still jams",
    "rope, cord; ratio: 1:2",
    "SHOUTING IN CAPS",
    "mixed CaSe WoRdS",
    "numbers 123 456",
    "hyphen-ated words",
    "under_score",
    "slash/es",
    "(parentheses) [brackets]",
    "@#$%^&*",
    "punctuation!!! ???",
    "  leading and trailing  ",
    "a very long question about which knot to use for a rappel anchor",
]
# Every printable ASCII character on its own, which checks 95 of the 256
# byte-table entries individually. This is not full byte-table coverage.
PROBES += [chr(code) for code in range(0x20, 0x7F)]

NON_ASCII_PROBE = "café"
BTK_ERR_NOT_ASCII = -3


def build_verifier(workdir: Path) -> Path:
    """Compile runtime/host_verify/tokenizer_conformance.c, warnings as errors."""
    binary = workdir / "tokenizer_conformance"
    # The gate reports a clean build, so it must also require one.
    subprocess.run(
        ["cc", "-O2", "-std=c11", "-Wall", "-Wextra", "-Werror", "-o", str(binary), str(CONFORMANCE_C)], check=True
    )
    return binary


def build_asset(tokenizer_path: str | Path, workdir: Path) -> Path:
    """Pack the tokenizer with the same generator the firmware header uses."""
    sys.path.insert(0, str(TOOLS))
    import generate_tokenizer_header as packer

    asset = workdir / "tokenizer.btk"
    packer.generate(tokenizer_path, workdir / "tokenizer_encoder.h", asset)
    return asset


def encode_all(verifier: Path, asset: Path, strings: list[str]) -> list[str]:
    """Run the C verifier over `strings`, one output line (space-separated ids) per input."""
    result = subprocess.run([str(verifier), str(asset), *strings], check=True, capture_output=True, text=True)
    lines = result.stdout.split("\n")[:-1]  # one line per string, trailing \n
    if len(lines) != len(strings):
        raise SystemExit(f"verifier returned {len(lines)} lines for {len(strings)} strings")
    return lines


def parse_ids(line: str) -> list[int]:
    """Parse one encode_all() output line into its list of token ids."""
    return [] if not line else [int(value) for value in line.split()]


def main() -> None:
    """Parse args, then run the encoder-conformance and word-map checks."""
    ap = argparse.ArgumentParser(
        description="Verify the device encoder against Hugging Face, and the "
        "reused word mappings against the tokenizer."
    )
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--layout", default=DEFAULT_LAYOUT)
    ap.add_argument("--vocab", default=DEFAULT_VOCAB)
    ap.add_argument("--verifier", default=None, help="prebuilt tokenizer_conformance binary; built if absent")
    ap.add_argument("--asset", default=None, help="prebuilt raw BTK1 asset; packed if absent")
    args = ap.parse_args()

    for label, path in (("tokenizer", args.tokenizer), ("layout", args.layout), ("vocab", args.vocab)):
        if not Path(path).is_file():
            raise SystemExit(
                f"{label} not found: {path}\nknots' canonical assets belong in {DEFAULT_TOKENIZER.parent}."
            )

    layout = json.loads(Path(args.layout).read_text())
    words = [t["token"] for t in json.loads(Path(args.vocab).read_text())["tokens"]]
    out2in, bpe_vocab = layout["out2in"], layout["bpe_vocab"]
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    failures = []

    # 1. the tokenizer the word tables were built against is this one
    if tokenizer.get_vocab_size() != bpe_vocab:
        failures.append(f"layout bpe_vocab {bpe_vocab} but tokenizer has {tokenizer.get_vocab_size()} tokens")

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        verifier = Path(args.verifier) if args.verifier else build_verifier(workdir)
        asset = Path(args.asset) if args.asset else build_asset(args.tokenizer, workdir)

        # 2. C ids equal Hugging Face ids on every probe
        mismatches = []
        observed = set()
        for text, line in zip(PROBES, encode_all(verifier, asset, PROBES)):
            got = parse_ids(line)
            want = tokenizer.encode(text, add_special_tokens=False).ids
            observed.update(want)
            if got != want:
                mismatches.append({"text": text, "expected": want, "actual": got})
        if mismatches:
            failures.append(f"{len(mismatches)} encoder mismatches")

        # 3. non-ASCII is refused rather than encoded differently
        rejected = encode_all(verifier, asset, [NON_ASCII_PROBE])[0]
        if rejected != f"ERROR:{BTK_ERR_NOT_ASCII}":
            failures.append(f"non-ASCII probe returned {rejected!r}")

        # 4. every reused class encodes its own word to exactly its own id.
        # Two separate claims, so they get two separate checks: the mapping in
        # layout.json must be what the tokenizer actually produces, and the
        # device must reproduce that. Comparing only the device against
        # layout.json would pass a layout that disagrees with the tokenizer, as
        # long as the device disagreed the same way.
        reused = [k for k, token_id in enumerate(out2in) if token_id < bpe_vocab]
        reused_lines = encode_all(verifier, asset, [words[k] for k in reused])
        wrong_reference, wrong_device = [], []
        for k, line in zip(reused, reused_lines):
            want = tokenizer.encode(words[k], add_special_tokens=False).ids
            got = parse_ids(line)
            observed.update(want)
            if want != [out2in[k]]:
                wrong_reference.append({"class": k, "word": words[k], "layout": [out2in[k]], "tokenizer": want})
            if got != want:
                wrong_device.append({"class": k, "word": words[k], "tokenizer": want, "device": got})
        if wrong_reference:
            failures.append(f"{len(wrong_reference)} reused words do not encode to their mapped id in the tokenizer")
        if wrong_device:
            failures.append(f"{len(wrong_device)} reused words encode differently on device")

        # 5. the count itself identifies the release
        if len(reused) != EXPECTED_REUSED:
            failures.append(f"{len(reused)} reused mappings, expected {EXPECTED_REUSED}")

    # Say what was actually looked at, not just that it passed. The count below
    # is distinct reference ids seen in the expected encodings: it is not a
    # coverage figure, since it measures neither vocabulary entries nor
    # byte-table entries nor merge rules verified. Fixed probes are a spot
    # check; only the reused-mapping check is exhaustive over its EXPECTED_REUSED.
    print(
        json.dumps(
            {
                "supported_input": "ASCII",
                "probes": len(PROBES),
                "encoder_mismatches": len(mismatches),
                "non_ascii_rejected": rejected == f"ERROR:{BTK_ERR_NOT_ASCII}",
                "reused_mappings": len(reused),
                "reused_reference_mismatches": len(wrong_reference),
                "reused_device_mismatches": len(wrong_device),
                "tokenizer_vocab": tokenizer.get_vocab_size(),
                "distinct_reference_ids_observed": len(observed),
                "examples": (mismatches + wrong_reference + wrong_device)[:3],
            },
            indent=2,
        )
    )

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        raise SystemExit(1)
    print("PASS")


if __name__ == "__main__":
    main()
