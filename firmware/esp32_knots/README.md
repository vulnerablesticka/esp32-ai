# ESP32-S3 Knots (pilot)

Knot-tying question answering, USB serial in and out. Same architecture as
Barista (`firmware/esp32_barista/`): it reads a wide ASCII vocabulary so it
can take varied questions, but writes only from a small curated output
alphabet, so answers stay on topic and words outside that alphabet are
literally unsayable.

**This is a pilot, not a release.** The model is trained on a small,
hand-generated corpus (`research/knots/qa_pairs.jsonl`): a how-to-tie
procedure for all 72 knots in `research/knots/knots.json`, 16 which-knot
scenario recommendations, a capability fact ("what type of knots do you
know?" -> the taxonomy's category list), and a per-category listing fact
("tell me about {category} knots" -> every knot name in that category), each
with 6-7 paraphrased questions (577 examples total). It validates the full
pipeline end to end -- tokenizer, vocabulary, training, export, firmware --
but the corpus is still one canonical answer per fact reworded several ways,
not independently-written variety, so the model has only seen enough to
memorize this set well, not to generalize to phrasing it has never seen.
There is no published Hugging Face release yet, so
`scripts/fetch_model.sh knots` does not apply: `artifacts/knots/` is filled
by running the `research/knots/` pipeline locally (see that directory's
README) rather than downloaded.

Content caution: several covered categories (climbing, rescue, boating
anchor bends) are safety-relevant in real use. This pilot's answers are
built from simplified, vocabulary-constrained procedures and have not had a
domain-expert accuracy pass -- treat them as a demo of the architecture, not
a safety reference.

Two board targets are supported, same as barista:

|            | devkit (default)                    | xiao_s3                          |
| ---------- | ------------------------------------ | --------------------------------- |
| Board      | generic ESP32-S3, 16MB flash, 8MB PSRAM | Seeed XIAO ESP32S3 (plain, non-Sense), 8MB flash, 8MB PSRAM |
| Display    | optional I2C OLED (`display.h`)      | none — serial only (`USE_DISPLAY=0`) |
| Model      | knots (pilot, ~1.29MB)               | knots (pilot, ~1.29MB) — comfortably under the 6MB model partition |

Toolchain setup is identical to barista's -- see
[`firmware/esp32_barista/README.md`](../esp32_barista/README.md) for the
macOS/arduino-cli install steps, which are shared across all three models in
this repo.

## Build and flash

```bash
# fill artifacts/knots/ locally first (no published release to fetch yet):
uv run python research/knots/generate_corpus.py
uv run python research/knots/prepare.py --vocab 2048
uv run python research/knots/build_vocab.py --vocab 2048
uv run python research/knots/build_dataset.py --vocab 2048 --seq-len 128
uv run python research/knots/train.py --steps 3000 --batch-size 32
uv run python firmware/esp32_knots/tools/export.py --checkpoint runs/knots-pilot-s0.pt

uv run scripts/deploy.sh knots                    # devkit (default), with display
uv run scripts/deploy.sh knots xiao_s3            # XIAO ESP32S3, serial only
```

`deploy.sh` regenerates the word tables and encoder asset, runs the host
gates, compiles, writes the model partition and then the firmware, and
prints the fingerprint the board should report back -- identical procedure
to barista's, see that model's README for the details of what each step
checks.

## Running a step by hand

```bash
cc -O3 -Wall -Wextra -DLLM_INT8_ACT=1 -o /tmp/staging runtime/host_verify/staging_verify.c -lm
/tmp/staging artifacts/knots/model.bin

uv run python firmware/esp32_knots/tools/verify_tokenizer.py \
  --tokenizer artifacts/knots/tokenizer.json \
  --vocab artifacts/knots/vocab.json \
  --layout artifacts/knots/layout.json
```

## Talking to it

Same as barista, 115200 baud:

```bash
ls /dev/cu.usbmodem*
arduino-cli monitor -p /dev/cu.usbmodemNNNN -c baudrate=115200
```

```text
=== ESP32 KNOTS ===
ask a knot-tying question; the model writes the answer.
model: Vin=... Vout=... D=... L=... H=... F=... P=...
...
READY>
```

Notes:

- ASCII only — anything else is rejected with `(ascii only)`.
- Answers are drawn only from the curated knot-tying output alphabet, not
  free text.
- Overlong questions are rejected with `(question too long)`.
- Extending coverage further -- more paraphrases per knot, richer which-knot
  scenarios, or words the current procedures had to simplify around: add or
  edit entries in `research/knots/generate_corpus.py`'s `KNOT_STEPS`/
  `WHICH_KNOT`, using only words in `research/knots/output_words.json`
  (extend that list first via `build_word_list.py` if a new step genuinely
  needs a word it doesn't have), then rerun the pipeline commands above in
  order.
