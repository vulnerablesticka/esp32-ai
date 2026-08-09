# ESP32-S3 Barista

This sketch runs the 8.9M-parameter Barista model on an ESP32-S3 N16R8. A
question arrives over USB serial, the answer streams back there and, when a
panel is wired, to an OLED as well. The model lives in the custom `model` flash
partition at `0x110000`.

What makes it different from the TinyStories sketch is the vocabulary: Barista
**reads** 8,057 input tokens so it can take varied ASCII questions, and
**writes** only 854 output classes. The head is therefore its own tensor rather
than a view of the embedding, and a sampled index is an output class, not a
token id.

Two board targets are supported:

|            | devkit (default)                    | xiao_s3                          |
| ---------- | ------------------------------------ | --------------------------------- |
| Board      | generic ESP32-S3, 16MB flash, 8MB PSRAM | Seeed XIAO ESP32S3 (plain, non-Sense), 8MB flash, 8MB PSRAM |
| Display    | optional I2C OLED (`display.h`)      | none — serial only (`USE_DISPLAY=0`) |
| Model      | tinystories or barista               | barista only — tinystories' ~14.9MB model.bin does not fit 8MB flash |

## Toolchain setup - macOS

Assumes homebrew already present

```bash
brew install uv
uv sync                     # Python deps (uv.lock) into .venv/ — fetch_model.sh and
                             # deploy.sh's header/verify tools run through `uv run`
brew install arduino-cli
arduino-cli config init
arduino-cli config set board_manager.additional_urls \
  https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
arduino-cli core update-index
arduino-cli core install esp32:esp32
uv pip install esptool
```

`fetch_model.sh` also needs the `hf` CLI (`uv tool install huggingface_hub[cli]`
or `pip install huggingface_hub[cli]`) to download released models.

The `xiao_s3` target does not need the Adafruit_GFX / Adafruit_SH110X
libraries — `USE_DISPLAY=0` means `display.h` is never included. They're only
required for `devkit` builds with a panel wired.

`xiao_s3` uses FQBN `esp32:esp32:XIAO_ESP32S3`. Its `boards.txt` differs from
the generic esp32s3 board in two ways `deploy.sh` already accounts for: there
is no `custom` `PartitionScheme` value (only `default_8MB`, `max_app_8MB`,
`tinyuf2`, `tinyuf2_noota`), and `CDCOnBoot`'s value names are inverted
(`default` is Enabled here, `cdc` is Disabled). The `partitions.csv` swap
still works with `PartitionScheme=default_8MB` — the core's prebuild hooks
always copy the sketch directory's own `partitions.csv` over whichever
scheme's table last, regardless of which enumerated value is selected. If a
compile rejects some other menu option, check the exact keys for your
installed core version with:

```bash
arduino-cli board details -b esp32:esp32:XIAO_ESP32S3
```

## Build and flash

```bash
uv run scripts/fetch_model.sh barista               # download and verify the artifacts

uv run scripts/deploy.sh barista                    # devkit (default), with display
uv run scripts/deploy.sh barista xiao_s3            # XIAO ESP32S3, serial only
```

`deploy.sh` is the authoritative procedure. It regenerates the word tables and
encoder asset from the tokenizer/vocab/layout being deployed, runs the host
gates including `verify_tokenizer.py`, compiles, writes the model partition
and then the firmware, and prints the fingerprint the board should report
back. Compilation happens before either flash, so a build failure cannot leave
new weights under old firmware.

The tracked source of truth for each board is named after it —
`partitions_devboard.csv` and `partitions_xiao_s3.csv` (an 8MB-flash layout;
same `model` partition offset, smaller sizes). Arduino's `PartitionScheme=custom`
always reads a file literally named `partitions.csv` from the sketch
directory, with no option to point it elsewhere, so `deploy.sh` copies
whichever named file the target board needs over `partitions.csv` before
every compile. That file is therefore a build artifact, not a source of
truth, and is gitignored rather than restored afterward.

The model argument is mandatory: the board holds one model at a time, and
deploying replaces whichever is installed. On `xiao_s3`, only `barista` is
accepted — `deploy.sh` refuses `tinystories xiao_s3` outright.

## The artifact set

Barista needs four files, against TinyStories' two. All four are frozen with the
trained model and are published together:

| file | why the device needs it |
|---|---|
| `model.bin` | the weights |
| `tokenizer.json` | encodes the question on-device |
| `vocab.json` | turns an output class into its output text |
| `layout.json` | maps an output class to the input token id to feed forward |

`golden.txt` is produced by the exporter and is not distributed; the golden gate
is skipped when it is absent.

## Generated headers

`deploy.sh` rebuilds all three from the artifacts being deployed, every run. They
are gitignored, so a fresh clone has none until the first deploy.

| header | from | contents |
|---|---|---|
| `generated/barista_words.h` | `vocab.json` | the 854 output classes as C strings |
| `generated/barista_out2in.h` | `layout.json` | class to input token id |
| `generated/tokenizer_encoder.h` | `tokenizer.json` | 43,056 B BTK1 encoder asset |

## The feedback path

The head emits a class. Feeding it back in needs the input token id that class
corresponds to, which is what `out2in` holds:

```c
llm_forward(&model, BARISTA_OUT2IN[best], pos++, &scratch);
```

Three boot guards refuse to run rather than answer wrongly if the tables and the
weights disagree: `out_vocab` against `BARISTA_WORD_COUNT`, the encoder's widest
id against the model's input vocabulary, and `max(BARISTA_OUT2IN) + 1` against
the same. Flashing Barista firmware over the TinyStories model produces
`word table mismatch: model 25353 vs table 854` and stops.

## Output

Serial is always on. Type a question, press return. The answer streams a class at
a time and ends with a timing line.

The OLED is optional and is probed once at startup. If nothing answers on the
bus, one line is printed and every later draw becomes a no-op, so an absent or
unplugged panel costs a single probe rather than a failed write per class.

128x64 I2C mono OLED, four wires:

```
GND -> GND    VCC -> 3V3    SCL -> GPIO46    SDA -> GPIO18
```

Set `OLED_CONTROLLER` to match the panel: 1.3" is usually SH1106, 0.96" usually
SSD1306. The panel shows the question, a rule, then the answer building up one
output piece at a time, scrolling once it fills.

## Compile switches

All are `#ifndef`-guarded and set the same way:

```bash
arduino-cli compile --build-property compiler.cpp.extra_flags=-DUSE_DISPLAY=0 ...
```

| switch | default | effect |
|---|---|---|
| `USE_DISPLAY` | 1 | 0 builds serial-only, with no display code at all |
| `BARISTA_DUAL_CORE` | 1 | 0 runs every matvec on one core |
| `BARISTA_PROFILE` | 0 | 1 prints where the time goes, per answer |
| `OLED_CONTROLLER` | `OLED_SH1106` | or `OLED_SSD1306` |
| `OLED_ADDR` | `0x3C` | some panels are `0x3D` |

## Expected boot output

The deployed model has SHA-256:

```text
1359a1cb74de4143d630c2c192990de814cd47255bcdfa9cc135f07ef0a39fc4
```

```text
=== ESP32 BARISTA ===
ask an espresso question; the model writes the answer.
model: Vin=8057 Vout=854 D=128 L=6 H=4 F=384 P=128
scratch in SRAM: 20940 B
norms in SRAM: 20/20 vectors, 10656 B
sram free 288 KB
int8-staged 44 tensors | psram free 5.55 MB
build: magic=00454c50 bytes=4600186 fp=e602146b scratch_sram=1 fallbacks=0
config: profile=0 dual_core_requested=1 dual_core_active=1 display_enabled=1 display_present=1
READY>
```

`build:` identifies the weights: `fp` is FNV-1a over the mapped image, and
`deploy.sh` prints the same value for the file it flashed. The two must agree.

`config:` identifies the build switches, so a measurement can require the
configuration it claims rather than trusting a label.
`benchmark_device.py --expect key=value` checks against this line and refuses on
a mismatch.

## Measured

On this board, eight fixed prompts, two passes per mode.

An output piece is one emitted class. Punctuation is a class, so pieces are not
readable words: over the benchmark set, 253 pieces render as 213 words.

| | |
|---|---:|
| serial only | 60.2 ms/piece |
| with the OLED | 88.8 ms/piece |
| panel redraw | +28.6 ms/piece |
| per forward | 49.6 ms |
| dual core against single | 1.75x |

Measure with `USE_DISPLAY=0`, or the per-piece panel redraw lands in the total.

## Running a step by hand

These are the pieces worth running on their own. For flashing, use the
script: regenerating the headers and writing the two images in order are part
of what makes the result correct.

Host gates:

```bash
cc -O3 -Wall -Wextra -DLLM_INT8_ACT=1 -o /tmp/staging runtime/host_verify/staging_verify.c -lm
/tmp/staging artifacts/barista/model.bin

uv run python firmware/esp32_barista/tools/verify_tokenizer.py \
  --tokenizer artifacts/barista/tokenizer.json \
  --vocab artifacts/barista/vocab.json \
  --layout artifacts/barista/layout.json
```

The first checks the staged int8 kernel and platform hooks, which is what the
device runs. The second checks the on-device encoder's ids against the
tokenizer and that the word map resolves to the ids it claims.

Regenerate the device headers by hand:

```bash
uv run python firmware/esp32_barista/tools/generate_vocab_headers.py \
  --vocab artifacts/barista/vocab.json \
  --layout artifacts/barista/layout.json \
  --out-dir firmware/esp32_barista/generated

uv run python firmware/esp32_barista/tools/generate_tokenizer_header.py \
  --tokenizer artifacts/barista/tokenizer.json \
  --out firmware/esp32_barista/generated/tokenizer_encoder.h
```

## Talking to it

Everything happens over the same USB port used to flash it, at 115200 baud:

```bash
ls /dev/cu.usbmodem*                                   # port deploy.sh printed
arduino-cli monitor -p /dev/cu.usbmodemNNNN -c baudrate=115200
```

`screen /dev/cu.usbmodemNNNN 115200` or `python3 -m serial.tools.miniterm
/dev/cu.usbmodemNNNN 115200` work too.

On boot you'll see:

```text
=== ESP32 BARISTA ===
ask an espresso question; the model writes the answer.
model: Vin=... Vout=... D=... L=... H=... F=... P=...
...
READY>
```

Type a plain-ASCII espresso question and hit Enter. The answer streams after
`A: `, followed by a stats line:

```text
A: increase your dose and grind finer
[7 pieces, 612 ms, 11.4 pieces/s]
READY>
```

Notes:

- ASCII only — anything else is rejected with `(ascii only)`.
- Answers are drawn only from the 854-class espresso output alphabet, not
  free text: it cannot say a word outside that set, and there are no digit
  tokens, so it cannot invent numbers.
- Overlong questions are rejected with `(question too long)`.
- The deployed barista model has SHA-256
  `1359a1cb74de4143d630c2c192990de814cd47255bcdfa9cc135f07ef0a39fc4`
  (4,600,186 bytes), pinned in `scripts/fetch_model.sh`. `deploy.sh` prints
  the same FNV-1a fingerprint it flashed; the board's boot line must match it.
