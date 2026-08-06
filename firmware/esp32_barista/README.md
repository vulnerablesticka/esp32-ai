# ESP32-S3 Barista

Espresso question answering, USB serial in and out. It reads a wide ASCII
vocabulary so it can take varied questions, but writes only from a small
854-class espresso alphabet, so answers stay on topic and words outside that
alphabet are literally unsayable. The model lives in the custom `model` flash
partition at `0x110000`.

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
(`default` is Enabled here, `cdc` is Disabled). The custom `partitions.csv`
swap still works with `PartitionScheme=default_8MB` — the core's prebuild
hooks always copy the sketch directory's own `partitions.csv` over whichever
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

For `xiao_s3`, `deploy.sh` temporarily swaps in `partitions_xiao_s3.csv` (an
8MB-flash layout; same `model` partition offset, smaller sizes) for the build
and restores the devkit `partitions.csv` afterward, so the tracked file is
never left modified.

The model argument is mandatory: the board holds one model at a time, and
deploying replaces whichever is installed. On `xiao_s3`, only `barista` is
accepted — `deploy.sh` refuses `tinystories xiao_s3` outright.

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
