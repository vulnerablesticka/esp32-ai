"""A chat front end for the Xiao board's USB serial output, agnostic to which
of the three models (firmware/esp32_barista, esp32_knots, esp32_tinystories)
is currently flashed.

Barista and knots speak the same line-based Q&A protocol: send an ASCII
question plus a newline, they stream back "A: <words>", then a "[N pieces,
M ms, R pieces/s]" stats line, then a "READY>" prompt. TinyStories has no
such loop - it generates one fixed-prompt story once in setup() and never
reads Serial again, ending in its own "--- N tokens in S s ---" /
"throughput: ..." lines instead of READY>. Which protocol applies is
detected from the boot banner (see detect_model_kind), captured by resetting
the board from the sidebar.

For the Q&A models, one serial.Serial connection is held open in session
state across Streamlit's reruns, a question is sent on submit, and the reply
is drawn into the assistant bubble as bytes arrive off the wire, mirroring
the word-by-word stream the firmware itself produces. For TinyStories, the
generated story is captured on reset and shown as a single message; there is
no question to send.

    uv sync --extra chat
    uv run streamlit run scripts/chat_ui.py

Needs pyserial and streamlit, both in the "chat" extra.
"""

import re
import sys
import time
from dataclasses import asdict, dataclass
from enum import StrEnum

import pandas as pd
import serial
import streamlit as st
from serial.tools import list_ports
from streamlit.delta_generator import DeltaGenerator

BAUD = 115200
REPLY_TIMEOUT_S = 60.0
# Tight enough that per-word arrivals are distinguishable at the pieces/s this
# board runs at (tens to low hundreds of ms/piece - see RESULTS.md); this loop
# is what the metrics timestamps below are measured against.
POLL_INTERVAL_S = 0.005

# ---- barista / knots: interactive Q&A protocol -----------------------------
QA_STATS_RE = re.compile(r"\[(\d+) pieces, (\d+) ms, ([\d.]+) pieces/s\]")
# Only present when the flashed firmware was built with BARISTA_PROFILE=1 or
# KNOTS_PROFILE=1 (off by default); mirrors scripts/benchmark_device.py's regex.
QA_PROFILE_RE = re.compile(
    r"\[profile (\d+) fwd \| input ([\d.]+)ms .*?\| attn ([\d.]+)ms .*?\| "
    r"ffn ([\d.]+)ms .*?\| ple ([\d.]+)ms .*?\| head ([\d.]+)ms .*?\| "
    r"accounted ([\d.]+)%\]")
READY_SUFFIX_RE = re.compile(r"READY>\s*$")
QA_STATS_SUFFIX_RE = re.compile(r"\n?\[\d+ pieces.*?\]\s*$", flags=re.DOTALL)
READY = "READY>"

# ---- tinystories: one-shot generation at boot ------------------------------
STORY_DONE_MARKER = "profile ms/token:"
STORY_STATS_RE = re.compile(
    r"--- (\d+) tokens in ([\d.]+) s ---\s*"
    r"throughput: ([\d.]+) tok/s\s+\(([\d.]+) ms/token\)")
STORY_PROFILE_RE = re.compile(
    r"profile ms/token: input ([\d.]+) \| attn ([\d.]+) \| ffn ([\d.]+) \| "
    r"ple ([\d.]+) \| head ([\d.]+)")

# TinyStories runs its one generation inline in setup(), before ever reaching
# loop() - a ~200-token story at ~10 tok/s (firmware/esp32_tinystories/README.md)
# takes on the order of 20s, far longer than barista/knots take to reach
# READY>. One shared timeout covers both: the Q&A boards exit early on READY>
# regardless, so only TinyStories actually waits this long.
RESET_TIMEOUT_S = 40.0


class ModelKind(StrEnum):
    """Which of the three flashed models is on the board, detected from its
    boot banner. Barista and knots share the interactive Q&A protocol,
    TinyStories doesn't - see module docstring."""
    BARISTA = "barista"
    KNOTS = "knots"
    TINYSTORIES = "tinystories"


# Models with no interactive protocol: they produce their output once at
# boot and never read Serial again, so there is no question to send.
ONE_SHOT_KINDS = frozenset({ModelKind.TINYSTORIES})

_BANNER_MARKERS: tuple[tuple[re.Pattern[str], ModelKind], ...] = (
    (re.compile(r"ESP32 BARISTA"), ModelKind.BARISTA),
    (re.compile(r"ESP32 KNOTS"), ModelKind.KNOTS),
    (re.compile(r"PLE TinyLM"), ModelKind.TINYSTORIES),
)

_SIDEBAR_CAPTION = {
    ModelKind.BARISTA: "ASCII-only espresso questions. Answers are drawn "
                        "from a fixed output-class alphabet, not free text.",
    ModelKind.KNOTS: "ASCII-only knot-tying questions. Answers are drawn "
                      "from a fixed output-class alphabet, not free text.",
    ModelKind.TINYSTORIES: "Generates one fixed-prompt story (\"Once upon a "
                            "time...\") at boot. Reset the board for a new one.",
}

# Device-path shape of a board's native USB CDC port, by platform. macOS also
# exposes a tty.usbmodem* pair for the same device; cu.* is the callout device
# and the one that behaves correctly for a program-initiated connection.
_PORT_PATTERN = {
    "darwin": re.compile(r"cu\.usbmodem"),
    "linux": re.compile(r"ttyACM|ttyUSB"),
    "win32": re.compile(r"^COM\d+$"),
}


def detect_model_kind(banner: str) -> ModelKind | None:
    """Which model produced this boot banner, or None if it doesn't (yet)
    contain a recognizable marker line."""
    for pattern, kind in _BANNER_MARKERS:
        if pattern.search(banner):
            return kind
    return None


@dataclass(slots=True)
class Stats:
    """The board's own "[N pieces, M ms, R pieces/s]" line, parsed.
    Barista/knots only - see StoryStats for TinyStories' equivalent."""
    pieces: int
    ms: int
    pieces_per_s: float


@dataclass(slots=True)
class Message:
    """One chat bubble: who said it, the text, and the reply's Stats if it
    was a completed assistant answer."""
    role: str
    content: str
    stats: Stats | None = None


@dataclass(slots=True)
class Timing:
    """Host clock timestamps (time.time(), seconds) taken as bytes arrive
    off the wire during one exchange. None until that milestone happens, or
    forever if it never does (e.g. prefill_done_at stays None for a
    no-prefix rejection reply like "(ascii only)")."""
    sent_at: float
    prefill_done_at: float | None = None
    first_token_at: float | None = None
    done_at: float | None = None


@dataclass(slots=True)
class Profile:
    """The board's own "[profile ...]" input/attn/ffn/ple/head breakdown,
    parsed. Only present on barista/knots firmware built with
    BARISTA_PROFILE=1 / KNOTS_PROFILE=1."""
    forwards: int
    input_ms: float
    attn_ms: float
    ffn_ms: float
    ple_ms: float
    head_ms: float
    accounted_pct: float


@dataclass(slots=True)
class TurnMetrics:
    """One question/answer exchange's latency, for the Metrics tab. Fields
    are grouped by clock: prefill_ms/ttft_ms/decode_ms/host_total_ms are
    host-observed (time.time() deltas around send_question's read loop), so
    they carry USB and polling overhead on top of on-device compute;
    device_pieces/device_ms/device_pieces_per_s are the board's own clock,
    off the final stats line, and are what to trust for comparing runs."""
    question: str
    prefill_ms: float | None
    ttft_ms: float | None
    decode_ms: float | None
    host_total_ms: float | None
    device_pieces: int | None
    device_ms: int | None
    device_pieces_per_s: float | None
    complete: bool
    profile: Profile | None


@dataclass(slots=True)
class StoryStats:
    """TinyStories' own "--- N tokens in S s ---" / "throughput: ..." lines,
    parsed. Its one-shot equivalent of Stats above."""
    tokens: int
    seconds: float
    tokens_per_s: float
    ms_per_token: float


@dataclass(slots=True)
class StoryProfile:
    """TinyStories' own "profile ms/token: ..." line, parsed. Unlike
    barista/knots' Profile, this is always present: LLM_PROFILE is
    unconditionally on in that sketch rather than an opt-in build flag."""
    input_ms: float
    attn_ms: float
    ffn_ms: float
    ple_ms: float
    head_ms: float


def discover_port() -> str:
    """Best guess at the board's port, cross-platform. Queries the OS device
    registry via pyserial rather than globbing a fixed path, since that path
    shape differs by platform (/dev/cu.usbmodem* on macOS, /dev/ttyACM* or
    /dev/ttyUSB* on Linux, COM<n> on Windows). Prefers a port matching the
    current platform's native-USB-CDC shape; falls back to whatever pyserial
    finds if nothing matches or the platform isn't one of the three above."""
    ports = sorted(p.device for p in list_ports.comports())
    pattern = _PORT_PATTERN.get(sys.platform)
    matched = [p for p in ports if pattern and pattern.search(p)]
    return (matched or ports or [""])[0]


def connect(port: str) -> None:
    """Open the serial port and store the connection in session state, so it
    survives Streamlit's rerun-per-interaction model."""
    st.session_state.ser = serial.Serial(port, BAUD, timeout=0)
    st.session_state.port = port


def disconnect() -> None:
    """Close the serial port, if open, and clear it from session state."""
    ser: serial.Serial | None = st.session_state.get("ser")
    if ser is not None:
        ser.close()
    st.session_state.ser = None


def reset_board() -> str:
    """Hardware-reset via DTR, the same technique scripts/benchmark_device.py
    uses to get a clean boot, then drain and return everything up to the
    point the board signals it's done: a READY> prompt for barista/knots, or
    TinyStories' own end-of-generation profile line."""
    ser: serial.Serial = st.session_state.ser
    ser.reset_input_buffer()
    ser.dtr = False
    time.sleep(0.3)
    ser.dtr = True
    buf, deadline = "", time.time() + RESET_TIMEOUT_S
    while time.time() < deadline:
        chunk = ser.read(4096)
        if chunk:
            buf += chunk.decode("utf-8", "replace")
            if buf.rstrip().endswith(READY) or STORY_DONE_MARKER in buf:
                break
        else:
            time.sleep(POLL_INTERVAL_S)
    return buf


def visible(buf: str) -> str:
    """What to show while a Q&A reply is still streaming: text after "A: " if
    that prefix has arrived, the raw buffer otherwise (covers boot noise and
    the no-prefix rejection notices), with the trailing READY> and a
    completed stats line trimmed off. Barista/knots only."""
    text = READY_SUFFIX_RE.sub("", buf)
    text = text.split("A: ", 1)[1] if "A: " in text else text
    text = QA_STATS_SUFFIX_RE.sub("", text)
    return text.strip()


def send_question(
    question: str, placeholder: DeltaGenerator,
) -> tuple[str, bool, Timing]:
    """Send one question and stream the reply into placeholder as it arrives.
    Returns the raw buffer received, whether it ended on READY> (False means
    REPLY_TIMEOUT_S was hit first, e.g. the board is stuck, gone, or running
    TinyStories' non-interactive firmware), and the milestone timestamps used
    to derive the Metrics tab's latency numbers.

    The milestones line up with two points in esp32_barista.ino / esp32_knots.ino's
    answer(): Serial.print("A: ") fires right after the prefill forwards
    finish and before decoding starts, so the first byte of "A: " is a
    prefill-done marker; the first visible answer byte after that is the
    first token."""
    ser: serial.Serial = st.session_state.ser
    ser.reset_input_buffer()
    sent_at = time.time()
    ser.write((question + "\n").encode("ascii", "replace"))
    ser.flush()
    timing = Timing(sent_at=sent_at)
    buf, deadline = "", sent_at + REPLY_TIMEOUT_S
    while time.time() < deadline:
        chunk = ser.read(4096)
        if not chunk:
            time.sleep(POLL_INTERVAL_S)
            continue
        now = time.time()
        buf += chunk.decode("utf-8", "replace")
        if timing.prefill_done_at is None and "A: " in buf:
            timing.prefill_done_at = now
        shown = visible(buf)
        if shown and timing.first_token_at is None:
            timing.first_token_at = now
        placeholder.markdown(shown or "…")
        if buf.rstrip().endswith(READY):
            timing.done_at = now
            return buf, True, timing
    return buf, False, timing


def parse_reply(buf: str) -> tuple[str, Stats | None]:
    """Final parse of a completed Q&A reply: the answer (or notice) text, and
    the pieces/ms/pieces-per-s stats if a stats line was present."""
    stats: Stats | None = None
    m = QA_STATS_RE.search(buf)
    if m:
        stats = Stats(pieces=int(m.group(1)), ms=int(m.group(2)),
                       pieces_per_s=float(m.group(3)))
    return visible(buf), stats


def parse_profile(buf: str) -> Profile | None:
    """The on-device input/attn/ffn/ple/head timing breakdown, if the reply
    carries one (requires barista/knots firmware built with
    BARISTA_PROFILE=1 / KNOTS_PROFILE=1)."""
    m = QA_PROFILE_RE.search(buf)
    if not m:
        return None
    return Profile(
        forwards=int(m.group(1)), input_ms=float(m.group(2)),
        attn_ms=float(m.group(3)), ffn_ms=float(m.group(4)),
        ple_ms=float(m.group(5)), head_ms=float(m.group(6)),
        accounted_pct=float(m.group(7)))


def parse_story(banner: str) -> tuple[str, StoryStats | None, StoryProfile | None]:
    """The generated story text (between the ">>> " prompt and the trailing
    stats block), plus TinyStories' own throughput and per-stage timing."""
    text = banner.split(">>> ", 1)[1] if ">>> " in banner else banner
    text = re.split(r"\n\n?--- \d+ tokens", text, maxsplit=1)[0]

    stats: StoryStats | None = None
    m = STORY_STATS_RE.search(banner)
    if m:
        stats = StoryStats(tokens=int(m.group(1)), seconds=float(m.group(2)),
                            tokens_per_s=float(m.group(3)),
                            ms_per_token=float(m.group(4)))

    profile: StoryProfile | None = None
    m = STORY_PROFILE_RE.search(banner)
    if m:
        profile = StoryProfile(
            input_ms=float(m.group(1)), attn_ms=float(m.group(2)),
            ffn_ms=float(m.group(3)), ple_ms=float(m.group(4)),
            head_ms=float(m.group(5)))

    return text.strip(), stats, profile


def build_turn_metrics(
    question: str, buf: str, timing: Timing, complete: bool,
) -> TurnMetrics:
    """Combine one exchange's host-observed timestamps with the board's own
    reported stats into one row for the Metrics tab."""
    def delta_ms(a: float | None, b: float | None) -> float | None:
        """Milliseconds between two Timing timestamps, or None if either
        milestone never happened."""
        return (b - a) * 1000 if a is not None and b is not None else None

    _, stats = parse_reply(buf)
    return TurnMetrics(
        question=question,
        prefill_ms=delta_ms(timing.sent_at, timing.prefill_done_at),
        ttft_ms=delta_ms(timing.sent_at, timing.first_token_at),
        decode_ms=delta_ms(timing.first_token_at, timing.done_at),
        host_total_ms=delta_ms(timing.sent_at, timing.done_at),
        device_pieces=stats.pieces if stats else None,
        device_ms=stats.ms if stats else None,
        device_pieces_per_s=stats.pieces_per_s if stats else None,
        complete=complete,
        profile=parse_profile(buf),
    )


def render_stats_caption(stats: Stats) -> None:
    """Render one reply's piece count and timing as a small caption."""
    st.caption(f"{stats.pieces} pieces, {stats.ms} ms, "
               f"{stats.pieces_per_s:.1f} pieces/s")


def render_sidebar() -> None:
    """Draw the connection controls, status, and boot banner in the sidebar.
    Resetting the board is also how the model currently on it is identified
    (see detect_model_kind) - there's no way to ask a connected-but-not-yet-
    reset board what it is."""
    with st.sidebar:
        st.subheader("Board")
        connected = st.session_state.ser is not None
        port = st.text_input("Serial port", value=st.session_state.port or discover_port())
        col1, col2 = st.columns(2)
        if not connected:
            if col1.button("Connect", width="stretch"):
                try:
                    connect(port)
                    st.rerun()
                except serial.SerialException as e:
                    st.error(f"could not open {port}: {e}")
        else:
            if col1.button("Disconnect", width="stretch"):
                disconnect()
                st.rerun()
        if col2.button("Reset board", width="stretch", disabled=not connected):
            with st.spinner("resetting..."):
                banner = reset_board()
            kind = detect_model_kind(banner)
            st.session_state.banner = banner
            st.session_state.model_kind = kind
            st.session_state.messages = []
            st.session_state.metrics = []
            st.session_state.story_stats = None
            st.session_state.story_profile = None
            if kind == ModelKind.TINYSTORIES:
                text, story_stats, story_profile = parse_story(banner)
                st.session_state.story_stats = story_stats
                st.session_state.story_profile = story_profile
                if text:
                    if story_stats:
                        text += (f"\n\n*{story_stats.tokens} tokens in "
                                  f"{story_stats.seconds:.1f}s — "
                                  f"{story_stats.tokens_per_s:.1f} tok/s "
                                  f"({story_stats.ms_per_token:.1f} ms/token)*")
                    st.session_state.messages.append(
                        Message(role="assistant", content=text))
            st.rerun()

        kind: ModelKind | None = st.session_state.get("model_kind")
        status = f"connected — {st.session_state.port}" if connected else "not connected"
        if kind:
            status += f" — {kind}"
        st.markdown(f"**status:** {status}")

        banner: str = st.session_state.get("banner", "")
        if banner:
            with st.expander("boot banner"):
                st.code(banner, language=None)

        if kind:
            st.caption(_SIDEBAR_CAPTION.get(
                kind, "Reset the board to identify the loaded model."))

def render_history() -> None:
    """Redraw the prior conversation from session state."""
    for msg in st.session_state.messages:
        with st.chat_message(msg.role):
            st.markdown(msg.content)
            if msg.stats:
                render_stats_caption(msg.stats)


def handle_question(question: str) -> None:
    """Render the user's question, send it, stream the assistant's reply into
    the chat, and append both the message and its latency metrics to session
    state. Barista/knots only - the Chat tab never collects a question for a
    one-shot model (see ONE_SHOT_KINDS in main())."""
    st.session_state.messages.append(Message(role="user", content=question))
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("…")
        try:
            raw, complete, timing = send_question(question, placeholder)
        except serial.SerialException as e:
            error = f"serial error: {e}"
            placeholder.markdown(error)
            st.session_state.messages.append(Message(role="assistant", content=error))
            return

        answer, stats = parse_reply(raw)
        if not complete:
            answer = (answer or "(no reply)") + "\n\n*(timed out waiting for READY>)*"
        placeholder.markdown(answer)
        if stats:
            render_stats_caption(stats)
        st.session_state.messages.append(
            Message(role="assistant", content=answer, stats=stats))
        st.session_state.metrics.append(
            build_turn_metrics(question, raw, timing, complete))


def render_qa_metrics() -> None:
    """Per-turn latency table and charts for the conversation so far.
    Barista/knots only.

    ttft_ms/prefill_ms/decode_ms are host-observed (time.time() around
    send_question's read loop), so they include USB and polling overhead on
    top of on-device compute - useful for judging felt responsiveness, not
    as a substitute for scripts/benchmark_device.py's on-device timing.
    device_ms/device_pieces_per_s are the board's own numbers, off its own
    clock, and are what to trust for comparing runs."""
    metrics: list[TurnMetrics] = st.session_state.metrics
    if not metrics:
        st.info("Ask a question in the Chat tab to start collecting metrics.")
        return

    # asdict() also recurses into the nested Profile dataclass; that column
    # isn't charted below, only read back off `latest` directly.
    df = pd.DataFrame([asdict(m) for m in metrics])
    df.index = pd.RangeIndex(1, len(df) + 1, name="turn")

    latest = metrics[-1]
    cols = st.columns(4)
    cols[0].metric("time to first token",
                    f"{latest.ttft_ms:.0f} ms" if latest.ttft_ms is not None else "—")
    cols[1].metric("prefill",
                    f"{latest.prefill_ms:.0f} ms" if latest.prefill_ms is not None else "—")
    cols[2].metric("device pieces/s",
                    f"{latest.device_pieces_per_s:.1f}"
                    if latest.device_pieces_per_s is not None else "—")
    cols[3].metric("device total",
                    f"{latest.device_ms} ms" if latest.device_ms is not None else "—")
    st.caption("First three are host-observed (send to first relevant byte); "
               "device figures are the board's own reported stats.")

    latency_cols = [c for c in ("ttft_ms", "prefill_ms", "decode_ms") if df[c].notna().any()]
    if latency_cols:
        st.subheader("Latency over the conversation")
        st.line_chart(df[latency_cols])

    if df["device_pieces_per_s"].notna().any():
        st.subheader("Device-reported throughput")
        st.line_chart(df[["device_pieces_per_s"]])

    if latest.profile:
        prof = latest.profile
        st.subheader("Latest reply: on-device time breakdown")
        prof_df = pd.DataFrame(
            {"ms": [prof.input_ms, prof.attn_ms, prof.ffn_ms,
                     prof.ple_ms, prof.head_ms]},
            index=["input", "attn", "ffn", "ple", "head"])
        st.bar_chart(prof_df)
        st.caption(f"{prof.forwards} forward passes, "
                   f"{prof.accounted_pct:.0f}% of wall time accounted for. "
                   f"Only appears on firmware built with profiling enabled.")

    st.subheader("Turns")
    st.dataframe(
        df[["question", "ttft_ms", "prefill_ms", "device_pieces", "device_ms",
            "device_pieces_per_s", "complete"]],
        width="stretch",
    )


def render_story_metrics() -> None:
    """TinyStories' equivalent of the Metrics tab: since there's one
    generation per reset rather than one row per turn, this shows that single
    run's throughput and on-device time breakdown instead of a turn table."""
    stats: StoryStats | None = st.session_state.get("story_stats")
    if not stats:
        st.info("Reset the board in the sidebar to generate a story and see its timing.")
        return

    cols = st.columns(3)
    cols[0].metric("tokens generated", stats.tokens)
    cols[1].metric("throughput", f"{stats.tokens_per_s:.1f} tok/s")
    cols[2].metric("per token", f"{stats.ms_per_token:.1f} ms")
    st.caption("The board's own reported numbers for its one fixed-prompt "
               "generation. Reset to regenerate and refresh these.")

    profile: StoryProfile | None = st.session_state.get("story_profile")
    if profile:
        st.subheader("On-device time breakdown")
        prof_df = pd.DataFrame(
            {"ms/token": [profile.input_ms, profile.attn_ms, profile.ffn_ms,
                           profile.ple_ms, profile.head_ms]},
            index=["input", "attn", "ffn", "ple", "head"])
        st.bar_chart(prof_df)


def render_metrics() -> None:
    """Metrics tab: per-turn Q&A latency for barista/knots, or the single
    generation's throughput/timing for TinyStories."""
    if st.session_state.get("model_kind") == ModelKind.TINYSTORIES:
        render_story_metrics()
    else:
        render_qa_metrics()


def main() -> None:
    """Entry point: page setup, session-state defaults, sidebar, and the
    Chat/Metrics tabs. chat_input is called before the tabs so its return
    value is available either way, but as Streamlit's pinned-to-bottom
    widget its own rendering position is independent of that call site."""
    st.set_page_config(page_title="ESP32 Chat", page_icon="☕")
    st.title("☕ ESP32 Chat")

    st.session_state.setdefault("ser", None)
    st.session_state.setdefault("port", "")
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("metrics", [])
    st.session_state.setdefault("model_kind", None)
    st.session_state.setdefault("story_stats", None)
    st.session_state.setdefault("story_profile", None)

    render_sidebar()

    one_shot = st.session_state.model_kind in ONE_SHOT_KINDS
    question = st.chat_input(
        "Reset the board in the sidebar to generate a new story..."
        if one_shot else "Ask a question...",
        disabled=st.session_state.ser is None or one_shot,
    )

    chat_tab, metrics_tab = st.tabs(["💬 Chat", "📊 Metrics"])
    with chat_tab:
        render_history()
        if question:
            handle_question(question)
    with metrics_tab:
        render_metrics()


main()
