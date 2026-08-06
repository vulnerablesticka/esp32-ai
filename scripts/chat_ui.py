"""A chat front end for the barista board's USB serial Q&A loop.

The firmware (firmware/esp32_barista/esp32_barista.ino) speaks a small
line-based protocol: send an ASCII question plus a newline, it streams back
"A: <words>", then a "[N pieces, M ms, R pieces/s]" stats line, then a
"READY>" prompt. This just wraps that protocol in a chat widget: one
serial.Serial connection is held open in session state across Streamlit's
reruns, a question is sent on submit, and the reply is drawn into the
assistant bubble as bytes arrive off the wire, mirroring the word-by-word
stream the firmware itself produces.

    uv sync --extra chat
    uv run streamlit run scripts/chat_ui.py

Needs pyserial and streamlit, both in the "chat" extra.
"""

import re
import sys
import time
from dataclasses import asdict, dataclass

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
STATS_RE = re.compile(r"\[(\d+) pieces, (\d+) ms, ([\d.]+) pieces/s\]")
# Only present when the flashed firmware was built with BARISTA_PROFILE=1
# (off by default); mirrors scripts/benchmark_device.py's BARISTA_PROFILE regex.
PROFILE_RE = re.compile(
    r"\[profile (\d+) fwd \| input ([\d.]+)ms .*?\| attn ([\d.]+)ms .*?\| "
    r"ffn ([\d.]+)ms .*?\| ple ([\d.]+)ms .*?\| head ([\d.]+)ms .*?\| "
    r"accounted ([\d.]+)%\]")
READY_SUFFIX_RE = re.compile(r"READY>\s*$")
STATS_SUFFIX_RE = re.compile(r"\n?\[\d+ pieces.*?\]\s*$", flags=re.DOTALL)
READY = "READY>"

# Device-path shape of a board's native USB CDC port, by platform. macOS also
# exposes a tty.usbmodem* pair for the same device; cu.* is the callout device
# and the one that behaves correctly for a program-initiated connection.
_PORT_PATTERN = {
    "darwin": re.compile(r"cu\.usbmodem"),
    "linux": re.compile(r"ttyACM|ttyUSB"),
    "win32": re.compile(r"^COM\d+$"),
}


@dataclass(slots=True)
class Stats:
    """The board's own "[N pieces, M ms, R pieces/s]" line, parsed."""
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
    parsed. Only present on firmware built with BARISTA_PROFILE=1."""
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
    uses to get a clean boot, then drain and return the boot banner."""
    ser: serial.Serial = st.session_state.ser
    ser.reset_input_buffer()
    ser.dtr = False
    time.sleep(0.3)
    ser.dtr = True
    buf, deadline = "", time.time() + 8.0
    while time.time() < deadline:
        chunk = ser.read(4096)
        if chunk:
            buf += chunk.decode("utf-8", "replace")
            if buf.rstrip().endswith(READY):
                break
        else:
            time.sleep(POLL_INTERVAL_S)
    return buf


def visible(buf: str) -> str:
    """What to show while a reply is still streaming: text after "A: " if
    that prefix has arrived, the raw buffer otherwise (covers boot noise and
    the no-prefix rejection notices), with the trailing READY> and a
    completed stats line trimmed off."""
    text = READY_SUFFIX_RE.sub("", buf)
    text = text.split("A: ", 1)[1] if "A: " in text else text
    text = STATS_SUFFIX_RE.sub("", text)
    return text.strip()


def send_question(
    question: str, placeholder: DeltaGenerator,
) -> tuple[str, bool, Timing]:
    """Send one question and stream the reply into placeholder as it arrives.
    Returns the raw buffer received, whether it ended on READY> (False means
    REPLY_TIMEOUT_S was hit first, e.g. the board is stuck or gone), and the
    milestone timestamps used to derive the Metrics tab's latency numbers.

    The milestones line up with two points in esp32_barista.ino's answer():
    Serial.print("A: ") fires right after the prefill forwards finish and
    before decoding starts, so the first byte of "A: " is a prefill-done
    marker; the first visible answer byte after that is the first token."""
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
    """Final parse of a completed reply: the answer (or notice) text, and the
    pieces/ms/pieces-per-s stats if a stats line was present."""
    stats: Stats | None = None
    m = STATS_RE.search(buf)
    if m:
        stats = Stats(pieces=int(m.group(1)), ms=int(m.group(2)),
                       pieces_per_s=float(m.group(3)))
    return visible(buf), stats


def parse_profile(buf: str) -> Profile | None:
    """The on-device input/attn/ffn/ple/head timing breakdown, if the reply
    carries one (requires firmware built with BARISTA_PROFILE=1)."""
    m = PROFILE_RE.search(buf)
    if not m:
        return None
    return Profile(
        forwards=int(m.group(1)), input_ms=float(m.group(2)),
        attn_ms=float(m.group(3)), ffn_ms=float(m.group(4)),
        ple_ms=float(m.group(5)), head_ms=float(m.group(6)),
        accounted_pct=float(m.group(7)))


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
    """Draw the connection controls, status, and boot banner in the sidebar."""
    with st.sidebar:
        st.subheader("Board")
        connected = st.session_state.ser is not None
        port = st.text_input("Serial port", value=st.session_state.port or discover_port())
        col1, col2 = st.columns(2)
        if not connected:
            if col1.button("Connect", use_container_width=True):
                try:
                    connect(port)
                    st.rerun()
                except serial.SerialException as e:
                    st.error(f"could not open {port}: {e}")
        else:
            if col1.button("Disconnect", use_container_width=True):
                disconnect()
                st.rerun()
        if col2.button("Reset board", use_container_width=True, disabled=not connected):
            with st.spinner("resetting..."):
                banner = reset_board()
            st.session_state.messages = []
            st.session_state.metrics = []
            st.session_state.banner = banner
            st.rerun()

        status = f"connected — {st.session_state.port}" if connected else "not connected"
        st.markdown(f"**status:** {status}")

        banner: str = st.session_state.get("banner", "")
        if banner:
            with st.expander("boot banner"):
                st.code(banner, language=None)

        st.caption("ASCII-only espresso questions. Answers are drawn from a "
                   "fixed 854-class output alphabet, not free text.")


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
    state."""
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


def render_metrics() -> None:
    """Per-turn latency table and charts for the conversation so far.

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
                   f"Only appears on firmware built with BARISTA_PROFILE=1.")

    st.subheader("Turns")
    st.dataframe(
        df[["question", "ttft_ms", "prefill_ms", "device_pieces", "device_ms",
            "device_pieces_per_s", "complete"]],
        use_container_width=True,
    )


def main() -> None:
    """Entry point: page setup, session-state defaults, sidebar, and the
    Chat/Metrics tabs. chat_input is called before the tabs so its return
    value is available either way, but as Streamlit's pinned-to-bottom
    widget its own rendering position is independent of that call site."""
    st.set_page_config(page_title="ESP32 Barista", page_icon="☕")
    st.title("☕ ESP32 Barista")

    st.session_state.setdefault("ser", None)
    st.session_state.setdefault("port", "")
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("metrics", [])

    render_sidebar()

    question = st.chat_input(
        "Ask an espresso question...",
        disabled=st.session_state.ser is None,
    )

    chat_tab, metrics_tab = st.tabs(["💬 Chat", "📊 Metrics"])
    with chat_tab:
        render_history()
        if question:
            handle_question(question)
    with metrics_tab:
        render_metrics()


main()
