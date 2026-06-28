#!/usr/bin/env python3
"""

Pipeline:
    yt-dlp (fetch <=240p stream)  ->  ffmpeg (decode / scale / colour-adjust)
    ->  numpy (luminance -> char ramp, + optional ANSI colour)  ->  terminal.

Output is monochrome by default. --color tints each cell with the source pixel:
    256        xterm 256-colour palette (broad terminal support)
    truecolor  24-bit RGB (modern terminals)
    halfblock  '▀' with fg=top / bg=bottom pixel -> double vertical resolution
The character is chosen by luminance, so structure reads the same in every mode.

Playback is interactive in a tty: space pauses, q/Esc quits, left/right seek 5s.
The window can be resized mid-play, --loop repeats, --start jumps to a timestamp,
and frames are dropped (not slow-mo'd) when the terminal can't keep up.

Audio is muted by default. Pass --audio to play sound through a parallel ffplay
chain: a separate bestaudio stream for URLs, or the file's own track for local
files. Local file paths skip yt-dlp and are fed straight to ffmpeg.

Dependencies: yt-dlp, ffmpeg (+ ffprobe), numpy; ffplay only when --audio is used.
"""

from __future__ import annotations

import argparse
import os
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np

try:
    import termios
    import tty
    HAVE_TERMIOS = True
except ImportError:  # non-POSIX
    HAVE_TERMIOS = False

# Dark -> light. Plain ASCII
DEFAULT_RAMP = " .:-=+*#%@"
LONG_RAMP = (
    " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao"
    "*#MW&8%B@$"
)
HALF_BLOCK = "▀"  # upper half block

SEEK_STEP = 5.0  # seconds per left/right keypress

# 4x4 Bayer matrix for ordered dithering (stateless -> safe for video).
BAYER4 = np.array(
    [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]],
    dtype=np.float32,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="ascii-tube",
        description="Render a YouTube or local video as ASCII in the terminal.",
    )
    p.add_argument("input",
                   help="YouTube URL (any yt-dlp-supported URL) or a path to a local video file.")
    p.add_argument("-w", "--width", type=int, default=0,
                   help="Output width in characters (default: terminal width).")
    p.add_argument("--fps", type=float, default=0.0,
                   help="Playback frame rate (default: source fps, capped at --max-fps).")
    p.add_argument("--max-fps", type=float, default=30.0,
                   help="Upper bound on fps when auto-detecting (default: 30).")
    p.add_argument("--max-height", type=int, default=240,
                   help="Max source height to request from yt-dlp (default: 240).")
    p.add_argument("--chars", default=DEFAULT_RAMP,
                   help="Character ramp, dark to light.")
    p.add_argument("--long", action="store_true",
                   help="Use a long 70-level ramp for finer gradation.")
    p.add_argument("--invert", action="store_true",
                   help="Invert brightness (for light-background terminals).")
    p.add_argument("--color", "--colour", dest="color", default="none",
                   choices=["none", "256", "truecolor", "halfblock"],
                   help="Colour mode: none (monochrome, default), 256, truecolor, halfblock.")
    p.add_argument("--dither", action="store_true",
                   help="Ordered (Bayer) dithering to reduce banding in gradients.")
    p.add_argument("--brightness", type=float, default=0.0,
                   help="ffmpeg eq brightness, -1..1 (default: 0).")
    p.add_argument("--contrast", type=float, default=1.0,
                   help="ffmpeg eq contrast (default: 1).")
    p.add_argument("--gamma", type=float, default=1.0,
                   help="ffmpeg eq gamma (default: 1).")
    p.add_argument("--diff", action="store_true",
                   help="Redraw only changed cells (helps for mostly-static content).")
    p.add_argument("--start", default="0",
                   help="Start position as seconds or [hh:]mm:ss.")
    p.add_argument("--loop", action="store_true",
                   help="Restart from the beginning when playback ends.")
    p.add_argument("--audio", action="store_true",
                   help="Play audio via a parallel ffplay chain (default: muted).")
    p.add_argument("--audio-format", default="ba/bestaudio/b",
                   help="yt-dlp format selector for the audio stream (URLs only).")
    p.add_argument("--char-aspect", type=float, default=0.5,
                   help="Cell width/height correction; lower = less vertical squash (default: 0.5).")
    p.add_argument("--frames", type=int, default=0,
                   help="Stop after N frames (0 = play to end). Use 1 for a single still.")
    p.add_argument("--format", default=None,
                   help="Override the yt-dlp format selector.")
    return p.parse_args(argv)


def have(cmd):
    return shutil.which(cmd) is not None


def parse_time(s):
    """Seconds from a number or a [hh:]mm:ss string. 0.0 on empty/garbage."""
    s = (s or "").strip()
    if not s:
        return 0.0
    try:
        if ":" in s:
            sec = 0.0
            for part in s.split(":"):
                sec = sec * 60 + float(part or 0)
            return sec
        return float(s)
    except ValueError:
        return 0.0


def build_eq(args):
    """ffmpeg eq filter string for brightness/contrast/gamma, or '' if all default."""
    terms = []
    if args.brightness:
        terms.append(f"brightness={args.brightness}")
    if args.contrast != 1.0:
        terms.append(f"contrast={args.contrast}")
    if args.gamma != 1.0:
        terms.append(f"gamma={args.gamma}")
    return "eq=" + ":".join(terms) if terms else ""


def make_dither(rows, cols):
    """A (rows, cols) ordered-dither offset map tiled from BAYER4, in [-0.5, 0.5)."""
    base = (BAYER4 + 0.5) / 16.0 - 0.5
    reps = ((rows + 3) // 4, (cols + 3) // 4)
    return np.tile(base, reps)[:rows, :cols]


def probe_url(url, fmt):
    """(width, height, fps) for a remote URL via yt-dlp metadata. Best-effort."""
    try:
        out = subprocess.run(
            ["yt-dlp", "-f", fmt, "--no-warnings", "--skip-download",
             "--print", "%(width)s %(height)s %(fps)s", url],
            capture_output=True, text=True, timeout=60,
        )
        w, h, f = out.stdout.strip().splitlines()[-1].split()
        return int(w), int(h), float(f)
    except Exception:
        return None


def probe_file(path):
    """(width, height, fps) for a local file via ffprobe. Best-effort."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,r_frame_rate",
             "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=30,
        )
        w, h, rate = out.stdout.strip().split(",")
        num, den = (rate.split("/") + ["1"])[:2]
        fps = float(num) / float(den) if float(den) else float(num)
        return int(w), int(h), fps
    except Exception:
        return None


def compute_dims(src_w, src_h, width_override, char_aspect):
    """Pick character grid (cols, rows) that preserves aspect and fits the screen."""
    term = shutil.get_terminal_size(fallback=(80, 24))
    aspect = (src_h / src_w) if (src_w and src_h) else (9 / 16)
    if width_override:
        cols = max(1, width_override)
        rows = max(1, round(cols * aspect * char_aspect))
        return cols, rows
    cols = max(1, term.columns)
    rows = max(1, round(cols * aspect * char_aspect))
    max_rows = max(1, term.lines - 1)
    if rows > max_rows:
        rows = max_rows
        cols = min(term.columns, max(1, round(rows / (aspect * char_aspect))))
    return cols, rows


def start_stream(input_, fmt, cols, decode_rows, fps, is_file, pix_fmt, position, eq, errf):
    """Spawn the (yt-dlp ->) ffmpeg pipeline. Returns (ytdlp_proc_or_None, ffmpeg_proc).

    pix_fmt is "gray" (1 byte/pixel) or "rgb24" (3 bytes/pixel). position seeks
    via input -ss (fast for files; decode-and-discard for piped URLs). eq is an
    optional ffmpeg filter; errf receives both processes' stderr for diagnostics.
    """
    vf_parts = [f"fps={fps:.6f}", f"scale={cols}:{decode_rows}:flags=area"]
    if eq:
        vf_parts.append(eq)
    vf_parts.append(f"format={pix_fmt}")
    vf = ",".join(vf_parts)

    ff_cmd = ["ffmpeg", "-loglevel", "error"]
    if position > 0:
        ff_cmd += ["-ss", f"{position:.3f}"]
    ytdlp = None
    if is_file:
        ff_cmd += ["-i", input_]
    else:
        ff_cmd += ["-i", "pipe:0"]
        ytdlp = subprocess.Popen(
            ["yt-dlp", "-q", "--no-warnings", "-f", fmt, "-o", "-", input_],
            stdout=subprocess.PIPE, stderr=errf,
        )
    ff_cmd += ["-an", "-vf", vf, "-pix_fmt", pix_fmt, "-f", "rawvideo", "pipe:1"]
    ffmpeg = subprocess.Popen(
        ff_cmd,
        stdin=(ytdlp.stdout if ytdlp else None),
        stdout=subprocess.PIPE, stderr=errf,
    )
    if ytdlp:
        ytdlp.stdout.close()  # let yt-dlp receive SIGPIPE if ffmpeg dies
    return ytdlp, ffmpeg


def start_audio(input_, fmt_audio, is_file, position=0.0):
    """Spawn a parallel audio-only playback chain. Returns procs (consumer first).

    Runs independently of the video pipeline: ffplay owns its own audio device and
    plays at true real-time rate, so it stays anchored to the wall clock just like
    the video loop does. stdin is detached so ffplay never grabs terminal keys.
    """
    base = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error"]
    if position > 0:
        base += ["-ss", f"{position:.3f}"]
    if is_file:
        player = subprocess.Popen(
            base + ["-vn", input_],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return [player]
    ytdlp_a = subprocess.Popen(
        ["yt-dlp", "-q", "--no-warnings", "-f", fmt_audio, "-o", "-", input_],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    player = subprocess.Popen(
        base + ["-i", "pipe:0"],
        stdin=ytdlp_a.stdout,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    ytdlp_a.stdout.close()  # let yt-dlp receive SIGPIPE if ffplay dies
    return [player, ytdlp_a]


def read_exact(stream, n):
    """Read exactly n bytes (pipe reads can be partial). None on EOF."""
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def _cadd(*parts):
    """Vectorised string concat: np.char.add folded over scalars and arrays."""
    acc = parts[0]
    for p in parts[1:]:
        acc = np.char.add(acc, p)
    return acc


def _rgb_to_256(r, g, b, lum):
    """Map RGB channels (uint16 arrays, 0..255) to xterm-256 palette indices.

    Near-neutral pixels use the 24-step grayscale ramp (232..255) for smoother
    darks; everything else snaps to the 6x6x6 colour cube (16..231).
    """
    cube = (16 + 36 * ((r * 5 + 127) // 255)
            + 6 * ((g * 5 + 127) // 255)
            + ((b * 5 + 127) // 255))
    spread = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    gray = 232 + np.clip((lum.astype(np.int16) - 8) // 10, 0, 23)
    return np.where(spread < 16, gray, cube).astype(np.uint16)


def quantize(vals, nlevels, dither):
    """Map luminance (0..255) to ramp indices (0..nlevels), optionally dithered."""
    if dither is None:
        return (vals.astype(np.uint32) * nlevels + 127) // 255
    t = vals.astype(np.float32) * (nlevels / 255.0) + dither
    return np.clip(np.rint(t), 0, nlevels).astype(np.intp)


def _color_sgr(r, g, b, lum, color):
    """Per-cell SGR foreground escape ("\\033[...m") for the non-halfblock modes."""
    if color == "truecolor":
        return _cadd("\033[38;2;", np.char.mod("%d", r), ";",
                     np.char.mod("%d", g), ";", np.char.mod("%d", b), "m")
    code = _rgb_to_256(r, g, b, lum)
    return _cadd("\033[38;5;", np.char.mod("%d", code), "m")


def _halfblock_sgr(top, bot):
    """Per-cell SGR with fg=top pixel and bg=bottom pixel (truecolor)."""
    def ch(a, i):
        return np.char.mod("%d", a[..., i])
    return _cadd("\033[38;2;", ch(top, 0), ";", ch(top, 1), ";", ch(top, 2),
                 ";48;2;", ch(bot, 0), ";", ch(bot, 1), ";", ch(bot, 2), "m")


def build_cells(raw, rows, cols, ramp_arr, nlevels, color, dither):
    """Turn one raw frame into (chars, sgr): two (rows, cols) str arrays.

    sgr is None in monochrome (no per-cell colour); otherwise each entry is the
    full SGR escape that colours that cell.
    """
    if color == "none":
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(rows, cols)
        return ramp_arr[quantize(arr, nlevels, dither)], None

    if color == "halfblock":
        px = np.frombuffer(raw, dtype=np.uint8).reshape(rows * 2, cols, 3).astype(np.uint16)
        chars = np.full((rows, cols), HALF_BLOCK)
        return chars, _halfblock_sgr(px[0::2], px[1::2])

    px = np.frombuffer(raw, dtype=np.uint8).reshape(rows, cols, 3).astype(np.uint16)
    r, g, b = px[..., 0], px[..., 1], px[..., 2]
    lum = (r * 77 + g * 150 + b * 29) >> 8  # ~Rec.601 luminance, 0..255
    chars = ramp_arr[quantize(lum, nlevels, dither)]
    return chars, _color_sgr(r, g, b, lum, color)


def _full_text(chars, sgr):
    """Full-frame text. In colour, drop a cell's escape when it matches the
    previous cell (row-major run-length) and append a reset."""
    if sgr is None:
        return "\n".join("".join(row) for row in chars)
    flat = sgr.reshape(-1)
    keep = np.ones(flat.shape, dtype=bool)
    keep[1:] = flat[1:] != flat[:-1]
    body = np.char.add(np.where(keep.reshape(sgr.shape), sgr, ""), chars)
    return "\n".join("".join(row) for row in body) + "\033[0m"


def render_diff(out, cur, prev):
    """Repaint only the cells that differ from the previous frame. Returns cur."""
    mask = cur != prev
    parts = []
    for r in np.flatnonzero(mask.any(axis=1)):
        changed = np.flatnonzero(mask[r])
        for run in np.split(changed, np.flatnonzero(np.diff(changed) > 1) + 1):
            c0, c1 = int(run[0]), int(run[-1])
            parts.append(f"\033[{r + 1};{c0 + 1}H" + "".join(cur[r, c0:c1 + 1]))
    if parts:
        parts.append("\033[0m")
        out.write("".join(parts))
        out.flush()
    return cur


def emit(ctx, chars, sgr, prev):
    """Write one frame; returns the self-contained cell grid when diffing else None."""
    out = ctx.out
    self_cells = chars if sgr is None else np.char.add(sgr, chars)
    if ctx.diff and prev is not None and prev.shape == self_cells.shape:
        return render_diff(out, self_cells, prev)
    text = _full_text(chars, sgr)
    out.write(("\033[H" + text) if ctx.interactive else (text + "\n\n"))
    out.flush()
    return self_cells if ctx.diff else None


def cleanup(procs):
    """Terminate every spawned process. Pass consumers before producers so the
    producers receive SIGPIPE as their pipes close."""
    for proc in procs:
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()


def _read_key(stdin_tty):
    """Non-blocking read of one keypress (raw bytes), or None."""
    if not stdin_tty:
        return None
    if select.select([sys.stdin], [], [], 0)[0]:
        try:
            return os.read(sys.stdin.fileno(), 3)
        except OSError:
            return None
    return None


def run_decode(ctx, ffmpeg, position0):
    """Drive one stream until it stops. Returns (reason, position_seconds) where
    reason is "eof", "quit", "seek", or "resize"."""
    fps = ctx.fps
    start = None
    prev = None
    paused = False
    pause_t = 0.0
    n = 0  # frames read this session
    while True:
        if ctx.resize_flag[0]:
            ctx.resize_flag[0] = False
            return "resize", position0 + n / fps

        key = _read_key(ctx.stdin_tty)
        if key:
            if key in (b"q", b"\x1b"):
                return "quit", position0 + n / fps
            if key == b" ":
                paused = not paused
                if paused:
                    pause_t = time.monotonic()
                elif start is not None:
                    start += time.monotonic() - pause_t
            elif key == b"\x1b[C":
                return "seek", position0 + n / fps + SEEK_STEP
            elif key == b"\x1b[D":
                return "seek", max(0.0, position0 + n / fps - SEEK_STEP)

        if paused:
            time.sleep(0.03)
            continue

        raw = read_exact(ffmpeg.stdout, ctx.frame_bytes)
        if raw is None:
            return "eof", position0 + n / fps

        now = time.monotonic()
        if start is None:
            start = now  # anchor time-zero to the first decoded frame
        pts = start + n / fps
        behind = now - pts

        if ctx.interactive and behind > 1.0 / fps:
            n += 1            # too late: drop the render, just keep draining
            ctx.dropped += 1
            continue
        if behind < 0:
            time.sleep(-behind)

        chars, sgr = build_cells(raw, ctx.rows, ctx.cols, ctx.ramp,
                                 ctx.nlevels, ctx.color, ctx.dither)
        prev = emit(ctx, chars, sgr, prev)
        ctx.rendered += 1
        n += 1
        if ctx.frames and n >= ctx.frames:
            return "quit", position0 + n / fps


def play(args):
    is_file = os.path.exists(args.input)

    ramp = LONG_RAMP if args.long else args.chars
    if args.invert:
        ramp = ramp[::-1]
    ramp_arr = np.array(list(ramp))
    nlevels = len(ramp_arr) - 1
    if nlevels < 1:
        sys.exit("Character ramp must contain at least 2 characters.")

    color = args.color
    channels = 1 if color == "none" else 3
    pix_fmt = "gray" if color == "none" else "rgb24"

    fmt = args.format or (
        f"bv*[height<={args.max_height}]/b[height<={args.max_height}]/wv*/worst"
    )
    meta = probe_file(args.input) if is_file else probe_url(args.input, fmt)
    src_w, src_h, src_fps = meta if meta else (0, 0, 0.0)
    fps = args.fps or (min(src_fps, args.max_fps) if src_fps else args.max_fps)
    if fps <= 0:
        fps = 24.0
    eq = build_eq(args)

    out = sys.stdout
    interactive = out.isatty()
    stdin_tty = interactive and HAVE_TERMIOS and sys.stdin.isatty()
    old_term = None
    if stdin_tty:
        try:
            old_term = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())  # keeps Ctrl-C working
        except Exception:
            stdin_tty = False

    resize_flag = [False]
    if interactive and hasattr(signal, "SIGWINCH"):
        signal.signal(signal.SIGWINCH, lambda *a: resize_flag.__setitem__(0, True))
    if interactive:
        out.write("\033[?25l")  # hide cursor
        out.flush()

    ctx = SimpleNamespace(
        fps=fps, ramp=ramp_arr, nlevels=nlevels, color=color,
        interactive=interactive, stdin_tty=stdin_tty, diff=args.diff and interactive,
        frames=args.frames, out=out, resize_flag=resize_flag,
        rendered=0, dropped=0, rows=0, cols=0, frame_bytes=0, dither=None,
    )

    errf = tempfile.TemporaryFile()
    position = parse_time(args.start)
    last_reason = "eof"
    try:
        while True:
            cols, rows = compute_dims(src_w, src_h, args.width, args.char_aspect)
            decode_rows = rows * 2 if color == "halfblock" else rows
            ctx.cols, ctx.rows = cols, rows
            ctx.frame_bytes = cols * decode_rows * channels
            ctx.dither = (make_dither(rows, cols)
                          if args.dither and color != "halfblock" else None)

            ytdlp, ffmpeg = start_stream(args.input, fmt, cols, decode_rows, fps,
                                         is_file, pix_fmt, position, eq, errf)
            audio = start_audio(args.input, args.audio_format, is_file, position) \
                if args.audio else []
            if interactive:
                out.write("\033[2J\033[H")  # clear for the fresh session
                out.flush()

            last_reason, position = run_decode(ctx, ffmpeg, position)
            cleanup([ffmpeg, ytdlp, *audio])

            if last_reason == "quit":
                break
            if last_reason == "eof":
                if args.loop:
                    position = 0.0
                    continue
                break
            # "seek" / "resize": respawn at the (possibly new) position
    except KeyboardInterrupt:
        pass
    finally:
        if stdin_tty and old_term is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term)
        if interactive and hasattr(signal, "SIGWINCH"):
            signal.signal(signal.SIGWINCH, signal.SIG_DFL)
        if interactive:
            out.write("\033[?25h\n")  # show cursor
            out.flush()

    if ctx.rendered == 0 and last_reason == "eof":
        errf.seek(0)
        err = errf.read().decode("utf-8", "replace").strip()
        errf.close()
        src = "the local file" if is_file else "yt-dlp/ffmpeg (check the URL and format)"
        msg = f"\nNo frames decoded from {src}."
        if err:
            msg += "\n--- ffmpeg/yt-dlp stderr ---\n" + "\n".join(err.splitlines()[-15:])
        sys.exit(msg)
    errf.close()
    return ctx.rendered


def main():
    args = parse_args()
    needed = ["ffmpeg"] + ([] if os.path.exists(args.input) else ["yt-dlp"])
    if args.audio:
        needed.append("ffplay")
    missing = [t for t in needed if not have(t)]
    if missing:
        sys.exit("Required tool(s) not found: " + ", ".join(missing))
    play(args)


if __name__ == "__main__":
    main()
