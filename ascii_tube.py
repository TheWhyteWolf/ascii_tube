#!/usr/bin/env python3
"""

Pipeline:
    yt-dlp (fetch <=240p stream)  ->  ffmpeg (decode / scale / gray or rgb24)
    ->  numpy (luminance -> char ramp, + optional ANSI colour)  ->  terminal.

Output is monochrome by default. Pass --color 256 or --color truecolor to tint
each cell with the source pixel's colour; the character is still chosen by
luminance so the picture's structure reads the same in every mode.

Audio is muted by default (240p YouTube is video-only anyway). Pass --audio to
play sound through a parallel ffplay chain: a separate bestaudio stream for URLs,
or the file's own audio track for local files. Local file paths skip yt-dlp and
are fed straight to ffmpeg, so this doubles as a local player.

Dependencies: yt-dlp, ffmpeg (+ ffprobe), numpy; ffplay only when --audio is used.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time

import numpy as np

# Dark -> light. Plain ASCII
DEFAULT_RAMP = " .:-=+*#%@"
LONG_RAMP = (
    " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao"
    "*#MW&8%B@$"
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="ascii-tube",
        description="Render a YouTube or local video as monochrome ASCII in the terminal.",
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
                   choices=["none", "256", "truecolor"],
                   help="Colour mode: none (monochrome, default), 256, or truecolor.")
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


def start_stream(input_, fmt, cols, rows, fps, is_file, pix_fmt):
    """Spawn the (yt-dlp ->) ffmpeg pipeline. Returns (ytdlp_proc_or_None, ffmpeg_proc).

    pix_fmt is "gray" (1 byte/pixel) for monochrome or "rgb24" (3 bytes/pixel)
    for the colour modes.
    """
    vf = f"fps={fps:.6f},scale={cols}:{rows}:flags=area,format={pix_fmt}"
    ff_cmd = ["ffmpeg", "-loglevel", "error"]
    ytdlp = None
    if is_file:
        ff_cmd += ["-i", input_]
    else:
        ff_cmd += ["-i", "pipe:0"]
        ytdlp = subprocess.Popen(
            ["yt-dlp", "-q", "--no-warnings", "-f", fmt, "-o", "-", input_],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
    ff_cmd += ["-an", "-vf", vf, "-pix_fmt", pix_fmt, "-f", "rawvideo", "pipe:1"]
    ffmpeg = subprocess.Popen(
        ff_cmd,
        stdin=(ytdlp.stdout if ytdlp else None),
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    if ytdlp:
        ytdlp.stdout.close()  # let yt-dlp receive SIGPIPE if ffmpeg dies
    return ytdlp, ffmpeg


def start_audio(input_, fmt_audio, is_file):
    """Spawn a parallel audio-only playback chain. Returns procs (consumer first).

    Runs independently of the video pipeline: ffplay owns its own audio device and
    plays at true real-time rate, so it stays anchored to the wall clock just like
    the video loop does. stdin is detached so ffplay never grabs terminal keys.
    """
    base = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error"]
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


def _color_escapes(rgb, lum, mode):
    """Per-cell SGR escape strings, emitting one only where the colour changes
    from the previous cell in row-major order (run-length to spare the terminal).
    Returns a (rows, cols) array of "" or "\\033[...m"."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    if mode == "truecolor":
        body = np.char.add(np.char.add(np.char.add(np.char.add(np.char.add(
            "38;2;", np.char.mod("%d", r)), ";"), np.char.mod("%d", g)), ";"),
            np.char.mod("%d", b))
        key = (r.astype(np.uint32) << 16) | (g.astype(np.uint32) << 8) | b
    else:  # "256"
        code = _rgb_to_256(r, g, b, lum)
        body = np.char.add("38;5;", np.char.mod("%d", code))
        key = code

    flat = key.ravel()
    changed = np.empty(flat.shape, dtype=bool)
    changed[0] = True
    changed[1:] = flat[1:] != flat[:-1]
    esc = np.char.add(np.char.add("\033[", body), "m")
    return np.where(changed.reshape(key.shape), esc, "")


def render(raw, rows, cols, ramp_arr, nlevels, color):
    """Turn one raw frame into terminal text (with trailing reset in colour modes)."""
    if color == "none":
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(rows, cols)
        idx = (arr.astype(np.uint16) * nlevels + 127) // 255
        return "\n".join("".join(row) for row in ramp_arr[idx])

    rgb = np.frombuffer(raw, dtype=np.uint8).reshape(rows, cols, 3).astype(np.uint16)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    lum = (r * 77 + g * 150 + b * 29) >> 8  # ~Rec.601 luminance, 0..255
    chars = ramp_arr[(lum * nlevels + 127) // 255]
    cells = np.char.add(_color_escapes(rgb, lum, color), chars)
    return "\n".join("".join(row) for row in cells) + "\033[0m"


def cleanup(procs, interactive, out):
    """Terminate every spawned process. Pass consumers before producers so the
    producers receive SIGPIPE as their pipes close."""
    for proc in procs:
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
    if interactive:
        out.write("\033[?25h\n")  # show cursor again
        out.flush()


def play(args):
    is_file = os.path.exists(args.input)

    ramp = LONG_RAMP if args.long else args.chars
    if args.invert:
        ramp = ramp[::-1]
    ramp_arr = np.array(list(ramp))
    nlevels = len(ramp_arr) - 1
    if nlevels < 1:
        sys.exit("Character ramp must contain at least 2 characters.")

    fmt = args.format or (
        f"bv*[height<={args.max_height}]/b[height<={args.max_height}]/wv*/worst"
    )

    meta = probe_file(args.input) if is_file else probe_url(args.input, fmt)
    src_w, src_h, src_fps = meta if meta else (0, 0, 0.0)

    fps = args.fps or (min(src_fps, args.max_fps) if src_fps else args.max_fps)
    if fps <= 0:
        fps = 24.0

    cols, rows = compute_dims(src_w, src_h, args.width, args.char_aspect)
    pix_fmt = "gray" if args.color == "none" else "rgb24"
    channels = 1 if args.color == "none" else 3
    frame_bytes = cols * rows * channels

    ytdlp, ffmpeg = start_stream(args.input, fmt, cols, rows, fps, is_file, pix_fmt)
    audio_procs = start_audio(args.input, args.audio_format, is_file) if args.audio else []

    out = sys.stdout
    interactive = out.isatty()
    if interactive:
        out.write("\033[2J\033[?25l")  # clear screen, hide cursor
        out.flush()

    start = None
    n = 0
    try:
        while True:
            raw = read_exact(ffmpeg.stdout, frame_bytes)
            if raw is None:
                break
            if start is None:
                start = time.monotonic()  # anchor time-zero to the first decoded frame
            text = render(raw, rows, cols, ramp_arr, nlevels, args.color)

            out.write("\033[H" + text if interactive else text + "\n\n")
            out.flush()

            n += 1
            if args.frames and n >= args.frames:
                break

            target = start + n / fps
            delay = target - time.monotonic()
            if delay > 0:
                time.sleep(delay)
    except KeyboardInterrupt:
        pass
    finally:
        cleanup([ffmpeg, ytdlp, *audio_procs], interactive, out)

    if ffmpeg.returncode not in (0, None) and n == 0:
        src = "the local file" if is_file else "yt-dlp/ffmpeg (check the URL and format)"
        sys.exit(f"\nNo frames decoded from {src}.")
    return n


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
