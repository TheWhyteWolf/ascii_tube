#!/usr/bin/env python3
"""ASCII-Cam: render a live webcam feed as ASCII in the terminal.

Pipeline:
    ffmpeg (capture device -> decode / scale / colour-adjust)
    ->  numpy (luminance -> char ramp, + optional ANSI colour)  ->  terminal.

This is the live-capture sibling of ascii_tube.py. The rendering half (char
ramp, colour modes, dithering, half-block, diff redraw) is shared with that
module; only the source differs: instead of yt-dlp/a file, frames come straight
off a camera via ffmpeg's platform capture backend:

    Linux    -f v4l2        -i /dev/video0
    macOS    -f avfoundation -i 0
    Windows  -f dshow       -i video="<name>"

Because the feed is live there is no seeking, looping or start offset. To keep
latency low the loop always renders the newest available frame, dropping any
backlog. The image is mirrored by default (like a real mirror); toggle with 'm'.

Playback is interactive in a tty: space freezes, m mirrors, q/Esc quits. The
window can be resized mid-stream. Pass --test to use a synthetic pattern instead
of a camera (handy for trying it out with no webcam attached).

Dependencies: ffmpeg (+ ffprobe), numpy.
"""

from __future__ import annotations

import argparse
import glob
import os
import select
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

# Reuse the proven rendering + plumbing helpers from the player.
from ascii_tube import (
    DEFAULT_RAMP,
    LONG_RAMP,
    build_cells,
    build_eq,
    cleanup,
    compute_dims,
    emit,
    have,
    make_dither,
    read_exact,
    _read_key,
)


def default_device():
    """The platform's conventional first-camera identifier."""
    if sys.platform == "darwin":
        return "0"
    if sys.platform.startswith("win"):
        return "video=Integrated Camera"
    return "/dev/video0"


def capture_format():
    """ffmpeg input demuxer for this platform's webcam stack, or None."""
    if sys.platform.startswith("linux"):
        return "v4l2"
    if sys.platform == "darwin":
        return "avfoundation"
    if sys.platform.startswith("win"):
        return "dshow"
    return None


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="ascii-cam",
        description="Render a live webcam feed as ASCII in the terminal.",
    )
    p.add_argument("-d", "--device", default=None,
                   help="Camera device (default: %s on this platform)." % default_device())
    p.add_argument("--list", action="store_true",
                   help="List candidate capture devices and exit.")
    p.add_argument("--test", action="store_true",
                   help="Use a synthetic test pattern instead of a camera (no webcam needed).")
    p.add_argument("-w", "--width", type=int, default=0,
                   help="Output width in characters (default: terminal width).")
    p.add_argument("--fps", type=float, default=30.0,
                   help="Target capture/render frame rate (default: 30).")
    p.add_argument("--video-size", default=None,
                   help="Request a capture resolution, e.g. 640x480 (default: camera default).")
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
    p.add_argument("--no-mirror", dest="mirror", action="store_false",
                   help="Don't horizontally mirror the image (mirroring is on by default).")
    p.add_argument("--diff", action="store_true",
                   help="Redraw only changed cells (helps for mostly-static scenes).")
    p.add_argument("--char-aspect", type=float, default=0.5,
                   help="Cell width/height correction; lower = less vertical squash (default: 0.5).")
    p.add_argument("--frames", type=int, default=0,
                   help="Stop after N frames (0 = run until quit). Use 1 for a single snapshot.")
    return p.parse_args(argv)


def list_devices():
    """Best-effort enumeration of capture devices for the current platform."""
    if sys.platform.startswith("linux"):
        devs = sorted(glob.glob("/dev/video*"))
        return devs or ["(no /dev/video* devices found)"]
    fmt = capture_format()
    if fmt in ("avfoundation", "dshow"):
        # ffmpeg prints the device list to stderr when asked to list them.
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-f", fmt, "-list_devices", "true", "-i", ""],
            capture_output=True, text=True,
        )
        return [out.stderr.strip() or "(use the device names ffmpeg printed above)"]
    return ["(device listing not supported on this platform)"]


def probe_webcam(device, fmt, video_size):
    """(width, height, fps) for a capture device via ffprobe. Best-effort.

    A forced --video-size already pins the geometry, so trust it without opening
    the camera; otherwise ask ffprobe for the device's default format.
    """
    if video_size and "x" in video_size:
        try:
            w, h = (int(v) for v in video_size.lower().split("x", 1))
            return w, h, 0.0
        except ValueError:
            pass
    if fmt != "v4l2":  # ffprobe can't reliably introspect avfoundation/dshow
        return None
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-f", fmt,
             "-show_entries", "stream=width,height,r_frame_rate",
             "-of", "csv=p=0", device],
            capture_output=True, text=True, timeout=15,
        )
        w, h, rate = out.stdout.strip().split(",")
        num, den = (rate.split("/") + ["1"])[:2]
        fps = float(num) / float(den) if float(den) else float(num)
        return int(w), int(h), fps
    except Exception:
        return None


def build_input(args, fmt, fps):
    """ffmpeg input args for the capture device (or the --test synthetic source)."""
    if args.test:
        size = args.video_size or "640x480"
        return ["-f", "lavfi", "-i",
                f"testsrc2=size={size}:rate={fps:g}"]
    device = args.device or default_device()
    pre = ["-f", fmt]
    if fps > 0:
        pre += ["-framerate", f"{fps:g}"]
    if args.video_size:
        pre += ["-video_size", args.video_size]
    if fmt == "dshow":
        device = device if device.startswith("video=") else f"video={device}"
    return pre + ["-i", device]


def start_capture(args, fmt, cols, decode_rows, fps, pix_fmt, eq, errf):
    """Spawn the capture ffmpeg pipeline. Returns the ffmpeg process."""
    vf_parts = [f"scale={cols}:{decode_rows}:flags=area"]
    if eq:
        vf_parts.append(eq)
    vf_parts.append(f"format={pix_fmt}")
    vf = ",".join(vf_parts)

    ff_cmd = ["ffmpeg", "-loglevel", "error"]
    ff_cmd += build_input(args, fmt, fps)
    ff_cmd += ["-an", "-vf", vf, "-pix_fmt", pix_fmt, "-f", "rawvideo", "pipe:1"]
    return subprocess.Popen(ff_cmd, stdout=subprocess.PIPE, stderr=errf)


def read_latest(stream, frame_bytes, max_skip=8):
    """Read the newest available frame, discarding any backlog. None on EOF.

    A live feed buffers frames if we render slower than the camera produces;
    draining to the most recent frame keeps the picture in sync with reality
    rather than playing an ever-growing delay.
    """
    raw = read_exact(stream, frame_bytes)
    if raw is None:
        return None
    skipped = 0
    while skipped < max_skip and select.select([stream], [], [], 0)[0]:
        nxt = read_exact(stream, frame_bytes)
        if nxt is None:
            break
        raw = nxt
        skipped += 1
    return raw


def run_capture(ctx, ffmpeg):
    """Drive the live stream until it stops. Returns "eof", "quit", or "resize"."""
    fps = ctx.fps
    interval = 1.0 / fps if fps > 0 else 0.0
    prev = None
    paused = False
    next_t = time.monotonic()
    n = 0
    while True:
        if ctx.resize_flag[0]:
            ctx.resize_flag[0] = False
            return "resize"

        key = _read_key(ctx.stdin_tty)
        if key:
            if key in (b"q", b"\x1b"):
                return "quit"
            if key == b" ":
                paused = not paused
            elif key in (b"m", b"M"):
                ctx.mirror = not ctx.mirror

        if paused:
            time.sleep(0.03)
            continue

        raw = read_latest(ffmpeg.stdout, ctx.frame_bytes)
        if raw is None:
            return "eof"

        chars, sgr = build_cells(raw, ctx.rows, ctx.cols, ctx.ramp,
                                 ctx.nlevels, ctx.color, ctx.dither)
        if ctx.mirror:
            chars = chars[:, ::-1]
            if sgr is not None:
                sgr = sgr[:, ::-1]
        prev = emit(ctx, chars, sgr, prev)
        ctx.rendered += 1
        n += 1
        if ctx.frames and n >= ctx.frames:
            return "quit"

        if interval:
            next_t += interval
            now = time.monotonic()
            if now < next_t:
                time.sleep(next_t - now)
            else:
                next_t = now  # fell behind; don't accumulate a sleep debt


def play(args):
    fmt = capture_format()
    if not args.test and fmt is None:
        sys.exit("Webcam capture is not supported on this platform.")

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

    fps = args.fps if args.fps > 0 else 30.0
    meta = probe_webcam(args.device or default_device(), fmt, args.video_size) \
        if not args.test else None
    if args.video_size and "x" in args.video_size and meta is None:
        meta = probe_webcam(None, fmt, args.video_size)
    src_w, src_h, _ = meta if meta else (640, 480, 0.0)  # 4:3 fallback
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
        frames=args.frames, out=out, resize_flag=resize_flag, mirror=args.mirror,
        rendered=0, rows=0, cols=0, frame_bytes=0, dither=None,
    )

    errf = tempfile.TemporaryFile()
    last_reason = "eof"
    try:
        while True:
            cols, rows = compute_dims(src_w, src_h, args.width, args.char_aspect)
            decode_rows = rows * 2 if color == "halfblock" else rows
            ctx.cols, ctx.rows = cols, rows
            ctx.frame_bytes = cols * decode_rows * channels
            ctx.dither = (make_dither(rows, cols)
                          if args.dither and color != "halfblock" else None)

            ffmpeg = start_capture(args, fmt, cols, decode_rows, fps,
                                   pix_fmt, eq, errf)
            if interactive:
                out.write("\033[2J\033[H")  # clear for the fresh session
                out.flush()

            last_reason = run_capture(ctx, ffmpeg)
            ffmpeg.stdout.close()  # unblock a write-stalled ffmpeg (EPIPE) so it exits at once
            cleanup([ffmpeg])

            if last_reason in ("quit", "eof"):
                break
            # "resize": respawn at the new geometry
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

    if ctx.rendered == 0:
        errf.seek(0)
        err = errf.read().decode("utf-8", "replace").strip()
        errf.close()
        msg = "\nNo frames captured (is the camera connected and free?)."
        if err:
            msg += "\n--- ffmpeg stderr ---\n" + "\n".join(err.splitlines()[-15:])
        sys.exit(msg)
    errf.close()
    return ctx.rendered


def main():
    args = parse_args()
    if args.list:
        print("\n".join(list_devices()))
        return
    if not have("ffmpeg"):
        sys.exit("Required tool not found: ffmpeg")
    play(args)


if __name__ == "__main__":
    main()
