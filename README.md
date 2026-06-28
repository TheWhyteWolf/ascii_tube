# ascii_tube

Render a YouTube or local video as ASCII in the terminal — monochrome by
default, with optional colour modes, interactive playback controls, and audio.

```
ascii_tube.py <URL|file> [options]
```

Install as a command (`ascii-tube`) with `pip install .` (or `pipx install .`).

## Colour

Output is monochrome by default. Add colour with:

```
--color 256          # xterm 256-colour palette (broad terminal support)
--color truecolor    # 24-bit RGB (modern terminals; --colour also accepted)
--color halfblock    # '▀' with fg=top / bg=bottom pixel -> 2x vertical detail
```

In the ramp modes (`256`/`truecolor`) the character is chosen by the pixel's
luminance, so the picture's structure reads identically in every mode — colour
just tints each cell. Escape codes are emitted only when the colour changes from
the previous cell, so flat regions stay cheap to draw. `halfblock` packs two
vertical pixels into every cell (foreground + background) for the highest
fidelity; it uses 24-bit colour.

## Picture tuning

```
--dither             # ordered (Bayer) dithering to reduce gradient banding
--brightness N       # ffmpeg eq brightness, -1..1 (default 0)
--contrast N         # ffmpeg eq contrast (default 1)
--gamma N            # ffmpeg eq gamma (default 1)
--invert             # invert brightness (light-background terminals)
--long               # 70-level character ramp for finer gradation
```

## Playback

In a terminal, playback is interactive:

| key            | action            |
|----------------|-------------------|
| `space`        | pause / resume    |
| `q` / `Esc`    | quit              |
| `←` / `→`      | seek ∓5 seconds   |

The window may be resized mid-play (the grid re-fits automatically). When the
terminal can't keep up, late frames are **dropped** rather than played in slow
motion, so video stays in sync with audio and wall-clock time.

```
--start [hh:]mm:ss   # begin at a timestamp (also accepts plain seconds)
--loop               # restart from the beginning when playback ends
--frames N           # stop after N frames (1 = a single still)
--diff               # redraw only changed cells (good for mostly-static content)
```

## Audio

Audio is muted by default. Pass `--audio` to play sound through a parallel
`ffplay` chain — a separate `bestaudio` stream for URLs, or the file's own
audio track for local files. Override the URL audio stream selector with
`--audio-format` (default `ba/bestaudio/b`).

Audio plays in its own process and tracks the wall clock independently of the
video loop, so the two stay within a roughly fixed offset rather than drifting
apart over time.

## Dependencies

yt-dlp, ffmpeg (+ ffprobe), numpy; `ffplay` is additionally required only when
`--audio` is used.

## Tests

```
pytest                 # or: python test_ascii_tube.py
```
