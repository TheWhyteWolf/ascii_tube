# ascii_tube

Render a YouTube or local video as ASCII in the terminal — monochrome by
default, with optional 256-colour and truecolor modes.

```
ascii_tube.py <URL|file> [options]
```

## Colour

Output is monochrome by default. Add colour with:

```
--color 256          # xterm 256-colour palette (broad terminal support)
--color truecolor    # 24-bit RGB (modern terminals; --colour also accepted)
```

The character is always chosen by the pixel's luminance, so the picture's
structure reads identically in every mode — colour just tints each cell. Escape
codes are emitted only when the colour changes from the previous cell, so flat
regions stay cheap to draw.

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
