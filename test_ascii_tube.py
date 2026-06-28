"""Unit tests for ascii_tube's pure rendering / mapping helpers.

Run with `pytest`, or directly: `python test_ascii_tube.py`.
"""
import io
import re

import numpy as np

import ascii_tube as a

RAMP = np.array(list(a.DEFAULT_RAMP))
NLEV = len(RAMP) - 1

# black, white, red / green, blue, mid-gray
FRAME = np.array([
    [[0, 0, 0], [255, 255, 255], [255, 0, 0]],
    [[0, 255, 0], [0, 0, 255], [128, 128, 128]],
], dtype=np.uint8)


def strip(s):
    return re.sub("\x1b\\[[0-9;]*m", "", s)


def test_parse_time():
    assert a.parse_time("") == 0.0
    assert a.parse_time("12") == 12.0
    assert a.parse_time("1:30") == 90.0
    assert a.parse_time("1:00:00") == 3600.0
    assert a.parse_time("garbage") == 0.0


def test_build_eq():
    assert a.build_eq(a.SimpleNamespace(brightness=0.0, contrast=1.0, gamma=1.0)) == ""
    eq = a.build_eq(a.SimpleNamespace(brightness=0.1, contrast=1.2, gamma=0.9))
    assert eq == "eq=brightness=0.1:contrast=1.2:gamma=0.9"


def test_quantize_matches_legacy_without_dither():
    vals = np.arange(256, dtype=np.uint8)
    got = a.quantize(vals, NLEV, None)
    want = (vals.astype(np.uint32) * NLEV + 127) // 255
    assert np.array_equal(got, want)


def test_quantize_dither_stays_in_range():
    vals = np.arange(256, dtype=np.uint8).reshape(16, 16)
    idx = a.quantize(vals, NLEV, a.make_dither(16, 16))
    assert idx.min() >= 0 and idx.max() <= NLEV


def test_make_dither_shape_and_range():
    d = a.make_dither(5, 7)
    assert d.shape == (5, 7)
    assert d.min() >= -0.5 and d.max() < 0.5


def test_rgb_to_256_known_values():
    def code(r, g, b):
        arr = lambda v: np.array([[v]], dtype=np.uint16)
        lum = (arr(r) * 77 + arr(g) * 150 + arr(b) * 29) >> 8
        return int(a._rgb_to_256(arr(r), arr(g), arr(b), lum)[0, 0])
    assert code(255, 0, 0) == 196       # pure red -> cube
    assert code(0, 255, 0) == 46        # pure green -> cube
    assert code(0, 0, 255) == 21        # pure blue -> cube
    # neutrals (zero spread) use the 24-step gray ramp, not the cube
    assert code(0, 0, 0) == 232         # darkest gray
    assert 232 <= code(255, 255, 255) <= 255


def test_mono_render_unchanged():
    lum = (FRAME.astype(np.uint16) @ np.array([77, 150, 29]) >> 8).astype(np.uint8)
    chars, sgr = a.build_cells(lum.tobytes(), 2, 3, RAMP, NLEV, "none", None)
    assert sgr is None
    legacy = "\n".join("".join(r) for r in RAMP[(lum.astype(np.uint16) * NLEV + 127) // 255])
    assert a._full_text(chars, sgr) == legacy


def test_color_preserves_characters():
    raw = FRAME.tobytes()
    lum = (FRAME.astype(np.uint16) @ np.array([77, 150, 29]) >> 8).astype(np.uint8)
    mono = "\n".join("".join(r) for r in RAMP[(lum.astype(np.uint16) * NLEV + 127) // 255])
    for mode in ("256", "truecolor"):
        chars, sgr = a.build_cells(raw, 2, 3, RAMP, NLEV, mode, None)
        assert strip(a._full_text(chars, sgr)) == mono


def test_truecolor_escapes_present():
    chars, sgr = a.build_cells(FRAME.tobytes(), 2, 3, RAMP, NLEV, "truecolor", None)
    text = a._full_text(chars, sgr)
    assert "\x1b[38;2;255;255;255m" in text
    assert "\x1b[38;2;255;0;0m" in text
    assert text.endswith("\x1b[0m")


def test_halfblock_packs_two_rows():
    # 4 source rows -> 2 character rows; fg=top pixel, bg=bottom pixel.
    px = np.zeros((4, 3, 3), dtype=np.uint8)
    px[0] = [255, 0, 0]    # top of cell-row 0
    px[1] = [0, 255, 0]    # bottom of cell-row 0
    chars, sgr = a.build_cells(px.tobytes(), 2, 3, RAMP, NLEV, "halfblock", None)
    assert chars.shape == (2, 3) and chars[0, 0] == a.HALF_BLOCK
    assert sgr[0, 0] == "\x1b[38;2;255;0;0;48;2;0;255;0m"


def test_full_text_color_runlength():
    # a row of identical colour should emit the escape once, not per cell.
    px = np.zeros((1, 4, 3), dtype=np.uint8)
    px[:] = [10, 20, 30]
    chars, sgr = a.build_cells(px.tobytes(), 1, 4, RAMP, NLEV, "truecolor", None)
    assert a._full_text(chars, sgr).count("\x1b[38;2;10;20;30m") == 1


def test_render_diff_only_repaints_changes():
    a_cells = np.array([["x", "x", "x"], ["x", "x", "x"]])
    b_cells = a_cells.copy()
    b_cells[1, 2] = "y"
    out = io.StringIO()
    a.render_diff(out, b_cells, a_cells)
    s = out.getvalue()
    assert "\x1b[2;3H" in s and "y" in s and "x" not in strip(s).replace("\x1b", "")
    # no changes -> nothing written
    out2 = io.StringIO()
    a.render_diff(out2, a_cells, a_cells)
    assert out2.getvalue() == ""


def test_read_exact_handles_partial_reads():
    class Drip:
        def __init__(self, data):
            self.data, self.i = data, 0
        def read(self, n):
            chunk = self.data[self.i:self.i + 1]  # one byte at a time
            self.i += 1
            return chunk
    assert a.read_exact(Drip(b"abcd"), 4) == b"abcd"
    assert a.read_exact(Drip(b"ab"), 4) is None  # EOF before n


def test_compute_dims_preserves_aspect():
    cols, rows = a.compute_dims(1600, 900, 100, 0.5)
    assert cols == 100
    assert rows == round(100 * (900 / 1600) * 0.5)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print(f"\n{len(fns)} tests passed")
