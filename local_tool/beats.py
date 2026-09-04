"""
Row-profile segmentation: the primary panel detector.

Webtoons are composed vertically, so nearly all of their structure shows
up in a per-row profile of the strip. Each row is classified as:

  WHITE_BLANK / BLACK_BLANK  gutter (nothing on the row)
  BORDER                     a dark full-width line - a panel frame edge
  TEXT                       mostly white with some ink and white margins:
                             dialogue/narration floating between panels
  ART                        everything else - drawing

Long blank runs split the strip into segments. Inside a segment, ART runs
are the cores of story beats; border lines separate touching frames;
dialogue zones are attached to the art they belong to. A speech bubble
sitting *inside* a drawing keeps art at the row margins, so it stays ART
and never splits its panel.
"""

import numpy as np

WHITE_BLANK, BLACK_BLANK, TEXT, ART, BORDER = 0, 1, 2, 3, 4
BLANK = 5  # short white gap inside a segment (not a gutter)


def classify_rows(image_bgr, config):
    import cv2
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    white = gray >= config.white_threshold
    white_frac = white.mean(axis=1)
    black_frac = (gray <= config.black_threshold).mean(axis=1)
    dark_frac = (gray <= config.border_dark_value).mean(axis=1)
    # Dialogue is black ink on white - colorless, with midtones only from
    # anti-aliased letter edges. Bright artwork (sky, pale skin, white
    # cloth against a background) carries color, and lots of midtones.
    midtone_frac = ((gray > config.black_threshold) & (gray < config.white_threshold)).mean(axis=1)
    chroma = (image_bgr.max(axis=2).astype(np.int16) - image_bgr.min(axis=2).astype(np.int16)).mean(axis=1)
    margin = max(1, int(width * config.margin_fraction))
    margin_white = (white[:, :margin].mean(axis=1) + white[:, -margin:].mean(axis=1)) / 2
    black = gray <= config.black_threshold
    margin_black = (black[:, :margin].mean(axis=1) + black[:, -margin:].mean(axis=1)) / 2

    # A black gutter is flat black. Very dark artwork (a night scene, a
    # dark brown close-up) can have every pixel under the black threshold
    # too, but it still has texture - so a black-blank row must also be
    # nearly constant, or dark scenes get chopped into slivers.
    row_std = gray.std(axis=1)

    codes = np.full(height, ART, dtype=np.uint8)
    colorless = (midtone_frac <= config.text_max_midtone) & (chroma <= config.text_max_chroma)
    # Dialogue on a white background: black ink, white margins.
    text_on_white = (white_frac >= config.text_white_fraction) & (margin_white >= config.text_margin_white)
    # Dialogue on a black background (some sections are drawn on black):
    # white bubbles/lettering, black margins. Same thing, inverted.
    text_on_black = (
        (black_frac >= config.text_white_fraction)
        & (margin_black >= config.text_margin_white)
        & (white_frac >= 0.005)
    )
    codes[colorless & (text_on_white | text_on_black)] = TEXT
    codes[dark_frac >= config.border_dark_fraction] = BORDER
    codes[(black_frac >= config.gutter_uniform_fraction) & (row_std <= config.black_blank_max_std)] = BLACK_BLANK
    codes[white_frac >= config.gutter_uniform_fraction] = WHITE_BLANK
    return codes, row_std


def blank_mask(codes):
    return (codes == WHITE_BLANK) | (codes == BLACK_BLANK)


def _rle(codes):
    runs = []
    start = 0
    for i in range(1, len(codes) + 1):
        if i == len(codes) or codes[i] != codes[start]:
            runs.append([int(codes[start]), start, i])
            start = i
    return runs


def _merge_adjacent(runs):
    merged = []
    for r in runs:
        if merged and merged[-1][0] == r[0]:
            merged[-1][2] = r[2]
        else:
            merged.append(list(r))
    return merged


def _smooth(runs, config):
    """Absorbs runs too short to be structural: a few art rows inside a
    dialogue zone (thick bubble outlines) become TEXT; a short bright band
    inside a drawing becomes ART. Repeats until stable."""
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for i, (cls, a, b) in enumerate(runs):
            prev_cls = runs[i - 1][0] if i > 0 else None
            next_cls = runs[i + 1][0] if i < len(runs) - 1 else None
            length = b - a
            # A dark line with drawing on both sides is part of the drawing
            # (a horizon, a floor, a motion line) - a frame edge touches
            # blank or dialogue space on at least one side.
            if cls == BORDER and prev_cls == ART and next_cls == ART:
                runs[i][0] = ART
                changed = True
                break
            if cls == ART and length < config.min_art_rows and ART not in (prev_cls, next_cls) \
                    and (prev_cls is not None or next_cls is not None):
                runs[i][0] = TEXT
                changed = True
                break
            if cls in (TEXT, BLANK) and length < config.min_text_rows and prev_cls == ART and next_cls == ART:
                runs[i][0] = ART
                changed = True
                break
        if changed:
            runs = _merge_adjacent(runs)
    return runs


def beats_from_codes(codes, row_std, balloon_row, config):
    """Returns [(y1, y2, has_art), ...] beats for one gutter-bounded
    segment, in local row coordinates. has_art is False for a beat made
    only of dialogue - the caller attaches those to a neighboring panel."""
    local = codes.copy()
    local[local == WHITE_BLANK] = BLANK
    runs = _merge_adjacent(_rle(local))
    # A short flat-black run is a frame line; a longer one is a gap in a
    # black-background section and behaves like white space.
    for r in runs:
        if r[0] == BLACK_BLANK:
            r[0] = BORDER if (r[2] - r[1]) <= config.border_max_rows else BLANK
    runs = _smooth(_merge_adjacent(runs), config)

    # A run classed ART is only a real story beat if it has actual drawing
    # in it - linework, shading, anything with row-to-row pixel variation -
    # OUTSIDE of any detected speech balloon. A smooth gradient/vignette
    # reads as neither blank nor text, so it falls through to ART by
    # default, but its rows are near-constant (median row_std well under
    # 5) versus 30-100+ for any real drawing, dark scenes included - two
    # orders of magnitude, not a close call. A wide or jagged burst bubble
    # ("!?", a shout) sitting on a plain background is the other way to
    # fool this test: the bubble's own black/white contrast pushes row_std
    # just as high as real linework, even though there is no drawing at
    # all - so rows inside a detected balloon are excluded before taking
    # the median, and what's left has to still show real content. A run
    # that's balloon end to end (no non-balloon rows at all) can't qualify
    # on this evidence alone. The median (not mean) guards against a
    # single contaminating row - a watermark sliver, a compression-seam
    # artifact at a page boundary - dragging an otherwise-flat run's mean
    # above threshold; it stays representative for genuine art too, which
    # is detailed throughout rather than in just a few rows. A
    # non-qualifying run can't anchor a beat on its own; it still rides
    # along with whichever real beat it ends up attached to.
    def is_core(run):
        cls, a, b = run
        if cls != ART:
            return False
        content_std = row_std[a:b][~balloon_row[a:b]]
        return len(content_std) > 0 and np.median(content_std) >= config.min_core_row_std

    cores = [i for i, r in enumerate(runs) if is_core(r)]
    if not cores:
        y1, y2 = _trim(runs, 0, len(runs))
        return [(y1, y2, False)] if y2 > y1 else []

    beats = []
    prev_end = 0  # run index where the current beat starts
    for k, core in enumerate(cores):
        if k == len(cores) - 1:
            beats.append(_trim(runs, prev_end, len(runs)))
            break
        between = runs[core + 1:cores[k + 1]]
        cut = _choose_cut(between, core + 1)
        beats.append(_trim(runs, prev_end, cut))
        prev_end = cut
    return [(y1, y2, True) for y1, y2 in beats if y2 > y1]


def _choose_cut(between, offset):
    """Where to split the non-art runs lying between two art cores.
    Prefer the widest blank gap (bubbles above/below a panel are usually
    separated by a gap); otherwise split right after a border line that
    closes the upper panel; otherwise everything goes with the panel above."""
    if not between:
        return offset
    blanks = [(r[2] - r[1], i) for i, r in enumerate(between) if r[0] == BLANK]
    if blanks:
        _, i = max(blanks)
        return offset + i + 1
    if between[0][0] == BORDER:
        return offset + 1
    return offset + len(between)


def _trim(runs, start, end):
    """(y1, y2) covering runs[start:end] minus leading/trailing blank rows."""
    chunk = runs[start:end]
    while chunk and chunk[0][0] == BLANK:
        chunk = chunk[1:]
    while chunk and chunk[-1][0] == BLANK:
        chunk = chunk[:-1]
    if not chunk:
        return (0, 0)
    return (chunk[0][1], chunk[-1][2])


def zones(codes, y_offset=0):
    """TEXT and BORDER runs as (cls, y1, y2) in strip coordinates - for the
    debug overlay."""
    out = []
    for cls, a, b in _merge_adjacent(_rle(codes)):
        if cls in (TEXT, BORDER):
            out.append((cls, a + y_offset, b + y_offset))
    return out
