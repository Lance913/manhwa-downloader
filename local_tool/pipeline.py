"""
Chapter pipeline:

  1. Classify every row of the stitched strip (gutter / border / dialogue /
     art) - page boundaries play no part, so a panel that straddles two
     source images is never cut.
  2. Long gutters split the strip into segments.
  3. Inside each segment, art runs become story beats with their dialogue
     attached (see beats.py).
  4. Optionally, YOLO runs on each beat for the debug overlay and as an
     agreement vote on confidence. It never decides boundaries.
"""

import cv2
import numpy as np

from detectors import Detection, segments_from_blank_rows
from beats import classify_rows, blank_mask, beats_from_codes, zones, TEXT, BORDER
from yolo_detector import yolo_detector
from balloon_detector import detect_balloons_on_strip


def enforce_balloon_boundaries(panels, balloons):
    """
    No beat boundary may pass through a speech balloon. For each balloon
    that straddles the boundary between two consecutive beats, the beat
    that already holds most of the balloon is extended to cover all of it.
    Beats are only ever extended, never shrunk, so no art is lost - the
    two beats may overlap by a few rows instead, which is harmless.
    """
    panels = sorted(panels, key=lambda d: d.y1)
    for _ in range(4):
        changed = False
        for a, b in zip(panels, panels[1:]):
            for bl in balloons:
                crosses_a_bottom = bl.y1 < a.y2 < bl.y2
                crosses_b_top = bl.y1 < b.y1 < bl.y2
                in_gap = a.y2 <= bl.y1 and bl.y2 <= b.y1
                if not (crosses_a_bottom or crosses_b_top or in_gap):
                    continue
                in_a = max(0, min(bl.y2, a.y2) - max(bl.y1, a.y1))
                in_b = max(0, min(bl.y2, b.y2) - max(bl.y1, b.y1))
                if in_a >= in_b:
                    if a.y2 < bl.y2:
                        a.y2 = bl.y2
                        changed = True
                else:
                    if b.y1 > bl.y1:
                        b.y1 = bl.y1
                        changed = True
        if not changed:
            break
    return panels


def attach_dialogue_only_beats(panels, has_art):
    """
    A beat holding only dialogue (a bubble floating between two panels,
    with gutters on both sides) is not a story beat of its own - it gets
    folded into the nearest neighboring panel (smaller gap wins, ties go
    to the panel above, which the dialogue most often follows).
    """
    items = [[p, art] for p, art in zip(panels, has_art)]
    changed = True
    while changed:
        changed = False
        for i, (p, art) in enumerate(items):
            if art:
                continue
            prev_i = next((j for j in range(i - 1, -1, -1) if items[j][1]), None)
            next_i = next((j for j in range(i + 1, len(items)) if items[j][1]), None)
            if prev_i is None and next_i is None:
                continue
            gap_prev = p.y1 - items[prev_i][0].y2 if prev_i is not None else None
            gap_next = items[next_i][0].y1 - p.y2 if next_i is not None else None
            if next_i is None or (prev_i is not None and gap_prev <= gap_next):
                items[prev_i][0].y2 = max(items[prev_i][0].y2, p.y2)
            else:
                items[next_i][0].y1 = min(items[next_i][0].y1, p.y1)
            items.pop(i)
            changed = True
            break
    return [p for p, _ in items]


def process_chapter(strip, config):
    classified = [classify_rows(page, config) for page in strip.pages]
    codes = np.concatenate([c for c, _ in classified])
    row_std = np.concatenate([s for _, s in classified])
    segments = segments_from_blank_rows(blank_mask(codes), config.min_gutter_height, config.min_segment_height)
    if not segments:
        segments = [(0, strip.height)]

    width = strip.width
    raw_by_detector = {"gutter": [], "text": [], "border": [], "balloon": [], "yolo": [], "yolo-text": []}

    # Run the balloon detector first: a wide/jagged shout bubble creates
    # huge black/white row-to-row contrast on its own, which reads exactly
    # like real linework to a pure pixel-variation test. Knowing which
    # rows are actually inside a detected balloon lets beat formation
    # ignore that contribution and ask whether there is anything ELSE
    # there - the whole reason a floating burst bubble on a plain
    # background was surviving as its own "art" beat.
    balloons = []
    if config.balloon_enabled:
        balloons = detect_balloons_on_strip(
            strip, config.balloon_model_path,
            confidence=config.balloon_confidence, imgsz=config.balloon_imgsz,
        )
        raw_by_detector["balloon"] = balloons
    # Padded generously before use as a content-exclusion mask (spike-style
    # "shout" bubbles have jagged points that extend well past the tight
    # oval the balloon model boxes - unpadded, two such bubbles close
    # together leave a gap where their interleaved spikes still read as
    # high-contrast "content"). The unpadded boxes are used everywhere
    # else (enforce_balloon_boundaries, the debug overlay).
    balloon_row = np.zeros(strip.height, dtype=bool)
    for b in balloons:
        pad = int(round(b.h * config.balloon_pad_fraction))
        balloon_row[max(0, b.y1 - pad):min(strip.height, b.y2 + pad)] = True

    panels = []
    has_art = []

    for y1, y2 in segments:
        raw_by_detector["gutter"].append(Detection(0, y1, width, y2, 0.85, "gutter", ["gutter"]))
        for cls, a, b in zones(codes[y1:y2], y1):
            name = "text" if cls == TEXT else "border"
            raw_by_detector[name].append(Detection(0, a, width, b, 1.0, name, [name]))

        for a, b, art in beats_from_codes(codes[y1:y2], row_std[y1:y2], balloon_row[y1:y2], config):
            panels.append(Detection(0, y1 + a, width, y1 + b, config.rows_weight, "rows", ["rows"]))
            has_art.append(art)

    panels = attach_dialogue_only_beats(panels, has_art)

    if balloons:
        panels = enforce_balloon_boundaries(panels, balloons)

    if config.yolo_enabled:
        for panel in panels:
            crop = strip.crop(panel.y1, panel.y2)
            yolo_panels, yolo_text = yolo_detector(
                crop,
                model_path=config.yolo_model_path,
                confidence_threshold=config.yolo_confidence,
                iou_threshold=config.yolo_iou,
                imgsz=config.yolo_imgsz,
                device=config.yolo_device,
                weight=config.yolo_weight,
                tile_aspect=config.yolo_tile_aspect,
                tile_overlap=config.yolo_tile_overlap,
            )
            for d in yolo_panels + yolo_text:
                raw_by_detector["yolo" if d.detector == "yolo" else "yolo-text"].append(
                    Detection(d.x1, d.y1 + panel.y1, d.x2, d.y2 + panel.y1, d.confidence, d.detector, list(d.sources))
                )
            if any(d.area >= 0.5 * panel.area for d in yolo_panels):
                panel.sources.append("yolo")
                panel.confidence = min(1.0, panel.confidence + 0.15)

    return {
        "segments": segments,
        "raw_by_detector": raw_by_detector,
        "panels": panels,
        "suspicious": [],
    }
