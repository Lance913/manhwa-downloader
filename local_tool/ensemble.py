"""
Merges detections from multiple independent detectors: IoU-based
deduplication (agreeing boxes become one, higher-confidence), plus
missing-panel gap analysis on what's left uncovered.
"""

import cv2
import numpy as np

from detectors import Detection

IOU_DUPLICATE_THRESHOLD = 0.5
CONTAINMENT_DUPLICATE_THRESHOLD = 0.85


def iou(a, b):
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def containment(a, b):
    """How much of the SMALLER box sits inside the bigger one. Plain IoU
    misses this: a small box nearly swallowed by a much larger one scores a
    low IoU (the union is dominated by the big box) even though it isn't a
    separate panel at all - it's the same region, just framed differently
    by two detectors of very different granularity."""
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    smaller = min(a.area, b.area)
    return inter / smaller if smaller > 0 else 0.0


def merge_detections(all_detections, iou_threshold=IOU_DUPLICATE_THRESHOLD,
                      containment_threshold=CONTAINMENT_DUPLICATE_THRESHOLD,
                      solo_min_confidence=None):
    """
    Groups detections that overlap enough to be "the same panel" and
    collapses each group into one box (the union of its members, so the
    final crop never cuts off part of a panel that one detector framed
    slightly tighter than another). Confidence rises when independent
    detectors agree; a box only one detector found is kept, just ranked
    lower - it may be exactly what the others missed.

    solo_min_confidence: {detector_name: threshold}. A group made up of
    only that one detector is dropped when its best confidence is below
    the threshold - it still counts as an agreement vote when it overlaps
    something another detector found, it just can't spawn a panel alone.
    Used to stop a noisy ML detector from inventing panels nothing else
    sees, without throwing away its high-confidence solo finds.
    """
    solo_min_confidence = solo_min_confidence or {}

    def is_duplicate(a, b):
        return iou(a, b) >= iou_threshold or containment(a, b) >= containment_threshold

    remaining = list(all_detections)
    groups = []

    while remaining:
        seed = remaining.pop(0)
        group = [seed]
        changed = True
        while changed:
            changed = False
            for other in remaining[:]:
                if any(is_duplicate(other, member) for member in group):
                    group.append(other)
                    remaining.remove(other)
                    changed = True
        groups.append(group)

    merged = []
    for group in groups:
        x1 = min(d.x1 for d in group)
        y1 = min(d.y1 for d in group)
        x2 = max(d.x2 for d in group)
        y2 = max(d.y2 for d in group)
        detector_names = sorted(set(d.detector for d in group))
        # More independent detectors agreeing -> higher confidence, capped at 1.
        base_confidence = max(d.confidence for d in group)
        if len(detector_names) == 1:
            solo_threshold = solo_min_confidence.get(detector_names[0])
            if solo_threshold is not None and base_confidence < solo_threshold:
                continue
        agreement_bonus = 0.15 * (len(detector_names) - 1)
        confidence = min(1.0, base_confidence + agreement_bonus)
        det = Detection(x1, y1, x2, y2, confidence, detector_names[0], sources=detector_names)
        merged.append(det)

    merged.sort(key=lambda d: (d.y1, d.x1))
    return merged


def suppress_covered_solo(image, merged, suppressible=("yolo",), min_new_content_edge_density=0.01):
    """
    Drops a single-detector box that adds no new content: everything inside
    it is either already inside other panels or blank gutter. Seen in
    practice with the Manga109-trained YOLO on webtoons - it frames a speech
    bubble that a classical detector already captured, plus the empty gap
    beneath it, plus a sliver of the next panel. Such a box overlaps two
    real panels too little for IoU/containment to merge it, so it would
    otherwise become a duplicate "panel" of its own. Boxes with genuinely
    uncovered content are kept regardless of confidence - that content is
    exactly what the other detectors missed.
    """
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160) > 0

    def is_suppressible_solo(d):
        return len(d.sources) == 1 and d.sources[0] in suppressible

    vouching = [d for d in merged if not is_suppressible_solo(d)]
    covered = np.zeros((height, width), dtype=bool)
    for other in vouching:
        covered[max(0, other.y1):min(height, other.y2), max(0, other.x1):min(width, other.x2)] = True

    kept = []
    for det in merged:
        if not is_suppressible_solo(det):
            kept.append(det)
            continue
        y1, y2 = max(0, det.y1), min(height, det.y2)
        x1, x2 = max(0, det.x1), min(width, det.x2)
        uncovered = ~covered[y1:y2, x1:x2]
        uncovered_pixels = int(uncovered.sum())
        if uncovered_pixels == 0:
            continue
        new_edge_pixels = int((edges[y1:y2, x1:x2] & uncovered).sum())
        if new_edge_pixels / uncovered_pixels < min_new_content_edge_density:
            continue
        kept.append(det)
    return kept


def find_missing_regions(image, merged_detections, min_area_fraction=0.02, coverage_threshold=0.15):
    """
    Flags large areas of the page not covered by any merged detection as
    "possibly missed" - not auto-added as panels, just surfaced so a human
    (or a future review UI) can check them.
    """
    height, width = image.shape[:2]
    coverage = np.zeros((height, width), dtype=np.uint8)
    for d in merged_detections:
        coverage[max(0, d.y1):min(height, d.y2), max(0, d.x1):min(width, d.x2)] = 1

    uncovered = 1 - coverage
    min_area = min_area_fraction * width * height

    # Scan uncovered rows for runs tall enough to plausibly hold a panel,
    # the same way the whitespace detector finds gutters - but here we want
    # the opposite: gaps that are NOT blank (i.e. still have visual content).
    gray = None
    try:
        import cv2
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except Exception:
        pass

    row_uncovered = uncovered.mean(axis=1) > 0.7
    suspicious = []
    y = 0
    while y < height:
        if row_uncovered[y]:
            start = y
            while y < height and row_uncovered[y]:
                y += 1
            band_height = y - start
            band_area = band_height * width
            if band_area >= min_area:
                # Only flag if the band actually has visual content (not a
                # genuine blank margin) - use edge density as a cheap proxy.
                if gray is not None:
                    band = gray[start:y, :]
                    edges = cv2.Canny(band, 60, 160)
                    has_content = np.count_nonzero(edges) / edges.size > 0.01
                else:
                    has_content = True
                if has_content:
                    suspicious.append({"x1": 0, "y1": int(start), "x2": int(width), "y2": int(y)})
        else:
            y += 1

    return suspicious
