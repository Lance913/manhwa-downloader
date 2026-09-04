"""
Panel detectors. Each detector takes a BGR page image and returns a list of
Detection boxes. Independent detectors give redundancy: a panel missed by
one can still be recovered by another.
"""

import sys
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))
from panels import (  # noqa: E402
    generate_panel_blocks,
    generate_background_mask,
    preprocess_image_with_dilation,
    get_page_without_background,
    is_contour_sufficiently_big,
    merge_fragmented_boxes,
    threshold_extraction,
)


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    detector: str
    sources: list = field(default_factory=list)  # filled in during merge

    @property
    def w(self):
        return self.x2 - self.x1

    @property
    def h(self):
        return self.y2 - self.y1

    @property
    def area(self):
        return max(0, self.w) * max(0, self.h)

    def as_dict(self):
        return {
            "x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2,
            "confidence": round(self.confidence, 3),
            "detectors": self.sources or [self.detector],
        }


def _contours_to_detections(image, detector_name, confidence, split_joint_panels=False):
    """Runs the existing background-mask detector but keeps bounding boxes
    (not cropped pixels) so the ensemble can merge/dedupe by geometry."""
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed_image = preprocess_image_with_dilation(grayscale_image)
    background_mask = generate_background_mask(processed_image)
    page_without_background = get_page_without_background(grayscale_image, background_mask, split_joint_panels)

    contours, _ = cv2.findContours(page_without_background, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if is_contour_sufficiently_big(c, image.shape[0], image.shape[1])]

    height, width = image.shape[:2]
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w >= width * 0.99 and h >= height * 0.99:
            continue
        boxes.append((x, y, w, h))
    # Re-unite boxes that are really fragments of one panel (a mid-panel
    # patch of background misread as a gutter, a bleeding speech bubble
    # picked up as its own tiny contour, etc.) - the same step the existing
    # tool already relies on. Skipping this is what caused one continuous
    # scene to come out as several thin side-by-side slivers.
    boxes = merge_fragmented_boxes(boxes)

    detections = [
        Detection(x, y, x + w, y + h, confidence, detector_name)
        for x, y, w, h in boxes
    ]

    if len(detections) < 2:
        # Same fallback the current tool already uses: a Laplacian/adaptive
        # threshold pass tends to catch panels the mask approach collapses.
        alt_panels = threshold_extraction(image, grayscale_image, mode="bounding")
        if len(alt_panels) > len(detections):
            # threshold_extraction returns cropped pixels, not boxes; re-derive
            # boxes is wasteful, so just rerun contour extraction the same way
            # threshold_extraction does, to keep coordinates.
            processed = cv2.GaussianBlur(grayscale_image, (3, 3), 0)
            processed = cv2.Laplacian(processed, -1)
            _, thresh = cv2.threshold(processed, 8, 255, cv2.THRESH_BINARY)
            processed = cv2.adaptiveThreshold(processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 0)
            processed = cv2.subtract(processed, thresh)
            processed = cv2.dilate(processed, np.ones((3, 3), np.uint8), iterations=2)
            alt_contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            alt_contours = [c for c in alt_contours if is_contour_sufficiently_big(c, height, width)]
            alt_boxes = []
            for c in alt_contours:
                x, y, w, h = cv2.boundingRect(c)
                if w >= width * 0.99 and h >= height * 0.99:
                    continue
                alt_boxes.append((x, y, w, h))
            alt_boxes = merge_fragmented_boxes(alt_boxes)
            detections = [
                Detection(x, y, x + w, y + h, confidence, detector_name + "-threshold")
                for x, y, w, h in alt_boxes
            ]

    return detections


def existing_detector(image, split_joint_panels=False, confidence=1.0):
    """Detector A: the current tool's background-mask contour detector."""
    return _contours_to_detections(image, "existing", confidence, split_joint_panels=split_joint_panels)


def whitespace_gutter_detector(image, confidence=0.85, min_gutter_height=20,
                                white_threshold=245, white_row_fraction=0.985,
                                min_segment_height=60):
    """
    Detector B: scans full-width horizontal bands for near-blank rows and
    treats a long-enough run of them as a gutter between panels.

    This targets the classic webtoon failure mode the contour/background-
    mask detector can struggle with: borderless panels separated only by
    plain whitespace, where the "background" the mask approach finds can
    swallow part of a panel if that panel's own background is close to
    white. This detector never looks at contours at all, so it fails
    differently and independently.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    width = gray.shape[1]
    is_blank_row = blank_row_mask(gray, white_threshold=white_threshold,
                                  uniform_fraction=white_row_fraction)
    return [
        Detection(0, y1, width, y2, confidence, "whitespace")
        for y1, y2 in segments_from_blank_rows(is_blank_row, min_gutter_height, min_segment_height)
    ]


def blank_row_mask(gray, white_threshold=235, black_threshold=25, uniform_fraction=0.985,
                   uniform_rows=False, max_row_std=3.0):
    """
    Marks rows that look like gutter: nearly every pixel near-white, or
    nearly every pixel near-black (some series switch to black gutters for
    night/dramatic sections). Optionally also any flat single-color row -
    off by default, because solid-color panel backgrounds are common in
    manhwa and would get sliced at every flat stretch.
    """
    white = (gray >= white_threshold).mean(axis=1) >= uniform_fraction
    black = (gray <= black_threshold).mean(axis=1) >= uniform_fraction
    blank = white | black
    if uniform_rows:
        blank |= gray.std(axis=1) <= max_row_std
    return blank


def segments_from_blank_rows(is_blank_row, min_gutter_height=20, min_segment_height=60):
    """
    Turns a per-row blank mask into (y1, y2) content segments separated by
    gutters at least min_gutter_height tall.

    A gutter-bounded segment that's very short is rarely a real panel -
    usually a caption strip, a stray decorative rule, or a thin sliver of
    a bubble/logo that briefly broke up an otherwise blank run. Rather
    than dropping that content, it's folded into a neighboring segment.
    """
    height = len(is_blank_row)
    segments = []
    panel_start = 0
    gutter_start = None
    for y in range(height):
        if is_blank_row[y]:
            if gutter_start is None:
                gutter_start = y
        else:
            if gutter_start is not None:
                if y - gutter_start >= min_gutter_height:
                    if gutter_start > panel_start:
                        segments.append([panel_start, gutter_start])
                    panel_start = y
                gutter_start = None
    if panel_start < height:
        # Trailing blank rows at the very end are gutter, not content.
        end = height
        if gutter_start is not None and height - gutter_start >= min_gutter_height:
            end = gutter_start
        if end > panel_start:
            segments.append([panel_start, end])

    if not segments:
        return []
    folded = [segments[0]]
    for y1, y2 in segments[1:]:
        if y2 - y1 < min_segment_height:
            folded[-1][1] = y2
        else:
            folded.append([y1, y2])
    if len(folded) > 1 and folded[0][1] - folded[0][0] < min_segment_height:
        folded[1][0] = folded[0][0]
        folded.pop(0)
    return [(y1, y2) for y1, y2 in folded if y2 > y1]


DETECTORS = {
    "existing": existing_detector,
    "whitespace": whitespace_gutter_detector,
}
