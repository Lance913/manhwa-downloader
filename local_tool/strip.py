"""
A chapter as one continuous vertical strip.

Webtoon sites slice a chapter into arbitrary fixed-height image chunks, so
a panel can straddle a "page" boundary. Treating the chapter as a single
strip makes those boundaries disappear. The full strip is never
materialized (a 100-page chapter would be ~150k px tall); pages stay
separate and slices are assembled on demand.
"""

import cv2
import numpy as np


class Strip:
    def __init__(self, pages):
        if not pages:
            raise ValueError("No pages")
        widths = sorted(p.shape[1] for p in pages)
        self.width = widths[len(widths) // 2]
        self.pages = []
        for page in pages:
            if page.shape[1] != self.width:
                new_height = max(1, round(page.shape[0] * self.width / page.shape[1]))
                page = cv2.resize(page, (self.width, new_height), interpolation=cv2.INTER_AREA)
            self.pages.append(page)
        self.offsets = [0]
        for page in self.pages:
            self.offsets.append(self.offsets[-1] + page.shape[0])
        self.height = self.offsets[-1]

    def page_span(self, y1, y2):
        """Indices of pages overlapping strip rows [y1, y2)."""
        first = max(0, np.searchsorted(self.offsets, y1, side="right") - 1)
        last = max(0, np.searchsorted(self.offsets, max(y1, y2 - 1), side="right") - 1)
        return list(range(first, last + 1))

    def crop(self, y1, y2, x1=0, x2=None):
        y1, y2 = max(0, y1), min(self.height, y2)
        x2 = self.width if x2 is None else min(self.width, x2)
        parts = []
        for i in self.page_span(y1, y2):
            top, bottom = self.offsets[i], self.offsets[i + 1]
            a, b = max(y1, top) - top, min(y2, bottom) - top
            if b > a:
                parts.append(self.pages[i][a:b, x1:x2])
        if not parts:
            return np.zeros((0, x2 - x1, 3), dtype=np.uint8)
        return parts[0] if len(parts) == 1 else np.vstack(parts)

    def to_page_coords(self, y1, y2):
        """Splits a strip box into per-page boxes: [(page_index, local_y1, local_y2), ...]."""
        out = []
        for i in self.page_span(y1, y2):
            top, bottom = self.offsets[i], self.offsets[i + 1]
            out.append((i, max(y1, top) - top, min(y2, bottom) - top))
        return out
