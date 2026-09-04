"""Sorts merged detections into reading order."""

ROW_BUCKET = 40


def sort_reading_order(detections, mode="vertical_webtoon"):
    """
    vertical_webtoon: top-to-bottom, then left-to-right within a row band
    (same row-bucket approach the existing tool already uses - panels that
    land at roughly the same height are treated as one row).
    """
    if mode == "vertical_webtoon":
        return sorted(detections, key=lambda d: (round(d.y1 / ROW_BUCKET), d.x1))
    # Other modes (manga right-to-left, custom) are future work; default to
    # the same behavior rather than silently doing something unexpected.
    return sorted(detections, key=lambda d: (round(d.y1 / ROW_BUCKET), d.x1))
