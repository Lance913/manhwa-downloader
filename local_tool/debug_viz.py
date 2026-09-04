"""Draws a debug overlay showing every detector's raw boxes, the merged
result, and any flagged possibly-missed regions."""

import cv2

COLORS = {
    "existing": (255, 120, 0),      # blue-ish (BGR)
    "whitespace": (0, 200, 255),    # yellow
    "gutter": (0, 200, 255),        # yellow (strip-level gutter segments)
    "text": (180, 180, 180),        # gray: dialogue zone rows
    "balloon": (200, 0, 200),       # purple: detected speech balloons
    "border": (255, 120, 0),        # blue: panel frame lines
    "yolo": (0, 200, 0),            # green
    "yolo-text": (160, 160, 160),   # gray, informational only
    "merged": (0, 0, 255),          # red
    "suspicious": (255, 0, 255),    # magenta
}


def draw_debug_image(image, raw_by_detector, merged, suspicious_regions):
    debug = image.copy()

    for name, detections in raw_by_detector.items():
        color = COLORS.get(name, (200, 200, 200))
        for d in detections:
            cv2.rectangle(debug, (d.x1, d.y1), (d.x2, d.y2), color, 2)

    for d in merged:
        cv2.rectangle(debug, (d.x1, d.y1), (d.x2, d.y2), COLORS["merged"], 3)
        label = f"{'+'.join(d.sources)} {d.confidence:.2f}"
        cv2.putText(debug, label, (d.x1 + 4, max(20, d.y1 + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["merged"], 2)

    for region in suspicious_regions:
        cv2.rectangle(debug, (region["x1"], region["y1"]), (region["x2"], region["y2"]),
                      COLORS["suspicious"], 3)
        cv2.putText(debug, "possible missed panel", (region["x1"] + 4, region["y1"] + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS["suspicious"], 2)

    return debug
