"""
Shared chapter-extraction pipeline, used by both cli.py and gui.py so the
two never drift apart. Reports progress through plain callbacks instead of
print(), so a GUI can show them in a window instead of a console.
"""

import json
import os

import cv2
import numpy as np

from config import Config
from detectors import Detection
from io_utils import load_from_url, load_from_folder, sanitize_name
from strip import Strip
from pipeline import process_chapter
from debug_viz import draw_debug_image


def _to_page_boxes(strip, detections, page_index):
    top, bottom = strip.offsets[page_index], strip.offsets[page_index + 1]
    out = []
    for d in detections:
        y1, y2 = max(d.y1, top), min(d.y2, bottom)
        if y2 > y1:
            out.append(Detection(d.x1, y1 - top, d.x2, y2 - top, d.confidence, d.detector, list(d.sources)))
    return out


def write_contact_sheet(panels_dir, out_path, thumb_width=200, max_height=520, columns=8):
    """One numbered overview image of every panel, for quick review."""
    files = sorted(f for f in os.listdir(panels_dir) if f.startswith("panel_") and f.endswith(".jpg"))
    if not files:
        return
    cells = []
    for name in files:
        img = cv2.imread(os.path.join(panels_dir, name))
        h, w = img.shape[:2]
        thumb = cv2.resize(img, (thumb_width, max(1, int(h * thumb_width / w))), interpolation=cv2.INTER_AREA)
        truncated = thumb.shape[0] > max_height
        thumb = thumb[:max_height]
        cell = np.full((max_height + 26, thumb_width, 3), 255, np.uint8)
        cell[26:26 + thumb.shape[0]] = thumb
        if truncated:
            cv2.line(cell, (0, max_height + 24), (thumb_width, max_height + 24), (0, 0, 255), 3)
        cv2.putText(cell, f"{name[6:9]}  {h}px", (2, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
        cells.append(cell)
    while len(cells) % columns:
        cells.append(np.full_like(cells[0], 255))
    rows = [np.hstack(cells[i:i + columns]) for i in range(0, len(cells), columns)]
    cv2.imwrite(out_path, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


class Cancelled(Exception):
    """Raised via should_cancel() to stop a run early (e.g. user closed the window)."""


def run_chapter(output_dir, url=None, input_dir=None, title_override=None,
                config=None, log=print, progress=None, should_cancel=None):
    """
    Runs the full extraction pipeline for one chapter.

    log(message): called with human-readable status lines.
    progress(done, total): called as panels are written (total may be 0
    briefly before detection finishes).
    should_cancel(): polled periodically; raise Cancelled if it returns True.
    Returns the metadata dict that also gets written to chapter.json.
    """
    config = config or Config()

    def check_cancel():
        if should_cancel and should_cancel():
            raise Cancelled()

    if config.yolo_enabled:
        from yolo_detector import load_model, unavailable_reason
        if load_model(config.yolo_model_path) is not None:
            log(f"YOLO detector: enabled ({config.yolo_model_path})")
        else:
            log(f"YOLO detector: unavailable - {unavailable_reason()} (continuing without it)")
            config.yolo_enabled = False

    if config.balloon_enabled:
        from balloon_detector import load_model as load_balloon_model, unavailable_reason as balloon_reason
        if load_balloon_model(config.balloon_model_path) is not None:
            log(f"Balloon detector: enabled ({config.balloon_model_path})")
        else:
            log(f"Balloon detector: unavailable - {balloon_reason()} (continuing without it)")
            config.balloon_enabled = False

    log("Loading pages...")
    check_cancel()
    if url:
        title, pages = load_from_url(url)
    else:
        title, pages = load_from_folder(input_dir)
    if title_override:
        title = sanitize_name(title_override)

    check_cancel()
    strip = Strip(pages)
    log(f"Chapter: {title}")
    log(f"Pages: {len(pages)}  ->  strip {strip.width}x{strip.height}")

    base_out = os.path.join(output_dir, title)
    panels_dir = os.path.join(base_out, "panels")
    pages_dir = os.path.join(base_out, "pages")
    debug_dir = os.path.join(base_out, "debug")
    for d in (panels_dir, pages_dir, debug_dir):
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(d):
            if f.startswith(("panel_", "page_")) and f.lower().endswith((".jpg", ".png")):
                os.remove(os.path.join(d, f))

    log("Detecting panels across the strip...")
    check_cancel()
    result = process_chapter(strip, config)
    panels = result["panels"]
    suspicious = result["suspicious"]
    raw = result["raw_by_detector"]
    log(f"  {len(result['segments'])} gutter segments -> {len(panels)} panels, {len(suspicious)} possible missed region(s)")

    panel_records = []
    for index, d in enumerate(panels, start=1):
        check_cancel()
        filename = f"panel_{index:03d}.jpg"
        crop = strip.crop(d.y1, d.y2, d.x1, d.x2)
        cv2.imwrite(os.path.join(panels_dir, filename), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        record = d.as_dict()
        record["id"] = index
        record["file"] = f"panels/{filename}"
        record["pages"] = [i + 1 for i in strip.page_span(d.y1, d.y2)]
        panel_records.append(record)
        if progress:
            progress(index, len(panels))
        if index % 25 == 0 or index == len(panels):
            log(f"  wrote {index}/{len(panels)} panels")

    for i, page in enumerate(strip.pages):
        check_cancel()
        page_name = f"page_{i + 1:03d}"
        cv2.imwrite(os.path.join(pages_dir, f"{page_name}.jpg"), page)
        if config.debug:
            raw_page = {name: _to_page_boxes(strip, dets, i) for name, dets in raw.items()}
            merged_page = _to_page_boxes(strip, panels, i)
            sus_page = []
            for r in suspicious:
                top, bottom = strip.offsets[i], strip.offsets[i + 1]
                y1, y2 = max(r["y1"], top), min(r["y2"], bottom)
                if y2 > y1:
                    sus_page.append({"x1": r["x1"], "y1": y1 - top, "x2": r["x2"], "y2": y2 - top})
            debug_image = draw_debug_image(page, raw_page, merged_page, sus_page)
            cv2.imwrite(os.path.join(debug_dir, f"{page_name}_debug.jpg"), debug_image)

    metadata = {
        "title": title,
        "strip_width": strip.width,
        "strip_height": strip.height,
        "page_offsets": strip.offsets,
        "panels": panel_records,
        "possible_missing_panels": suspicious,
    }
    with open(os.path.join(base_out, "chapter.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    write_contact_sheet(panels_dir, os.path.join(base_out, "contact_sheet.jpg"))

    log("")
    log(f"Done. {len(panels)} panels from {len(pages)} pages, {len(suspicious)} possible missed region(s) flagged.")
    log(f"Output: {base_out}")

    metadata["output_dir"] = base_out
    return metadata
