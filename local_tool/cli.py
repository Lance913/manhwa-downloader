"""
Multi-detector manhwa panel extraction - local CLI.

Usage:
    python cli.py --url https://example.com/manga/x/chapter-1 --output-dir out
    python cli.py --input-dir ./my_pages --output-dir out

The chapter is treated as one continuous strip, so panels are numbered
across the whole chapter in reading order (panels/panel_001.jpg ...), not
per source image.
"""

import argparse
import json
import os

import cv2

from config import Config
from detectors import Detection
from io_utils import load_from_url, load_from_folder
from strip import Strip
from pipeline import process_chapter
from debug_viz import draw_debug_image


def _to_page_boxes(strip, detections, page_index):
    """Detections (strip coords) clipped to one page, in that page's coords."""
    top, bottom = strip.offsets[page_index], strip.offsets[page_index + 1]
    out = []
    for d in detections:
        y1, y2 = max(d.y1, top), min(d.y2, bottom)
        if y2 > y1:
            out.append(Detection(d.x1, y1 - top, d.x2, y2 - top, d.confidence, d.detector, list(d.sources)))
    return out


def write_contact_sheet(panels_dir, out_path, thumb_width=200, max_height=520, columns=8):
    """One numbered overview image of every panel, for quick review."""
    import numpy as np
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


def main():
    parser = argparse.ArgumentParser(description="Multi-detector manhwa panel extractor")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="Chapter URL to scrape")
    source.add_argument("--input-dir", help="Folder of already-downloaded page images")
    parser.add_argument("--output-dir", default="output", help="Where to write panels/metadata/debug")
    parser.add_argument("--title", help="Override the chapter/output folder name")
    parser.add_argument("--no-debug", action="store_true", help="Skip writing debug overlay images")
    parser.add_argument("--yolo", action="store_true", help="Also run the YOLO model (overlay + confidence vote)")
    parser.add_argument("--no-yolo", action="store_true", help="Force the YOLO detector off")
    args = parser.parse_args()

    config = Config()
    if args.no_debug:
        config.debug = False
    if args.yolo:
        config.yolo_enabled = True
    if args.no_yolo:
        config.yolo_enabled = False

    if config.yolo_enabled:
        from yolo_detector import load_model, unavailable_reason
        if load_model(config.yolo_model_path) is not None:
            print(f"YOLO detector: enabled ({config.yolo_model_path})")
        else:
            print(f"YOLO detector: unavailable - {unavailable_reason()} (continuing without it)")
            config.yolo_enabled = False
    else:
        print("YOLO detector: disabled")

    if config.balloon_enabled:
        from balloon_detector import load_model as load_balloon_model, unavailable_reason as balloon_reason
        if load_balloon_model(config.balloon_model_path) is not None:
            print(f"Balloon detector: enabled ({config.balloon_model_path})")
        else:
            print(f"Balloon detector: unavailable - {balloon_reason()} (continuing without it)")
            config.balloon_enabled = False

    print("Loading pages...")
    if args.url:
        title, pages = load_from_url(args.url)
    else:
        title, pages = load_from_folder(args.input_dir)
    if args.title:
        from io_utils import sanitize_name
        title = sanitize_name(args.title)

    strip = Strip(pages)
    print(f"Chapter: {title}")
    print(f"Pages: {len(pages)}  ->  strip {strip.width}x{strip.height}")

    base_out = os.path.join(args.output_dir, title)
    panels_dir = os.path.join(base_out, "panels")
    pages_dir = os.path.join(base_out, "pages")
    debug_dir = os.path.join(base_out, "debug")
    for d in (panels_dir, pages_dir, debug_dir):
        os.makedirs(d, exist_ok=True)
        # Clear this tool's own files from a previous run of the same
        # chapter - stale panels from an older run would otherwise mix
        # with (and be misnumbered against) the new ones.
        for f in os.listdir(d):
            if f.startswith(("panel_", "page_")) and f.lower().endswith((".jpg", ".png")):
                os.remove(os.path.join(d, f))

    print("Detecting panels across the strip...")
    result = process_chapter(strip, config)
    panels = result["panels"]
    suspicious = result["suspicious"]
    raw = result["raw_by_detector"]
    print(f"  {len(result['segments'])} gutter segments -> {len(panels)} panels, {len(suspicious)} possible missed region(s)")

    panel_records = []
    for index, d in enumerate(panels, start=1):
        filename = f"panel_{index:03d}.jpg"
        crop = strip.crop(d.y1, d.y2, d.x1, d.x2)
        cv2.imwrite(os.path.join(panels_dir, filename), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        record = d.as_dict()
        record["id"] = index
        record["file"] = f"panels/{filename}"
        record["pages"] = [i + 1 for i in strip.page_span(d.y1, d.y2)]
        panel_records.append(record)
        if index % 25 == 0 or index == len(panels):
            print(f"  wrote {index}/{len(panels)} panels")

    for i, page in enumerate(strip.pages):
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

    print()
    print(f"Done. {len(panels)} panels from {len(pages)} pages, {len(suspicious)} possible missed region(s) flagged.")
    print(f"Output: {base_out}")


if __name__ == "__main__":
    main()
