"""
Multi-detector manhwa panel extraction - local CLI.

Usage:
    python cli.py --url https://example.com/manga/x/chapter-1 --output-dir out
    python cli.py --url <chapter 1 url> --url <chapter 2 url> --output-dir out
    python cli.py --input-dir ./my_pages --output-dir out

The chapter is treated as one continuous strip, so panels are numbered
across the whole chapter in reading order (panels/panel_001.jpg ...), not
per source image.

Passing --url more than once switches to batch mode: each chapter gets
its own output_dir/<chapter title>/ folder holding just the cropped panel
images (no pages/debug/metadata) - a single --url keeps the full,
unchanged single-chapter output (panels/, pages/, debug/, chapter.json,
contact_sheet.jpg).
"""

import argparse

from config import Config
from runner import run_chapter, run_batch


def main():
    parser = argparse.ArgumentParser(description="Multi-detector manhwa panel extractor")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", action="append", help="Chapter URL to scrape (repeat for multiple chapters)")
    source.add_argument("--input-dir", help="Folder of already-downloaded page images")
    parser.add_argument("--output-dir", default="output", help="Where to write panels/metadata/debug")
    parser.add_argument("--title", help="Override the chapter/output folder name (single --url only)")
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

    if args.url and len(args.url) > 1:
        run_batch(urls=args.url, output_dir=args.output_dir, config=config)
    else:
        run_chapter(
            output_dir=args.output_dir,
            url=args.url[0] if args.url else None,
            input_dir=args.input_dir,
            title_override=args.title,
            config=config,
        )


if __name__ == "__main__":
    main()
