"""
Vercel serverless function: given a chapter URL, fetches the page and returns
the chapter title plus the ordered list of page image URLs, guaranteed to be
in reading sequence.
"""

from http.server import BaseHTTPRequestHandler
import json
import re
from urllib.parse import urlparse, urljoin

import requests
from bs4 import BeautifulSoup

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Class/id keywords used by most manga/manhwa reader themes to wrap the
# actual chapter page images (as opposed to logos, ads, related-series
# thumbnails, or comment-avatar placeholders elsewhere on the page).
READER_CONTAINER_HINTS = (
    "chapter-reader", "chapter-content", "chapter-img", "chapter-images",
    "reading-content", "read-container", "reader-area", "reader-content",
    "page-container", "comic-page", "viewer",
)

# Attributes, in preference order, that reader themes stash the real image
# URL in - many lazy-load and only populate `src` with a loading spinner.
IMAGE_SRC_ATTRS = ("src", "data-src", "data-original", "data-lazy-src", "data-url")

PLACEHOLDER_PATTERNS = ("loading", "spinner", "blank.gif", "data:image")


def _best_src(img, base_url):
    for attr in IMAGE_SRC_ATTRS:
        value = (img.get(attr) or "").strip()
        if not value or value == "#":
            continue
        if any(p in value.lower() for p in PLACEHOLDER_PATTERNS):
            continue
        return urljoin(base_url, value)
    return None


NOISE_KEYWORDS = (
    "logo", "avatar", "icon", "loading",
    # Promo/watermark banners some scanlation groups splice in as their own
    # "page" image (e.g. "Read this on our site", Discord/Telegram plugs).
    "banner", "advert", "sponsor", "notice", "watermark", "donate",
    "read-online", "readonline", "read_online", "promo", "socials",
    "discord", "telegram", "patreon",
)


def _is_probably_noise(img):
    src_values = [img.get(attr) or "" for attr in IMAGE_SRC_ATTRS]
    haystack = " ".join([
        *src_values, img.get("alt") or "", img.get("class") and " ".join(img.get("class")) or "",
    ]).lower()
    return any(k in haystack for k in NOISE_KEYWORDS)


def _extract_ordered_images(html, base_url):
    soup = BeautifulSoup(html, "html.parser")

    # Strategy 1: Asura Scans (and similar reader themes) mark every chapter
    # page image with a data-page-index attribute - use that as the single
    # source of truth for ordering when it's present.
    indexed = []
    for img in soup.find_all("img"):
        idx_attr = img.get("data-page-index")
        if idx_attr is None:
            continue
        if _is_probably_noise(img):
            continue
        src = _best_src(img, base_url)
        if not src:
            continue
        try:
            idx = int(idx_attr)
        except ValueError:
            continue
        indexed.append((idx, src))

    if indexed:
        indexed.sort(key=lambda pair: pair[0])
        return [src for _, src in indexed]

    # Strategy 2: most reader themes (MangaNato and its many clones/forks
    # included) wrap every page image in one clearly-named container with
    # no per-image ordering attribute - the images are just in reading
    # order in the DOM.
    for container in soup.find_all(["div", "section"]):
        identifiers = " ".join(filter(None, [container.get("id"), " ".join(container.get("class", []))])).lower()
        if not any(hint in identifiers for hint in READER_CONTAINER_HINTS):
            continue
        images = []
        for img in container.find_all("img"):
            if _is_probably_noise(img):
                continue
            src = _best_src(img, base_url)
            if src:
                images.append(src)
        if len(images) >= 2:
            return images

    # Strategy 3: last-resort fallback - images whose URL path clearly
    # looks like a chapter page, kept in document order.
    fallback = []
    for img in soup.find_all("img"):
        if _is_probably_noise(img):
            continue
        src = _best_src(img, base_url)
        if src and "/chapter" in src.lower():
            fallback.append(src)

    return fallback


class handler(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(length) if length else b"{}"
            data = json.loads(raw or b"{}")

            url = (data.get("url") or "").strip()
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https") or not parsed.netloc:
                raise ValueError("Please enter a valid chapter URL")

            resp = requests.get(url, headers=REQUEST_HEADERS, timeout=15)
            resp.raise_for_status()

            images = _extract_ordered_images(resp.text, resp.url)
            if not images:
                raise ValueError("Couldn't find any chapter page images on that link")

            title_match = re.search(r"<title>(.*?)</title>", resp.text, re.I | re.S)
            title = re.sub(r"\s+", " ", title_match.group(1)).strip() if title_match else "chapter"
            # Strip common SEO suffixes like " | Asura Scans" or
            # " - Read Online Free", " - Read Online For Free", etc.
            title = title.split(" | ")[0].strip()
            title = re.sub(r"\s*[-|]\s*Read\s+Online\b.*$", "", title, flags=re.I).strip()

            # Most reader sites' image CDNs reject hotlink requests that
            # don't carry a Referer from the site itself - remember the
            # chapter page's own origin so /api/panels can send it back
            # when it downloads each image.
            referer = f"{parsed.scheme}://{parsed.netloc}/"

            self._send(200, {"title": title, "images": images, "referer": referer})
        except Exception as exc:  # noqa: BLE001 - surface any failure to the client
            self._send(400, {"error": str(exc)})
