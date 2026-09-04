# Manhwa Panel Downloader

Paste a chapter link, the app scrapes every page image in the correct
reading order, runs each page through a panel/image-detection algorithm
(ported from [adenzu/Manga-Panel-Extractor](https://github.com/adenzu/Manga-Panel-Extractor)'s
classical OpenCV pipeline), and packages every cropped panel into a ZIP you
can download.

## How it works

- `api/scrape.py` — fetches the chapter page and extracts the ordered list
  of page image URLs (uses each `<img data-page-index>` attribute, which is
  how Asura Scans marks reading order).
- `api/panels.py` — downloads one page image, finds its background/gutter
  regions with OpenCV (background-color masking, contour detection, and an
  optional "split touching panels" pass), and returns each cropped panel as
  a base64 PNG.
- `index.html` — the UI. It calls `scrape`, then calls `panels` for every
  page (3 at a time), and zips the results client-side with JSZip before
  triggering the browser download. Nothing is stored server-side.

## Run locally

```bash
npm install -g vercel   # if you don't have it
vercel dev
```

Then open the printed local URL.

## Deploy to Vercel

```bash
vercel        # first deploy / preview
vercel --prod # production deploy
```

Or push this folder to a GitHub repo and import it in the Vercel dashboard
(Framework Preset: **Other** — no build step needed, static `index.html` +
Python functions in `api/` are picked up automatically).

## Notes

- For personal use only — respect the source site's terms of service and
  copyright holders. Don't hammer the origin site; the UI already limits
  concurrent page requests.
- The panel-splitting heuristic is classical computer vision (no ML model),
  so it works well on clean panels with visible gutters/background but can
  miss splits in busy full-bleed art. The "split touching panels" checkbox
  enables an extra pass for panels with no visible gap, at the cost of
  speed.
