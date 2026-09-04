#!/usr/bin/env bash
# One-time setup on Mac/Linux. Run once: bash setup.sh
set -e
cd "$(dirname "$0")"

PYTHON=python3.12
if ! command -v $PYTHON &> /dev/null; then
    echo "python3.12 not found. Install it first:"
    echo "  brew install python@3.12"
    echo "(torch does not yet support the newest Python - 3.12 is required, not whatever 'python3' points to.)"
    exit 1
fi

$PYTHON -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

echo
echo "Setup done. Double-click 'Manhwa Downloader.app' in this folder to open it -"
echo "no terminal needed from here on. (Command-line alternative: ./run.sh --url \"<chapter url>\" --output-dir ~/Downloads/\"Manhwa Panels\")"
