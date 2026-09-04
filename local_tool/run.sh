#!/usr/bin/env bash
# Run the tool: ./run.sh --url "<chapter url>" --output-dir ~/Downloads/Manhwa Panels
set -e
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
    echo "No .venv found - run ./setup.sh first."
    exit 1
fi

.venv/bin/python cli.py "$@"
