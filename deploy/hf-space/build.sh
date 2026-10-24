#!/usr/bin/env sh
# Assemble the Hugging Face Space repository from backend/ + deploy/hf-space/.
# Usage: sh deploy/hf-space/build.sh <output-dir>
set -eu
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${1:?usage: build.sh <output-dir>}"
mkdir -p "$OUT"
rm -rf "$OUT/app" "$OUT/sample_data"
cp -R "$ROOT/backend/app" "$ROOT/backend/sample_data" "$OUT/"
cp "$ROOT/deploy/hf-space/README.md" "$ROOT/deploy/hf-space/space_app.py" "$ROOT/deploy/hf-space/requirements.txt" "$OUT/"
find "$OUT/app" -name "__pycache__" -type d -prune -exec rm -rf {} +
printf '__pycache__/\n*.pyc\ndata/\n.env\n' > "$OUT/.gitignore"
echo "Space files written to $OUT"
