#!/bin/bash
# Download the public detector weights into $RAP_MODELS (default: <repo>/models).
# TensorRT engines are built from them by scripts/00_build_engines.py on the target device.
set -euo pipefail
cd "$(dirname "$0")/.."
M="${RAP_MODELS:-$PWD/models}"
mkdir -p "$M/engines"
for w in yolov8s.pt rtdetr-l.pt; do
  [ -f "$M/$w" ] || curl -L --fail -o "$M/$w" "https://github.com/ultralytics/assets/releases/download/v8.3.0/$w"
done
ls -la "$M"
