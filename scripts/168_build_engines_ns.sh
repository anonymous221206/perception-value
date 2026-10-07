#!/bin/bash
# Follow-up track: export ONNX (as scripts/00_build_engines.py) and build FP16 TensorRT engines with the flags of the
# frozen nuScenes engines (trtexec --fp16 --memPoolSize=workspace:3072), one engine at a time.
# Usage: scripts/168_build_engines_ns.sh <weights_stem>:<mode> ...   e.g. yolov8m:ns_full_640
set -u
cd "$(dirname "$0")/.."; REPO=$(pwd)
ENG=${RAP_MODEL_ROOT:-$HOME/models/yolo}/engines
for spec in "$@"; do
  stem=${spec%%:*}; mode=${spec##*:}
  w=${RAP_MODEL_ROOT:-$HOME/models/yolo}/$stem.pt
  onnx=$ENG/${stem}_${mode}.onnx; eng=$ENG/${stem}_${mode}.engine
  if [ -f "$eng" ]; then echo "skip $stem $mode"; continue; fi
  if [ ! -f "$w" ]; then
    echo "=== $(date +%H:%M:%S) download $stem"
    (cd "$(dirname "$w")" && $REPO/scripts/py -c "from ultralytics import YOLO; YOLO('$stem.pt')") || { echo "FAIL download $stem"; exit 1; }
  fi
  if [ ! -f "$onnx" ]; then
    echo "=== $(date +%H:%M:%S) onnx $stem $mode"
    ./scripts/py - "$w" "$mode" "$onnx" <<'PY' || { echo "FAIL onnx $stem $mode"; exit 1; }
import sys
from pathlib import Path
sys.path.insert(0, "src")
from rap.detect import MODES
from ultralytics import YOLO
w, mode, onnx = sys.argv[1:4]
m = MODES[mode]
p = YOLO(w).export(format="onnx", imgsz=(m.net_h, m.net_w), opset=12, simplify=False, dynamic=False, half=False, batch=1)
Path(p).replace(onnx)
PY
  fi
  echo "=== $(date +%H:%M:%S) trt $stem $mode"
  /usr/src/tensorrt/bin/trtexec --onnx="$onnx" --saveEngine="$eng.tmp" --fp16 --memPoolSize=workspace:3072 \
      > "$ENG/build_${stem}_${mode}.log" 2>&1
  if [ -f "$eng.tmp" ]; then mv "$eng.tmp" "$eng"; echo "OK $stem $mode $(md5sum < "$eng" | cut -c1-32)";
  else echo "FAIL trt $stem $mode"; tail -3 "$ENG/build_${stem}_${mode}.log"; exit 1; fi
done
echo BUILD_DONE
