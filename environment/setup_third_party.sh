#!/bin/bash
# Clone the prior-work code used by the pipeline at the exact commits used (details: third_party/PROVENANCE.md).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p third_party
clone () {   # name url commit
  if [ ! -d "third_party/$1/.git" ]; then git clone --quiet "$2" "third_party/$1"; fi
  git -C "third_party/$1" fetch --quiet origin "$3" 2>/dev/null || true
  git -C "third_party/$1" checkout --quiet "$3"
  echo "third_party/$1 @ $(git -C "third_party/$1" rev-parse --short HEAD)"
}
clone nuplan_devkit  https://github.com/motional/nuplan-devkit.git              e9241677997dd86bfc0bcd44817ab04fe631405b
clone tuplan_garage  https://github.com/autonomousvision/tuplan_garage.git      b51d5d04fac1bd4389653b9ab2ff73ea88f435a3
clone pkl            https://github.com/nv-tlabs/planning-centric-metrics.git   f6865f2b473303f2ff01a477bf6de4dce7109742
clone tip            https://github.com/qcraftai/tip.git                         e52fd48de0624a9c54e93c8436f6bd7529b2a5d3
echo "PKL's own weight links no longer resolve; the pipeline uses the planner.pt and masks_trainval.json vendored by TIP."
