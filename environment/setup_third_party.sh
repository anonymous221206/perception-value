#!/bin/bash
# Clone the prior-work code used by the pipeline at the exact commits used (details: third_party/PROVENANCE.md).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p third_party
want () {    # with no arguments, everything; otherwise only the names given
  [ "$#" -eq 0 ] && return 1
  for w in "$@"; do [ "$w" = "$WANT_NAME" ] && return 0; done
  return 1
}
clone () {   # name url commit
  WANT_NAME="$1"
  if [ -n "${SELECTED:-}" ] && ! want $SELECTED; then return 0; fi
  if [ ! -d "third_party/$1/.git" ]; then git clone --quiet "$2" "third_party/$1"; fi
  git -C "third_party/$1" fetch --quiet origin "$3" 2>/dev/null || true
  git -C "third_party/$1" checkout --quiet "$3"
  echo "third_party/$1 @ $(git -C "third_party/$1" rev-parse --short HEAD)"
}
SELECTED="$*"
clone nuplan_devkit  https://github.com/motional/nuplan-devkit.git              e9241677997dd86bfc0bcd44817ab04fe631405b
clone tuplan_garage  https://github.com/autonomousvision/tuplan_garage.git      b51d5d04fac1bd4389653b9ab2ff73ea88f435a3
clone pkl            https://github.com/nv-tlabs/planning-centric-metrics.git   f6865f2b473303f2ff01a477bf6de4dce7109742
clone tip            https://github.com/qcraftai/tip.git                         e52fd48de0624a9c54e93c8436f6bd7529b2a5d3
clone edgeml-object-detection https://github.com/qiujiaming315/edgeml-object-detection.git 859f70240aa090359ba827b374b68ca7821b7d55
clone bgt-ada       https://github.com/ViGeng/bgt-ada.git                         6669ab0089a04fbe6257ebdc2601de13ed0e5398
echo "PKL's own weight links no longer resolve; the pipeline uses the planner.pt and masks_trainval.json vendored by TIP."
