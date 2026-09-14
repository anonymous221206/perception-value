#!/usr/bin/env python
"""Report this machine against the reference platform the results were measured on (see src/rap/device.py)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rap.device import REFERENCE, detect, differences                          # noqa: E402

if __name__ == "__main__":
    info = detect()
    print("this machine:", json.dumps(info, indent=1))
    print("reference:   ", json.dumps(REFERENCE, indent=1))
    diff = differences(info)
    if diff:
        print("\nNOT the reference platform:\n  - " + "\n  - ".join(diff))
        print("Hardware-dependent steps (engine build, profiling, detection, budget overheads) will not reproduce the\n"
              "paper's numbers exactly. `python reproduce.py --tier cached` reproduces every table from shipped caches.")
        sys.exit(1)
    print("\nreference platform: hardware-dependent steps should reproduce the paper's numbers")
