#!/usr/bin/env python
"""Report this machine against the reference platform the results were measured on (see src/rap/device.py)."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rap.device import REFERENCE, detect, differences                          # noqa: E402


def _norm(version):
    """Compare versions as PEP 440 does for local labels: "nv23.06" and "nv23.6" are the same release."""
    return [str(int(t)) if t.isdigit() else t.lower() for t in re.split(r"[.+]", version)]

if __name__ == "__main__":
    info = detect()
    print("this machine:", json.dumps(info, indent=1))
    print("reference:   ", json.dumps(REFERENCE, indent=1))
    diff = differences(info)
    if info["torch"] is None:
        diff.append("PyTorch not importable in this Python: the PyTorch and CUDA versions cannot be checked "
                    "(run this inside the main environment)")
    elif _norm(info["torch"]) != _norm(REFERENCE["torch"]) or info["cuda"] != REFERENCE["cuda"]:
        diff.append(f"PyTorch {info['torch']} / CUDA {info['cuda']} (reference {REFERENCE['torch']} / {REFERENCE['cuda']})")
    if diff:
        print("\nNOT the reference platform:\n  - " + "\n  - ".join(diff))
        print("Hardware-dependent steps (engine build, profiling, detection, budget overheads) will not reproduce the\n"
              "paper's numbers exactly. `python reproduce.py --tier cached` reproduces every table from shipped caches.")
        sys.exit(1)
    print("\nreference platform: hardware-dependent steps should reproduce the paper's numbers")
