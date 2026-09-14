"""Reference-platform check.

Every latency and energy number, and every cached detection, was produced on one platform:

    NVIDIA Jetson AGX Xavier (Tegra194), JetPack 5.1.6 / L4T R35.6.5, CUDA 11.4, cuDNN 8.6,
    TensorRT 8.5.2.2, PyTorch 2.1.0a0+nv23.6, nvpmodel MAXN, FP16 TensorRT engines.

What differs elsewhere:
  * latency, energy and allocator overheads are properties of the hardware and will not match;
  * detections from TensorRT FP16 engines depend on the GPU, the TensorRT version and the chosen kernels, so
    re-running detection elsewhere will not reproduce data/cache bit for bit, and every downstream table can
    shift slightly;
  * statistics recomputed from the shipped caches (reproduce.py --tier cached) are deterministic on any machine.
"""
from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

REFERENCE = {"soc": "tegra194 (Jetson AGX Xavier)", "l4t": "R35.6.5", "tensorrt": "8.5.2.2", "cuda": "11.4",
             "torch": "2.1.0a0+41361538.nv23.6"}


def _read(path: str) -> str:
    try:
        return Path(path).read_bytes().replace(b"\x00", b" ").decode(errors="replace").strip()
    except OSError:
        return ""


def detect() -> dict:
    info = {"machine": platform.machine(), "model": _read("/proc/device-tree/model"),
            "compatible": _read("/proc/device-tree/compatible"), "l4t": "", "tensorrt": None, "torch": None, "cuda": None}
    rel = _read("/etc/nv_tegra_release")
    if rel.startswith("# R"):
        major = rel.split()[1]
        rev = rel.split("REVISION:")[1].split(",")[0].strip() if "REVISION:" in rel else ""
        info["l4t"] = f"{major}.{rev}"
    try:
        import tensorrt
        info["tensorrt"] = tensorrt.__version__
    except Exception:                                                          # noqa: BLE001
        pass
    try:
        import torch
        info["torch"], info["cuda"] = torch.__version__, torch.version.cuda
    except Exception:                                                          # noqa: BLE001
        pass
    return info


def differences(info: dict | None = None) -> list[str]:
    info = info or detect()
    out = []
    if "tegra194" not in info["compatible"]:
        out.append(f"SoC is not Jetson AGX Xavier (tegra194): {info['model'] or info['machine']}")
    if info["l4t"] and info["l4t"] != REFERENCE["l4t"]:
        out.append(f"L4T {info['l4t']} (reference {REFERENCE['l4t']})")
    if info["tensorrt"] is not None and info["tensorrt"] != REFERENCE["tensorrt"]:
        out.append(f"TensorRT {info['tensorrt']} (reference {REFERENCE['tensorrt']})")
    if info["tensorrt"] is None:
        out.append("TensorRT not importable")
    return out


def warn_if_not_reference(what: str, strict: bool = False) -> list[str]:
    """Print a warning when this machine is not the reference platform.

    `strict` steps (detection that overwrites the shipped caches, profiling) refuse to run unless
    RAP_ALLOW_OTHER_DEVICE=1 is set, so shipped reference outputs are not replaced by accident.
    """
    diff = differences()
    if not diff:
        return diff
    bar = "=" * 96
    msg = (f"\n{bar}\nWARNING: {what} is hardware-dependent and this machine is not the reference platform:\n  - "
           + "\n  - ".join(diff)
           + "\nLatency and energy will not match the paper, and TensorRT FP16 detections (hence every table built on "
             "them) can differ.\nTo reproduce the paper's tables exactly on any machine, use the shipped caches: "
             "python reproduce.py --tier cached\n" + bar + "\n")
    print(msg, file=sys.stderr, flush=True)
    if strict and os.environ.get("RAP_ALLOW_OTHER_DEVICE") != "1":
        sys.exit("refusing to run on a non-reference device; set RAP_ALLOW_OTHER_DEVICE=1 to run anyway")
    return diff
