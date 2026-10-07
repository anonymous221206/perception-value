"""Local-file stand-in for the one mmengine module Diffusion Planner imports (fileio), used by 197 only.

Diffusion Planner reads its args and checkpoints through mmengine.fileio's local backend; for local paths that is
plain file IO, which this reproduces (get, get_text, put)."""
