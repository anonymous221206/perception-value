#!/usr/bin/env python
"""Follow-up E9, Jetson: a TensorRT FP16 engine of a Depth Anything V2 ONNX export (213) with its layer norms in FP32.

addendum E9 in docs/followup_protocol.md. At opset 16 each layer norm is exported as elementary operations (ReduceMean, Sub,
Pow, Sqrt, Div, ...), and TensorRT 8.5 runs them in FP16 like any other layer. Every layer whose ONNX name lies in a
norm module (norm1, norm2, layernorm*) is held in FP32; everything else may run in FP16, as trtexec --fp16 builds it.
"""
from __future__ import annotations

import argparse, re
from pathlib import Path

import tensorrt as trt

NORM = re.compile(r"/(norm\d|layernorm(_\d+)?)/")
FLOAT = (trt.float32, trt.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--workspace-mb", type=int, default=4096)
    args = ap.parse_args()
    log = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(log)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(net, log)
    if not parser.parse_from_file(args.onnx):
        raise SystemExit("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_mb << 20)
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    held = 0
    for i in range(net.num_layers):
        layer = net.get_layer(i)
        if NORM.search(layer.name) and all(layer.get_output(j).dtype in FLOAT for j in range(layer.num_outputs)):
            layer.precision = trt.float32
            for j in range(layer.num_outputs):
                layer.set_output_type(j, trt.float32)
            held += 1
    print(f"{net.num_layers} layers, {held} held in FP32", flush=True)
    plan = builder.build_serialized_network(net, config)
    if plan is None:
        raise SystemExit("build failed")
    Path(args.engine).write_bytes(plan)
    print("wrote", args.engine, flush=True)


if __name__ == "__main__":
    main()
