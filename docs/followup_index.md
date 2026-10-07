# Follow-up experiments — index

The index of the follow-up experiments, each specified before its values were computed (`docs/followup_protocol.md`). It records the gates, every deviation from the protocol (with date and reason) and every point the
protocol left open, with the reading taken.

The protocol was committed before any decision value on the new test scenes was computed. Its public copy is the
commit after release tag `v1.0`. Each addendum is appended to `docs/followup_protocol.md`, and the outputs of each experiment are under `results/followup/<experiment>/`.


## Gates

| gate | result | note |
|---|---|---|
| parity: the 320 / 640 engines on the frozen 85 scenes against the shipped detections | **pass**, 0 of 85 files differ in either mode | L4T R35.6.5, TensorRT 8.5.2.2, CUDA 11.4 |
| parity, second check: the frozen 85 scenes inside the 850-scene cache against the shipped cache | **pass**, 0 of 85 files differ | |
| (a) the new-scene pipeline reproduces the frozen 85-scene values | **pass**, 0 of 3,852 comparisons differ | `gate_a.json`: tables (78 shared columns, both geometries), 132 cell descriptives, 480 official nDG values, 288 R1 values, 72 R2 values and 2,880 significance-table values, exactly |
| planner chain on the frozen 85 scenes inside the 850-scene run | **pass**, 0 of 2,655 frames differ in either geometry | ADE / FDE, cheap and full |
| (b) uniform 320→640 braking on the new test scenes against the exploratory value (informational) | recorded: −0.1817 [−0.636, +0.231] against −0.181 (difference 0.0007); harmed 52.1% of 792 affected | |
| planner rasters of E5: E1's YOLOv8s 384×640 rasters redrawn from E1's submissions | **pass**, 14,468 of 14,468 test frames byte-identical | only the object channel is redrawn |

## Deviations

| what | why |
|---|---|
| Scene-0162 (new test) re-detected in every mode after one truncated CAM_FRONT image was re-fetched | an interrupted copy left one partial file; found by comparing all 34,149 file sizes with the source |
| Fine-tuning environment: `np.trapz` aliased to `np.trapezoid` | Ultralytics 8.3.40 calls `np.trapz`, which numpy 2 removed; same integral, recipe unchanged |

## Points the protocol left open, and the reading taken

| question | reading taken |
|---|---|
| "Ego-speed tercile (terciles of the pooled 850 scenes)": scene-level or frame-level speed | frame level: terciles of the controller's ego speed over all frames of the 850 scenes; cut points saved with the run |
| "budget_binding" in x1_settings | true when more frames have V > 0 than the 20% capacity allows |
| E5 planner rows need rasters per detector | run at the primary operating point only (the pairs of X7a and the two fine-tuned pairs); the other operating point (X7b) has braking rows only |
| X2 before the pixel router was trained on the follow-up scenes | X2 / X3 were first written without R2 and then rewritten with it; the shipped files include R2 |
