# Follow-up experiments — index

The index of the follow-up experiments, each specified before its values were computed (`docs/followup_protocol.md`). It records the gates, every deviation from the protocol (with date and reason) and every point the
protocol left open, with the reading taken.

The protocol was committed before any decision value on the new test scenes was computed. Its public copy is the
commit after release tag `v1.0`. Each addendum is appended to `docs/followup_protocol.md`, and the outputs of each experiment are under `results/followup/<experiment>/`.

E8, E9 and E10 extend the protocol's list (Section 9 names E1–E7); each has its own addendum under the protocol's
Sections 3, 4 and 6.

Each addendum was committed before the first decision value of its experiment.

## Gates

| gate | result | note |
|---|---|---|
| parity: the 320 / 640 engines on the frozen 85 scenes against the shipped detections | **pass**, 0 of 85 files differ in either mode | L4T R35.6.5, TensorRT 8.5.2.2, CUDA 11.4 |
| parity, second check: the frozen 85 scenes inside the 850-scene cache against the shipped cache | **pass**, 0 of 85 files differ | |
| (a) the new-scene pipeline reproduces the frozen 85-scene values | **pass**, 0 of 3,852 comparisons differ | `gate_a.json`: tables (78 shared columns, both geometries), 132 cell descriptives, 480 official nDG values, 288 R1 values, 72 R2 values and 2,880 significance-table values, exactly |
| planner chain on the frozen 85 scenes inside the 850-scene run | **pass**, 0 of 2,655 frames differ in either geometry | ADE / FDE, cheap and full |
| (b) uniform 320→640 braking on the new test scenes against the exploratory value (informational) | recorded: −0.1817 [−0.636, +0.231] against −0.181 (difference 0.0007); harmed 52.1% of 792 affected | |
| planner rasters of E5: E1's YOLOv8s 384×640 rasters redrawn from E1's submissions | **pass**, 14,468 of 14,468 test frames byte-identical | only the object channel is redrawn |
| E2-0, reference fidelity (the port's reference branch against the published PDM-Closed score) | **pass with a documented residual** | on 1,013 navtest tokens the reference branch equals the exploratory wrapper token for token (PDMS 0.8958); the 0.1 s handover raises it by 0.003; NAVSIM v1.0's scorer lowers it by 0.007; about 0.008 against the published 0.881 remains unisolated (v1.1 ships no PDM-Closed agent); every difference found is common to cheap and full |
| E2-1, agreement with the exploratory values on 300 tokens (informational) | recorded: 97.3% of cheap and 98.7% of full scores equal to 1e-6 | |
| E6 determinism: the first scenario run twice on the GPU | **pass**, identical rows | |
| E7-0: OpenPCDet's nuScenes evaluation of each checkpoint on the 150 validation scenes, published mAP within 1 point | **pass**, all three: CenterPoint-PointPillar 50.03 (published 50.03), VoxelNeXt 60.52 (60.53), TransFusion-L 64.55 (64.58) | 6,019 keyframes |
| E8-0, E8a (Argoverse 1.1 alignment: recall ≥ 0.5 at 640 on boxes ≥ 30 px, median lift error within ±20%) | **fail, E8a stopped**: recall 0.444, then 0.449 with Section 3's LiDAR-point rule; lift error +1.7%, +1.9% | reported as not run |
| E8-0, E8b (Argoverse 2, both image versions) | **fail, E8b stopped**: recall 0.325 (raw) / 0.316 (undistorted); lift error +2.5% / +0.8% | no adapter fault; input scale and occlusion (breakdown) |
| E9-0: TensorRT engines against PyTorch FP32, median relative difference of box ranges < 1% | **pass on the third run**: Small 0.07% / 0.33%, Large 0.03% / 0.12% (nuScenes / KITTI) | run 1 exposed a loading fault (relative head), run 2 failed Large on FP16 layer norms; both fixed before any decision value |
| E10-0: the fresh branch reproduces the benchmark's tables | **pass**, all 16 cells equal | 8,008 KITTI and 34,149 nuScenes frames each |

## Deviations

| what | why |
|---|---|
| Scene-0162 (new test) re-detected in every mode after one truncated CAM_FRONT image was re-fetched | an interrupted copy left one partial file; found by comparing all 34,149 file sizes with the source |
| Fine-tuning environment: `np.trapz` aliased to `np.trapezoid` | Ultralytics 8.3.40 calls `np.trapz`, which numpy 2 removed; same integral, recipe unchanged |
| E2 development set: the tokens of the 10 OpenScene mini logs that also belong to navtest (2,086 tokens) are excluded from allocator fitting | the addendum assumed disjoint sets |
| E6 amendment 1 (before any value): the GPU instead of the edge CPU; seed 0 before every planner call; one A/A call with seed 1 | one planner call did not finish in two hours on the edge CPU; the released planner draws fresh noise at every call |
| E6 amendment 2 (before any value): the 180 states whose history reaches before the scenario start failed in every branch on duplicated padding frames; those frames now come from the logged data before the start, and only those states were rerun | an input-construction defect, not a planner failure; the first run's 1,260 states are reported beside the primary rows |
| E7 amendment 1 (before any value): VoxelNeXt voxel 0.075 replaces CenterPoint voxel 0.075 as the full detector; TransFusion-L as a secondary full | the registered checkpoint and CenterPoint voxel 0.1 can no longer be downloaded from OpenPCDet's model zoo |
| E7 environment (no value involved): av2 and kornia 0.6.12, numpy's removed aliases restored in 210, torch_scatter | OpenPCDet imports its Argoverse 2 loader unconditionally; it calls `np.int` (removed in numpy 1.24); CenterPoint-PointPillar's dynamic pillar encoder needs torch_scatter |
| E8a (before any decision value): poses interpolated where a log releases none at an image time | an adapter fix found while building the tables |
| E8a: Section 3's visibility rule (at least one LiDAR point in the box) restored after the first gate run, counted from the released sweeps (217) | the amodal labels carry no visibility; the addendum's dated change. The gate failed again and E8a stopped as registered |
| E9 (before any value): ONNX opset 16, the layer norms written as elementary operations | the Jetson's TensorRT 8.5 cannot import opset 17's LayerNormalization |
| E9 (before any decision value): transformers 4.45.2, asserted in 213; export, engines, reference, inference and gate redone | gate E9-0's first run showed that 4.44.2 built the relative head for the metric checkpoints |
| E9 (before any decision value): Large's engines hold the layer norms in FP32 (224); Small keeps its FP16 engines | gate E9-0's second run passed Small and failed Large; held in FP32, the same gate boxes differ by 0.03% |

## Points the protocol left open, and the reading taken

| question | reading taken |
|---|---|
| "Ego-speed tercile (terciles of the pooled 850 scenes)": scene-level or frame-level speed | frame level: terciles of the controller's ego speed over all frames of the 850 scenes; cut points saved with the run |
| "budget_binding" in x1_settings | true when more frames have V > 0 than the 20% capacity allows |
| E5 planner rows need rasters per detector | run at the primary operating point only (the pairs of X7a and the two fine-tuned pairs); the other operating point (X7b) has braking rows only |
| X2 before the pixel router was trained on the follow-up scenes | X2 / X3 were first written without R2 and then rewritten with it; the shipped files include R2 |
| E2: NAVSIM v1.1 has no PDM-Closed agent, so no official per-token PDM-Closed run exists to compare with | E2-0 compares the reference branch three ways (v1.1 at the addendum's 0.5 s handover, v1.1 at 0.1 s, v1.0's scorer) on a 1-in-12 navtest sample |
| E2: the validation-selected comparison needs a perception objective, which the addendum does not define for NAVSIM | the decision-objective side only |
| E2 and E6: harm ratio below 30 affected units | undefined, with the count (both addenda) |
| E6: Diffusion Planner's input is 21 observations at 0.1 s; the track defines branch observations on the last four 20 Hz frames | the branch replaces the logged tracks on the history frames inside that span; older frames keep the logged tracks in every branch |
| E7 planner rows: operating points, geometry and field of view | the shared operating point only (as E5's planner rows); one geometry, since LiDAR boxes are metric; the addendum's field-of-view rule and class set, as for the braking rows; E1's 228 new test scenes |
| E8b: a landscape input (for example a 5:3 band of the portrait image around the horizon) would likely lift recall above the gate | not run: it changes the registered input after seeing the gate |
| E9: used as released, both depth networks read ranges about 40–50% longer than the reference on nuScenes | not run: a focal-length rescaling or scale fit would change the registered estimator after seeing the result |
