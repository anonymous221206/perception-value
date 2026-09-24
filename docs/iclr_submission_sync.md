# The submission path scores the tracks the paper reports (Task 28)

What was wrong with Task 25's submission path, what changed, and the evidence that it now reproduces every table the
paper uses. No detector, planner or benchmark allocator was re-run and no number in the paper's tables changed; the
only files under `results/final/` that changed are the submission exports (G2).

## What was wrong

1. **nuPlan.** The submission cells, decision values and legal inputs for nuPlan were the transported miss-model
   track (`t92.nuplan_cells`, `benchmark_nuplan_signals.csv`, the `*_nuplan_track_lists`). The paper reports the
   real-perception track (Task 5 labels, `scripts/120_nuplan_real_allocation.py`). Same 384 test states, different
   labels: 11 / 42 / 14 / 28 affected test states in the four cells, against 27 / 48 / 4 / 10 on the real track; and
   the real track leaves nDG undefined below 10 affected states, which the path did not.
2. **Measured budgets.** Detector costs came from `benchmark_budget_overheads.json`: nuPlan charged the KITTI
   detector, energy on the old convention (GPU rail only). The paper's energy results use the module convention and
   nuPlan's own Task 5 costs.
3. **G1 compared against the wrong tables** for nuPlan and energy, so it proved less than it appeared to.

## What changed

| piece | change |
|---|---|
| `scripts/156_cost_registry.py`, `results/final/cost_registry.json` | new: the versioned cost registry (below) |
| `src/rap/submission.py` | detector costs from the registry; each track scored on its own table's bootstrap stream (`benchmark_table` for KITTI and nuScenes, `nuplan_real` for nuPlan; measured budgets `benchmark_budget`, `nuplan_real_budget`); 120's rule that nDG is undefined below 10 affected test states or at a zero oracle prize (`undefined_reason`); an energy profile must be on the registry's rails or it is refused; every scored row carries `registry_version`, `labels`, `source_run`, and on the measured track `cost_convention`, `timing_boundary`, `overhead_provenance` |
| `scripts/150_submission_tables.py` | cells, decision values and plans for nuPlan from 120's `cells` (the transported track is no longer a submission cell and is not kept under another name: nothing in the path needs it); nuPlan inputs from the real CHEAP branch (119's 18 gate features and 25-track list); gate G1' with identity assertions |
| `scripts/151_profile_allocator.py` | default rails are the registry's (GPU, SOC, CPU), not CPU |
| `evaluate_submission.py` | per-track default plan; prints the registry version and "undefined (too few affected test states)" |
| examples | `confidence_cost_profile.json` re-profiled with the harness on the module rails on the reference board (it had been measured on the CPU rail only, which the scorer now refuses); both scored outputs regenerated |
| docs | `docs/SUBMITTING.md` (nuPlan track, inputs, plans, undefined rule, registry); `docs/iclr_submission_path.md` (a note on what Task 28 replaced); `scripts/152_evidence_pack.py`'s submission paragraph |
| `reproduce.py` | new stage C30b (the registry); C31-C33 unchanged in form |
| `scripts/147_ego_frame_convention.py` | the Task 23 inventory leaves out `cost_registry.json`, a later task's file (G2) |

## The cost registry, version 2026-09-23.1

Energy convention **module**: per-frame energy = sum over the GPU, SOC and CPU rails of (busy mean - idle mean) x
per-frame time, for detectors and allocators alike (133's convention; content hash in the file).

| detector, per input | CHEAP ms | FULL ms | CHEAP mJ | FULL mJ | timing boundary | source |
|---|---|---|---|---|---|---|
| KITTI | 13.179 | 18.469 | 41.113 | 84.433 | end-to-end, image decoding included | `20260912_021305_profile` (01) |
| nuScenes | 12.710 | 19.351 | 54.554 | 132.701 | end-to-end, image decoding included | `20260912_073228_profile` (01) |
| nuPlan | 14.374 | 23.571 | 69.512 | 143.845 | pre-processing + inference + post-processing, no decoding | `nuplan_task5/detect_summary.json` |

KITTI also carries the 384 and 512 levels the multi-fidelity tables use. Allocator overheads (ms from 93's
`signal_overhead`, mJ = ms x module power over idle) are kept per overhead measurement, because the tables use three
and they differ:

| measurement | used by | gate power (mW over idle) | e.g. gate_gbm KITTI, ms / mJ |
|---|---|---|---|
| primary | `benchmark_budget_two_level.csv`, energy `table=primary`, multi-fidelity | 8574.0 | 19.977 / 171.281 |
| 1thread | energy `table=1thread` | 2567.1 | 16.777 / 43.066 |
| routers | `benchmark_budget_routers.csv`, `*_nuplan_real.csv`, energy `table=routers` | 7888.2 | 21.383 / 168.673 |

R1 and R2 are charged 132's measured module power (9982.0 and 2676.1 mW). nuPlan has no timed allocator workload:
every nuPlan entry carries the provenance "measured on nuScenes, inherited by nuPlan as a proxy (track features not
timed)", and the scored rows carry it into `overhead_provenance`. (The registry lists the proxy for every gate-family
signal, including `uncertainty`, which the nuPlan track does not have; nothing charges it there.) Superseded entries
(the old overheads file's costs, the old-convention mJ rows) are listed in the registry's `superseded` block.

## G1': the submission path reproduces the paper's tables

`scripts/150_submission_tables.py --stage g1` (run `20260923_162918_submission_g1`): every shipped deployable signal
written out as a submission, scored through `rap.submission`, compared field by field with no tolerance. Before any
number is compared, the scored rows and the canonical rows must agree on track (the registry names each table's
tracks), label run (`labels`, `source_run` against 120's `labels` / `features_run`; for KITTI and nuScenes the
exported decision values are re-derived from 92's builders and must be identical), cost convention, detector costs
and budget per input (`cheap_cost`, `full_cost`, `budget_per_frame`) and registry version; any mismatch stops the gate.

| table | signals | fields | values compared | matched | differing | structural |
|---|---|---|---|---|---|---|
| `benchmark_table.csv` (KITTI, nuScenes, test) | 5 | 16 | 3,200 | 3,200 | 0 | 0 |
| `benchmark_table_routers.csv` (core) | 6 | 13 | 2,600 | 2,600 | 0 | 0 |
| `benchmark_table_nuplan_real.csv` (test, incl. the undefined IDM-safety cell) | 8 | 19 | 2,432 | 2,416 | 0 | 16 |
| `statistics_hardening.csv`, ego speed, core (point values) | 1 | 3 | 120 | 120 | 0 | 0 |
| `statistics_hardening.csv`, ego speed, nuPlan (point values) | 1 | 3 | 48 | 48 | 0 | 0 |
| `benchmark_budget_two_level.csv` (core, ms) | 5 | 10 | 2,000 | 2,000 | 0 | 0 |
| `benchmark_budget_routers.csv` (core, ms) | 11 | 10 | 4,400 | 4,400 | 0 | 0 |
| `benchmark_budget_nuplan_real.csv` (ms) | 9 | 16 | 2,304 | 2,304 | 0 | 0 |
| `benchmark_budget_multifidelity.csv` (single-level rows, ms) | 3 | 4 | 96 | 96 | 0 | 0 |
| `energy_module_budget_two_level.csv` (core, module; primary, 1thread, routers) | 21 | 10 | 8,400 | 8,400 | 0 | 0 |
| `energy_module_budget_nuplan_real.csv` (module) | 9 | 16 | 2,304 | 2,304 | 0 | 0 |
| `energy_module_budget_multifidelity.csv` (single-level rows, module) | 6 | 4 | 192 | 192 | 0 | 0 |
| **total** | | | **28,096** | **28,080** | **0** | **16** |

Signals: random, uncertainty, criticality_cheap, gate_ridge, gate_gbm (and gate_gbm_batched where the routers
measurement charges it), ego speed, R1 x 4, R2 on KITTI and nuScenes. The rows are in
`results/final/submission_path_g1.csv`.

**Structural: what the submission path cannot reproduce, and why.**

1. *An official signal whose scores are all equal* (16 values). `R1_mlp_clf` and `R1_gbm_clf` score every nuPlan
   IDM-safety test state exactly 0. The schema scores an all-tied submission as the random allocator (closed form;
   Task 25), while 120 scored these two through the tie expectation, so the official table gives the same constant
   ranking `tie_frac` NaN (the random row) and 1.0 (these rows), and `responsive_frac` a few ulps apart. No rule that
   sees only the scores can give both. nDG, its interval and the comparison with random are undefined in that cell
   (4 affected states) on both paths, and `gain`, `prize` and every other field match. Counted as structural, not as
   matched.
2. *The multi-fidelity rows* (`X (multi-fidelity)`, both multi-fidelity tables): the allocator chooses 384, 512 or 640
   per input, which a two-level submission cannot express. Not compared.
3. *The single-level multi-fidelity rows* (`X (640 only)`): their nDG divides by the multi-fidelity oracle, not the
   two-level one; `gain`, `share_640`, `overhead` and `feasible` are compared, nDG is not.
4. *Ego speed* appears only in `statistics_hardening.csv`, whose interval is a bootstrap shared across a dataset's
   cells on the gain scale; the point values (`ndg`, `gain`, `oracle_prize`) are compared, the intervals are not.
5. *Not the paper's rows, so not compared:* the transported nuPlan rows of `benchmark_table.csv`, the router tables,
   `benchmark_budget_two_level.csv`, `benchmark_budget_routers.csv` and `energy_module_budget_two_level.csv`; the
   old-convention mJ rows of the latency tables (superseded by the `energy_module_*` tables); the `all_rails` and
   `as_specified` sensitivity rows of the energy tables; the `split = all` rows; oracle and diagnostic signals.

## G2: no official output changes

`reproduce.py --tier cached --verify` on a fresh clone passes: every compared output reproduces. Two stages
rewrite a shipped table with the same rows in another order (`benchmark_table_routers.csv`,
`paper_evidence_pack.csv`), which the verify does not count as a change. Stage C28's inventory of the ego-frame
change leaves later files out by name (`AFTER_TASK_23` in `scripts/147_ego_frame_convention.py`), now including
the cost registry.

The files that changed, all submission exports:

| file | SHA-256 before (release a35f099) | SHA-256 after |
|---|---|---|
| `data/submission_inputs/MANIFEST.json` | `23663064e4b83c54e8215d926497d926f73f6dd062023d0864d14c4ee56f3b5d` | `7362644ad750c78820d3f8f471781a94d2eecd8e59a7b5d183ee821a9a27549e` |
| `data/submission_inputs/nuplan_states.csv.gz` | `da2a6b47409523b19a29fe1ae81e715ca02ec7f87a974b2adeaed6ab929810fe` | `cf462808ee1c919824b0a70aa52a41d6b4dd3de05273273c055d77879b0dbf3a` |
| `examples/submissions/confidence_cost_profile.json` | `be3f289b743f41751737ce22a2b0442c9cb9da61e0411bfe7a93b30b4927d72e` | `df242632bd4f2c36f2a60767e50d6be02ab4021b324348c4226127f9dea97b6f` |
| `examples/submissions/confidence_scored.csv` | `65b5969c5114782cdbc401bb10aa4ed5420c29f5117db8d64d7a1ce3657022f9` | `f0ec7eb27c4f3105fc70ad330a83124ab5a999c118c99451432d2752a48ae1c6` |
| `examples/submissions/random_scored.csv` | `3515a9d0ca2f77b661ffb7a2b994c35b8584e086a01e860a759ce9fbd2c0cf0f` | `050bd1631e1d151c47763790c5222c0e047e5664dee59d81993ceecc961f36dd` |
| `results/final/benchmark_bootstrap_plans.json` | `fc17ed056b758571aed3c1bd6285b15d3675a5dc0c188dfee9465a52bf178cfc` | `343f686f4a80b7d9e3ee788e77391837ed7ef29692ef547ba0336e7c5544ecdc` |
| `results/final/benchmark_decision_values.csv.gz` | `a6b40c2fc8df5558c1e758db5a0698c8c9ea3dfefa4c050cf8f170cc56b0523e` | `b4569c0fd1807f3a2e8fbb990eb3a35be07525f7dc8c768bbef07f14362c3d1a` |
| `results/final/cost_registry.json` | (new) | `2c9a0224c2b4034c650a24efeb8c1fe5162ad86ca65cb28c776a1bcd988d6ea5` |
| `results/final/submission_path_g1.csv` | `54aea05d6e5a029f2797543677e7902e63c178d1bf30d1ca07662e3266402691` | `f693cf93890a787e5b3467f1f38c833491a4fb6194b1f2b8abb65c4bfe61736e` |

## G3: the nuPlan check

```
$ ./scripts/py evaluate_submission.py examples/submissions/random.csv --cells 'nuPlan|n/a|pdm_closed|safety' 'nuPlan|n/a|idm|safety' --out <tmp> --nboot 1000
random.csv: form ranking, 2 cells, cost registry 2026-09-23.1
  nuPlan|n/a|pdm_closed|safety           plan nuplan_real      10%: +0.063 [+0.000, +0.000] 20%: +0.128 [+0.000, +0.000] 30%: +0.191 [+0.000, +0.000] 50%: +0.319 [+0.000, +0.000]
  nuPlan|n/a|idm|safety                  plan nuplan_real      undefined (too few affected test states)  [4 affected, nDG needs 10]
  nDG, and in brackets the paired 95% interval of nDG minus random; * marks an interval excluding zero
wrote <tmp>
```

nDG@20 = 0.128 for PDM-Closed safety, as `benchmark_table_nuplan_real.csv`; IDM safety undefined.

## Stale documents

`scripts/157_doc_staleness.py` looks up every decimal a document quotes in the `results/final/` files it names, as
they were at the document's own commit and as they are now; a number found then and nowhere now has moved. It
cannot match integer counts, and a document that names no file is checked against all of `results/final/`, which
proves little, so its findings were read one by one.

In this release every report whose figures predate the ego-frame lift already says so in a note under its title,
and `docs/iclr_ego_frame_convention.md` gives each registered quantity old beside new. The two findings without
such a note are false matches: the 2.44 m in `iclr_idm_route_fix.md` is IDM's own pre-fix log deviation, and
`iclr_nuplan_real_allocation.md` quotes only the old KITTI detector cost profile, which the lift does not touch.
Nothing under `results/final/` was edited to match a document.

## Protocol gaps found

1. **Three allocator overhead measurements.** The gate's measured overhead differs between the primary, single-thread
   and routers measurements (e.g. gate_gbm on KITTI 19.98, 16.78 and 21.38 ms), and each table uses one. The registry
   keeps all three, keyed by the table that uses them; a submission is charged its own measured cost.
2. **nuPlan allocator costs are a proxy.** No allocator workload was timed on nuPlan; the nuScenes feature time
   stands in, and every nuPlan cost says so.
3. **The minimum of 10 affected states** existed only in 120. The path applies it to every cell; no KITTI or nuScenes
   test cell has fewer than 46, so it binds on nuPlan only.
4. **All-tied official signals** (G1' structural gap 1): the official tables score a constant ranking two ways.
5. **The example profile** had been measured on the CPU rail only, which the module convention does not allow; it was
   re-profiled on the module rails (decided with the task).
6. **Gate refits.** In the first G1' run the refit gate scores of one cell (KITTI oracle brake) were not bit-identical
   to the saved scores 133 charged, in the second they were; no compared value was affected either time (each table
   was compared with the scores it was built from).
