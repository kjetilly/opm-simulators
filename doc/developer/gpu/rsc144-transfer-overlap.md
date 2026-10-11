# RSC_144 GPU transfer scheduling

The changes are measured on the complete, unmodified
`/workspaces/opm/spe11/RSC_144/deck/RSC_144.DATA` case: 3,027,456 active
cells, 26 report steps, and 72.2 simulated days. GPU properties, TPFA assembly,
linear solves, and resident Newton updates are enabled.

## Changes

- Pipeline full intensive-quantity downloads through two pinned regions of
  16,384 cells each. After one chunk completes, queue the next download before
  copying the completed chunk's fields into the CPU cache. Alternating regions
  keep the DMA destination disjoint from the CPU reader. Every download has
  completed when the materialization call returns.
- Reuse pinned storage for convergence factors, relative-change contributions,
  and rock-compaction data. Return a read-only span instead of allocating and
  initializing a new pageable vector for every query. A returned span expires
  at the next compact query or bridge reset.
- Skip advective energy calculations for zero mass-rate sources before
  requesting their intensive quantities. Explicit heat-rate sources retain
  their independent contribution. Preserve the original conversion to the
  simulator's scalar type, including single-precision instantiations.

The property staging allocation falls from **2,203,987,968 bytes to
23,855,104 bytes** on this grid. Compact staging adds 72,658,944 reusable pinned
bytes. Registrations are released before their backing vectors are resized or
reset, and bridge destruction waits for outstanding device work.

## Environment and reproduction

Measurements use an AMD Radeon RX 9070 XT (`gfx1201`) and an AMD Ryzen 9 5950X,
with one MPI rank. The HIP translation unit has no OpenMP compiler flag, and
Flow reports one simulation thread; the exact command below records the thread
option used throughout.

The starting source revision is `9417eb1ea`, which already includes the
[GPU bridges in PR #7462](https://github.com/OPM/opm-simulators/pull/7462)
and earlier bulk-transfer improvements. The initial `build_release` cache
actually lacked optimization flags and enabled `-pg`. Both the source baseline
and candidate were rebuilt with `-O3 -DNDEBUG`, `-march=znver3 -mtune=znver3`,
and no `-pg`. Timing comparisons below therefore isolate the source changes;
they do not claim a measured speedup from correcting the build flags.

The installed Hypre has no GPU backend, so GPU DILU is used for every measured
run. The runtime requires `--matrix-add-well-contributions=true` for gpuISTL.

```sh
/workspaces/opm/build_release/opm-simulators/bin/flow_gpu \
  /workspaces/opm/spe11/RSC_144/deck/RSC_144.DATA \
  --experimental-compute-properties-on-gpu=true \
  --linear-solver-accelerator=gpu \
  --linear-solver=dilu \
  --matrix-add-well-contributions=true \
  --threads-per-process=16 \
  --output-extra-convergence-info=steps,iterations,performance \
  --output-dir=/path/to/isolated/output
```

## Repeated unprofiled timings

Runs are sequential, with identical numerical and output settings and no
concurrent compilation or other benchmark run. Times are seconds. The two baseline
runs use the unchanged source; the two optimized runs include all three
transfer changes.

| Run | Baseline simulation | Optimized simulation | Baseline process wall time | Optimized process wall time |
| --- | ---: | ---: | ---: | ---: |
| 1 | 149.05 | 145.99 | 214.18 | 209.70 |
| 2 | 148.46 | 145.89 | 212.80 | 208.25 |
| Median | **148.76** | **145.94** | **213.49** | **208.98** |

Median simulation time is **1.9% lower** (2.82 s), and median full-process
wall time is **2.1% lower** (4.51 s). Both optimized simulation times are below
both baseline times. This is a modest, case-specific improvement, based on two
runs per version.

| Simulation component, median | Baseline | Optimized |
| --- | ---: | ---: |
| Assembly | 32.03 | 30.15 |
| Linear solve | 45.47 | 45.12 |
| Properties/update | 7.36 | 7.25 |
| Pre/post step | 53.56 | 53.38 |

The clearest component improvement is assembly, where zero-rate sources avoid
unnecessary CPU enthalpy evaluation and synchronized property downloads. The
pre/post difference is smaller than its run-to-run variation; the trace proves
copy/CPU overlap and the allocation reduction, but these timings do not establish
a separate pre/post speedup. DILU kernel autotuning remains enabled and can
choose different block sizes between runs.

All four runs complete 28 timesteps, 84 linearizations, 56 Newton iterations,
and 685 linear iterations, without wasted iterations. An intermediate run with
only chunking and compact pinning took 147.69 s; that single measurement is
insufficient to establish an independent runtime benefit for those changes.

## rocprofv3 evidence

The complete baseline and optimized runs were traced with:

```sh
rocprofv3 --hip-runtime-trace --memory-copy-trace --kernel-trace --stats \
  --output-format csv pftrace --output-directory /path/to/profile \
  --output-file run -- flow_gpu [the options above]
```

The optimized trace contains **5,520** intervals in which a pinned property
copy overlaps the CPU overlay of the preceding chunk, totaling **3.419 s**.
The corresponding baseline overlap is zero. These intervals are measured from
memory-copy timestamps intersected with the host interval between
`hipMemcpyAsync` returning and the following `hipStreamSynchronize` call.
Those API boundaries bracket the CPU overlay in the chunk loop.

| Full-run trace metric | Baseline | Optimized |
| --- | ---: | ---: |
| Source-cell property downloads | 9,856 | 2,640 |
| Source-cell property bytes | 7,175,168 | 1,921,920 |
| Time inside `hipStreamSynchronize` | 2.754 s | 0.855 s |
| Time inside `hipHostRegister` | 0.199 s | 0.004 s |
| Traced D2H copy duration, summed | 3.069 s | 4.163 s |
| D2H copy overlap with CPU overlay | 0 s | 3.419 s |
| Profiled simulation time | 155.71 s | 152.04 s |

Chunking increases the sum of copy durations, but permits concurrent CPU work.
Total simulation time is the acceptance criterion. The much larger cumulative
`hipMemcpyAsync` API duration also contains waits for preceding solver kernels
and must not be interpreted as pure transfer time. The installed rocprofv3 CSV
memory-copy schema omits byte counts; byte counts above come from Flow's bridge
counters.

## Validation

The final rebuilt executable completed an additional full-deck verification in
**145.65 s simulation time** and **208.61 s process wall time**. Its restart
file, timestep history, and per-timestep iteration counts exactly match the
baseline. The executable in `build_release/opm-simulators/bin/flow_gpu` matches
the archived `flow_gpu.final` SHA-256 recorded in `verification.json`.


- HIP build and focused Boost tests pass:
  `TestProductionGpuDispatcherContract` and
  `TestProductionGpuDispatcherChunkedDownload`.
- The larger test uses a 33 × 33 × 33 grid, spans more than two chunks, and
  includes a partial final chunk. It poisons and re-downloads the host cache
  twice and checks interleaved compact queries against CPU properties.
- The optimized full-run unified restart file is byte-for-byte identical to
  the baseline, including all 27 saved states. `compareECL -t SMRY` passes with
  zero absolute and relative tolerances for all 64 compared vectors; the
  comparator excludes its five standard performance vectors.
- Both complete profiled runs perform 28 timesteps, 84 linearizations,
  56 Newton iterations, and 685 linear iterations, with no wasted iterations.
- ROCm clang-tidy runs with the repository `.clang-tidy`. Existing warnings in
  unrelated validation and helper code remain; the added test complexity was
  removed by factoring the checks into a helper.

CUDA, multiple MPI ranks, and other solver backends were not exercised.

## Local artifacts

`/workspaces/opm/build_release/transfer_optimization` contains exact executables,
commands, logs, timings, and analysis scripts. In particular:

- `flow_gpu.baseline`: optimized build of the unchanged source.
- `flow_gpu.optimized`: implementation used for the timing pairs.
- `flow_gpu.final`: final rebuilt executable, also installed as
  `build_release/opm-simulators/bin/flow_gpu`. The final cleanup makes the
  original scalar conversions explicit and explicitly deletes already-disabled
  move operations; it does not alter the measured double-precision path.
- `CMakeCache.original.txt` and `flow_gpu.original`: original build configuration
  and executable, preserved before correcting the optimization flags.
- `baseline_profile_full/profile/*_results.pftrace` and
  `optimized_profile_full/profile/*_results.pftrace`: complete Perfetto timelines.
- `transfer_overlap.html`: interactive detail of actual copy/CPU overlap from
  the optimized rocprofv3 trace.
- `verification.json`: output hashes, identical-history checks, and final
  executable hashes.
- `run_case.py`, `analyze_trace.py`, `summarize_runs.py`, and `timings.json`:
  reproducible run and analysis records.
