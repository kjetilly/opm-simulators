# RSC_144 transfer tuning with one minimum Newton iteration

Measurements use the complete, unmodified RSC_144 deck (3,027,456 active cells,
26 report steps, 72.2 days) with GPU properties, assembly, and DILU. The exact
requested settings, including disabled storage caching, are recorded in each
run's `run.json` under `/workspaces/opm/build_release/transfer_tuning`.

## Implementation

- Download intensive quantities through two pinned 16,384-cell staging regions.
  While the next region is filled by DMA, the CPU publishes the completed region
  into its property cache. The full slot is synchronized before the call returns.
- Keep compact convergence and relative-change downloads in reusable pinned
  storage. Callers receive a read-only span valid until the next compact query
  or bridge reset; all existing consumers finish before that boundary.
- Pin both persistent primary-variable import mirrors. Uploads use pinned memory,
  and downloads overlap 65,536-cell transfers with conversion of the preceding
  chunk into the host solution. Each chunk occupies disjoint storage.
- Skip advective energy calculations for zero mass-rate sources, avoiding their
  individual property downloads. Explicit heat-rate sources are still applied.
- Skip clearing the unused CPU matrix and residual before a full GPU assembly
  that supplies the GPU solver directly. GPU assembly overwrites the entire host
  residual at its download boundary. CPU/fallback assembly, local solves, and
  contributions from auxiliary modules other than the empty well module retain
  their host clearing. The initial host system is still initialized normally.

The property staging allocation decreases from 2,203,987,968 to 23,855,104 bytes.
Compact staging uses 72,658,944 bytes. The existing two primary-variable mirrors
now pin 290,635,776 bytes, making the bridge's combined pinned staging footprint
387,149,824 bytes on this case. Registrations are released before their vectors
are resized or destroyed; asynchronous transfers finish before their storage is
reused by the CPU or released.

The supported property/assembly/solver bridges already retain the numerical
Jacobian and Newton correction on the GPU. This change preserves those device
handoffs. Clearing the large unused CPU matrix was additional host memory traffic
on every linearization; removing it targets total simulation time as well as
transfer scheduling.

## Reproduction and baseline

```sh
cd /workspaces/opm/build_release
./opm-simulators/bin/flow_gpu \
  --experimental-compute-properties-on-gpu=true \
  --matrix-add-well-contributions=true \
  --output-dir=/path/to/isolated/output \
  --newton-min-iterations=1 \
  --enable-storage-cache=false \
  --threads-per-process=16 \
  --linear-solver-accelerator=gpu \
  --linear-solver=dilu \
  /workspaces/opm/spe11/RSC_144/deck/RSC_144.DATA
```

Hardware: AMD Radeon RX 9070 XT (`gfx1201`), AMD Ryzen 9 5950X, one MPI rank.
The existing HIP build reports one simulation thread even with the requested
thread option: its HIP translation units have no OpenMP compiler flag. Compiler
settings are held fixed: `-O3 -DNDEBUG -march=znver3 -mtune=znver3`.
ROCm reports version 10.0.0; rocprofv3 reports 1.3.5.

The initial tracked source was revision `9417eb1ea`, including the
[PR #7462 GPU bridges](https://github.com/OPM/opm-simulators/pull/7462).
The originally installed executable contained earlier local transfer changes
absent from that source. Both baselines are archived and measured:
`flow_gpu.baseline` is rebuilt from the unchanged source; `flow_gpu.original`
is the executable originally installed. `flow_gpu.optimized` measures the transfer
changes before removing the unused host clears; `flow_gpu.final` is the final build.

All timed runs are sequential, without concurrent compilation, static analysis,
unit tests, or other benchmark runs. Wall time includes setup and output.

## Measurements

Times below are seconds; each median is from two full unprofiled runs.

| Version | Simulation run 1 | Simulation run 2 | Median simulation | Median process wall |
| --- | ---: | ---: | ---: | ---: |
| Rebuilt source baseline | 112.64 | 112.11 | 112.38 | 179.06 |
| Originally installed executable | 112.37 | 112.60 | 112.49 | 178.38 |
| Transfer changes, including primary pinning | 111.41 | 112.46 | 111.94 | 175.70 |
| Final, including skipped host clears | **108.27** | **109.11** | **108.69** | **174.12** |

Relative to the originally installed executable, median simulation time falls
**3.4% (3.80 s)** and median process wall time falls **2.4% (4.26 s)**.
Relative to the rebuilt source baseline, the reductions are **3.3%** and **2.8%**.
Both final simulation times and both final process wall times are below all
corresponding baseline measurements. This is a case-specific result from two runs
per version, with DILU autotuning enabled.

| Component, median | Source baseline | Originally installed | Transfer changes | Final |
| --- | ---: | ---: | ---: | ---: |
| Assembly | 25.48 | 24.14 | 24.76 | **21.48** |
| Linear solve | 10.79 | 11.03 | 10.93 | 11.06 |
| Properties/update | 4.90 | 4.83 | 4.79 | 4.80 |
| Pre/post step | 60.34 | 62.46 | 61.36 | 61.20 |

The clearest additional improvement is assembly: avoiding unused host clears
reduces it by about 3.28 s relative to the transfer-only candidate. The
transfer-only simulation-time difference overlaps run-to-run variation; these
measurements do not establish a separate simulation-time benefit for primary
pinning or chunked property downloads. Transfer-only process wall time was lower
in both runs. Chunking also substantially reduces staging memory and permits
measurable CPU/DMA overlap.

All unprofiled runs finish 28 timesteps, 56 linearizations, 28 Newton iterations,
and 3 linear iterations, with no wasted iterations. Full records are in
`timings.json`; correctness checks are in `verification.json`.

## Profiling

Full simulations are traced with:

```sh
rocprofv3 --hip-runtime-trace --memory-copy-trace --kernel-trace --stats \
  --output-format csv pftrace --output-directory /path/to/profile \
  --output-file run -- flow_gpu [the options above]
```

| Full-run trace metric | Source baseline | Final |
| --- | ---: | ---: |
| Source-cell property downloads | 4,928 | 1,320 |
| Source-cell property bytes | 3,587,584 | 960,960 |
| Summed D2H copy duration | 2.922 s | 4.159 s |
| D2H overlap with CPU publication intervals | 0 s | **3.699 s** |
| Overlapping intervals | 0 | **6,808** |
| Time inside `hipStreamSynchronize` | 2.635 s | 0.701 s |
| Time inside `hipHostRegister` | 0.052 s | 0.017 s |
| Profiled simulation time | 114.41 s | 111.46 s |

Chunking increases the sum of DMA durations while allowing CPU work to proceed.
The overlap metric is not an estimate of elapsed time saved; acceptance uses the
unprofiled end-to-end timings above. Full property and primary-variable payloads
are preserved, and Newton correction downloads remain zero.

The CSV memory-copy schema does not include byte counts; transferred byte counts
come from the bridge's counters. HIP API durations include waits for preceding
GPU work and must not be interpreted as pure transfer time. CPU publication
intervals are bounded by the asynchronous copy's API return and the subsequent
stream synchronization call. The intersection of those intervals with DMA
intervals quantifies overlap and includes a small amount of API-boundary overhead.

`transfer_overlap.html` presents actual copy/CPU timestamps interactively;
`baseline_profile/profile/*_results.pftrace` and
`final_profile/profile/*_results.pftrace` are full Perfetto timelines.

## Validation

The focused Boost tests cover small transfers, repeated property downloads,
reused compact buffers, primary-variable values and phase meanings in both time
slots, and a 53 × 53 × 53 grid spanning multiple chunks with a partial final chunk.
Full-deck checks compare restart bytes, non-performance summary vectors with
zero tolerances, timestep history, and Newton/linear iteration counts.

Both final unprofiled runs and the final profiled run have **byte-for-byte
identical unified restart files** to the source baseline. `compareECL -t SMRY`
passes with zero absolute and relative tolerances: 64 vectors checked, with the
five standard performance vectors excluded. Timestep and iteration histories
also match exactly. The installed executable's SHA-256 matches `flow_gpu.final`
in `verification.json`.

Both focused Boost tests pass (4,020,432 assertions). A 27-cell GPU-assembly
case using the CPU ILU0 solver also completes with the same iteration counts as
the source baseline. The HIP build introduces no compiler warnings. ROCm
clang-tidy completes with the workspace `.clang-tidy`; the new helpers have no
warnings. Existing complexity, C-array, conversion, and special-member warnings
remain in the surrounding implementation and test utilities.

CUDA, multiple MPI ranks, and GPU hardware other than the RX 9070 XT were not
exercised. The returned compact spans are intentionally temporary views.

## Artifacts

All benchmark artifacts are under `/workspaces/opm/build_release/transfer_tuning`:

- `run_case.py`, per-run `run.json`, and `console.log`: exact commands and timings.
- `timings.json`, `verification.json`, `verify_runs.py`: measurements and output checks.
- `flow_gpu.baseline`, `flow_gpu.original`, `flow_gpu.optimized`, `flow_gpu.final`:
  preserved executables for every measured version.
- `baseline_profile/profile/` and `final_profile/profile/`: rocprof CSV and Perfetto traces.
- `analyze_trace.py`, `make_timeline.py`, `transfer_overlap.html`: overlap analysis and view.
- `build_final.log`, `boost_final.log`, `fallback.json`, and `*.tidy.log`: validation records.
- `implementation.patch`: source changes relative to the initial revision.
