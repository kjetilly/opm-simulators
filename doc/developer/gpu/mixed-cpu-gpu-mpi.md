# Mixed CPU/GPU MPI simulation

`flow_gpu` and `flow_gpu_cpu` can participate in the same MPI simulation.
Each rank owns a grid partition. GPU ranks evaluate properties, assemble the
reservoir equations, and solve their part of the linear system on the GPU.
CPU ranks use the host implementations, including OpenMP when enabled.
The CPU companion inherits the GPU model's gas-water thermal physics, including
its disabled diffusion and dispersion. Do not substitute the general `flow`
executable without checking that its model configuration matches.

Build both executables in the configured release build:

```sh
cd /workspaces/opm/build_release
ninja flow_gpu flow_gpu_cpu test_GpuOwnerOverlapCopy test_GpuSparseTable \
      test_blackoilintensivequantities_gpu test_networkpressure
ctest -R '^(GpuOwnerOverlapCopy|GpuOwnerOverlapCopyMixed|GpuSparseTable|GpuPropertyBatches|GpuSolutionPinLifetime|networkpressure)$' --output-on-failure
```

Use the rank launcher below, or an MPI MPMD launch with colon-separated
executable groups. All ranks must
use the same deck, output directory, timestep/convergence options, Krylov solver,
and scalar precision. The tested linear solver is BiCGSTAB with local DILU.
Use `--gpu-aware-mpi=false`: the GPU rank stages halo vectors through host memory
and uses the same Dune owner/copy exchange as CPU ranks. Device-direct exchange
and mixed CPR/AMG hierarchies are outside this implementation's validated scope.

For a machine with 16 physical cores, launch one GPU rank and 15 CPU ranks:

```sh
cd /workspaces/opm/build_release
deck=/workspaces/opm/spe11/RSC_144/deck/RSC_144.DATA
output="$PWD/mixed-mpi-output"
common=("$deck" --matrix-add-well-contributions=true --newton-min-iterations=1
        --enable-storage-cache=false --linear-solver=dilu --gpu-aware-mpi=false
        "--output-dir=$output")
UCX_TLS=sysv,self CPU_THREADS=1 mpirun --bind-to none -np 16 \
  ../opm-simulators/examples/mixed_cpu_gpu.sh "${common[@]}"
```

For one GPU rank and one CPU rank with 15 threads, use `CPU_THREADS=15` and
`-np 2`. The script selects `flow_gpu` on rank zero and `flow_gpu_cpu` on other
ranks, and unsets `OMP_NUM_THREADS` so the requested thread counts take effect.
Set `FLOW_BINARY_DIR` if the binaries are elsewhere. This also works around the
installed MPI launcher's crash with colon-separated MPMD commands. On this
container, `UCX_TLS=sysv,self` avoids a UCX initialization failure with the 64 MiB
POSIX shared-memory mount; that setting is for a single-node run only.
Bind the GPU process to one physical core and the CPU process(es) to the other
cores for reproducible measurements. The normal
partitioner balances cells per MPI rank, not estimated CPU/GPU execution time;
`--external-partition` can supply a different cell distribution.

The CPU companion is compiled with the host compiler because GPU translation
units need not enable OpenMP. In particular, the HIP configuration used here
does not parallelize host assembly inside a HIP translation unit.

## Implementation and regression coverage

Property-update errors are reduced collectively after either the CPU or GPU
path completes. Previously the GPU path returned before the CPU collective.
CPU linear-solver setup now writes its local diagnostic directly, avoiding a
log-gather collective when only CPU ranks rebuild their solver. These changes
allow the existing CPU and GPU Krylov implementations to communicate together.
The GPU solver pins each Newton solution vector only for the duration of its
solve. The previous persistent registration outlived the caller's temporary
vector and caused invalid host-to-device transfers when its address was reused
in the large two-rank layout. `GpuSolutionPinLifetime` checks repeated solves
and confirms that the solution registration is released before returning.

Thermal boundary transmissibilities use sparse per-cell rows, accommodating
partition cells with more than three boundary faces. The GPU boundary kernel
now calls the GPU residual implementation unconditionally: testing the host
fluid-system type had silently skipped all thermal boundary contributions,
which made CPU/GPU partitions solve different equations. `GpuSparseTable`
checks GPU transfer and access for empty interior rows and a six-face cell.

Property buffers and assembly neighbor metadata are staged in 65,536-cell
batches. This bounds temporary device memory when the GPU solver and its
preconditioner are resident; full-grid staging exceeded this card's 16 GiB on
RSC_144. Assembly retains complete intensive-quantity views for halo access.
`GpuPropertyBatches` checks the production dispatcher on a 68,921-cell grid
with varying porosity, crossing a batch boundary and including a partial final
batch. The offset must refer to the original local cell when fetching properties.

`GpuOwnerOverlapCopyMixed` exercises three-component halo exchanges, dot
products, and norms with one GPU rank and CPU peers. It checks both rank zero
and the last rank as the GPU participant. Full simulation validation and
benchmark measurements are recorded below.

## Validation environment

The local build combines `opm-simulators` at `4f4603ba8`, `opm-common` at
`a9121b318`, and `opm-grid` at `4888e311`. Rebuilding against these dependency
versions also required adapting the moved `SparseTable` include, `BCState`,
satellite-production accessors, summary-state initialization, and the new
`WELDRAW` serialization include. The fmt headers now explicitly include
`fmt/format.h` wherever `fmt::format` is used.

The validation machine has a Ryzen 9 5950X (16 physical cores, 32 logical CPUs)
and one Radeon RX 9070 XT (`gfx1201`). The Release build uses GCC 15 for host
code and the ROCm Clang 23 HIP compiler for GPU code. Tests use double precision,
host-staged MPI exchanges, and BiCGSTAB/DILU. CUDA and multiple-node execution
have not been tested here.

A shortened copy of the RSC_16 deck (six blocks of five 3.6-day report steps,
108 days total, 4,608 cells) completes with CPU-only, GPU-only, GPU + 15 CPU
processes, and GPU + one 15-thread CPU process. With
`--linear-solver-reduction=1e-10 --relaxed-linear-solver-reduction=1e-10`,
`compareECL` passes for `PRESSURE`, `SWAT`, `SGAS`, and `TEMP` at every report
step using absolute and relative tolerances of `1e-5` against the CPU-only run.
These tighter solver options are for validation, not the timing runs below.

Focused CTest coverage: `GpuSparseTable`, `GpuOwnerOverlapCopy`,
`GpuOwnerOverlapCopyMixed`, `GpuPropertyBatches`, `GpuSolutionPinLifetime`,
and `networkpressure` all pass. Clang-tidy was run with the workspace
configuration; the new MPI and batch regressions and CPU companion pass the
targeted checks. Existing warnings remain in the original sparse-table test, the
large `linearize_cell` function, and an ISTL explicit-instantiation macro;
GCC also emits its existing Dune
`-Wfree-nonheap-object` diagnostics.

The broader `blackoilintensivequantities_gpu` CTest also was attempted. Its
existing hard-coded real-deck fixture requests diffusion from a disabled model,
and its derivative fixture reports NaNs on both CPU and GPU. Those unrelated
fixtures fail; the two production-dispatcher cases pass when run separately.

The checked-in `examples/mixed_cpu_gpu.sh` launcher also completes the small
case with 15 CPU threads. Its final results pass the same four-field `1e-5`
comparison against the CPU reference.

Benchmark affinity is requested as CPU 0 for the GPU rank, CPUs 1–15 for the
threaded CPU rank, and CPU `rank` for each single-thread CPU rank. `/proc`
inspection shows that the HIP runtime expands some thread masks to CPUs 0–31;
these runs use the runtime's default affinity behavior. The CPU OpenMP workers
retain CPUs 1–15. The two-rank CPU process grows from four runtime threads to
18 threads on entering simulation, consistent with 14 additional OpenMP workers
plus its main thread. The rank-zero log's “threads per process” line describes
the GPU rank and does not report the CPU rank's separate thread setting.

## Benchmark method

The full, unmodified input is
`/workspaces/opm/spe11/RSC_144/deck/RSC_144.DATA`: 3,027,456 active cells,
26 report steps, and 72.2 simulated days. Runs use the options from
`/workspaces/opm/toprofile.sh`, plus explicit host-staged MPI, convergence
history output, and separate output directories. All final measurements run
sequentially without concurrent builds, tests, static analysis, or other
benchmarks. Each final layout is measured once; the results are not medians.
Wall time includes input, partitioning, initialization, simulation, output,
and process shutdown.

The two-rank layout assigns 1,665,100 owned cells to the GPU and 1,362,356 to
the CPU. In the 16-rank layout, the GPU owns 177,976 cells. These are the
default partitioner's distributions, with no CPU/GPU performance weighting.

The executable preserved before this task is
`build_release/mixed_cpu_gpu/flow_gpu.baseline`. It fails this full case with
device-memory allocation errors and has no completed benchmark time. The
older `rsc144-transfer-*` notes describe different archived binaries and source
revisions; their timings are not measurements of this change.

Exact commands, rank-specific arguments, logs, binary hashes, and timings are
under `/workspaces/opm/build_release/mixed_cpu_gpu`. The final run directories
are `rsc144_pinned_gpu`, `rsc144_pinned_mixed_mpi`, and
`rsc144_pinned_mixed_threads`; `rsc144_original_gpu` records the failed original
executable. `collect_results.py` produces `benchmark_results.json`.

## RSC_144 results

Times are seconds, using 16 physical CPU cores and the Radeon RX 9070 XT.

| Layout | Setup | Simulation | Total elapsed |
| --- | ---: | ---: | ---: |
| GPU only, rebuilt source | 52.04 | 481.27 | 539.79 |
| One GPU rank + 15 single-thread CPU ranks | 86.81 | 134.58 | 224.49 |
| One GPU rank + one 15-thread CPU rank | 98.65 | 325.73 | 429.73 |

The 15-CPU-rank layout is 2.40 times faster end to end than the rebuilt GPU-only
run, and 1.91 times faster than the threaded CPU layout. Its simulation phase
is 3.58 times faster than GPU only. All three finish 28 timesteps,
56 linearizations, and 28 Newton iterations with no wasted iterations. The
GPU-only and threaded layouts report three linear iterations; the 16-rank
layout reports four. These are measurements at the requested original solver
reduction of `0.01`, with `--newton-min-iterations=1`.

| Simulation component | GPU only | GPU + 15 CPU ranks | GPU + 15 CPU threads |
| --- | ---: | ---: | ---: |
| Assembly | 370.12 | 55.50 | 219.19 |
| Linear solve | 15.70 | 17.87 | 23.51 |
| Properties/update | 35.42 | 25.57 | 22.78 |
| Pre/post step | 50.13 | 20.72 | 33.80 |

Both mixed runs pass `compareECL` for `PRESSURE`, `SWAT`, `SGAS`, and `TEMP`
at every saved report step against the rebuilt GPU-only run, with absolute
and relative tolerances of `1e-3`. The largest differences across these runs
are 0.02887 bar in pressure, 0.01299 degrees Celsius in temperature, and
0.000894 in saturation. Partition-dependent solves at the original `0.01`
linear reduction do not reproduce all fields at `1e-5`; the small-case
validation above uses tighter solves and passes that stricter comparison.
`validation_results.json` and the `*_final_validation.log` / `*_final_analysis.log`
files contain the comparison results. Analysis maxima there describe only
entries exceeding both `1e-5` tolerances; null means no such entry.

The final Ninja check succeeds. It refreshes generated version information
and relinks the executables without changing simulation objects.
`benchmark_binaries.json` records both the timed binary hashes and the hashes
after that version-information refresh.
