# Mixed CPU/GPU MPI simulation

`flow_gpu` and `flow_gpu_cpu` participate in the same MPI simulation, with
each rank owning a grid partition. GPU ranks evaluate properties and assemble
reservoir equations on the GPU. CPU ranks use the host implementations and
can use OpenMP. Both executables use the same thermal gas-water model, with
diffusion and dispersion disabled.

This implementation is rebased onto `tobias/spe11cbenchmark` in all three OPM
repositories. It retains Tobias's resident property, assembly, Newton, and
output implementations, with the MPI adaptations described below.

## Build and run

```sh
cd /workspaces/opm/build_release
ninja -j4 flow_gpu flow_gpu_cpu test_GpuOwnerOverlapCopy test_GpuSparseTable \
      test_blackoilintensivequantities_gpu test_networkpressure compareECL
ctest -R '^(GpuOwnerOverlapCopy|GpuOwnerOverlapCopyMixed|GpuSparseTable|GpuPropertyTransfers|GpuSolutionPinLifetime|networkpressure)$' --output-on-failure
```

For one GPU rank and 15 single-thread CPU ranks on a 16-core machine:

```sh
cd /workspaces/opm/build_release
UCX_TLS=sysv,self HSA_OVERRIDE_CPU_AFFINITY_DEBUG=0 CPU_THREADS=1 mpirun --bind-to none -np 16 \
  ../opm-simulators/examples/mixed_cpu_gpu.sh prepared-case.DATA \
  --matrix-add-well-contributions=true --newton-min-iterations=1 \
  --enable-storage-cache=false --linear-solver=dilu --gpu-aware-mpi=false \
  --output-dir=mixed-mpi-output
```

For one GPU rank and one CPU rank with 15 threads, use `CPU_THREADS=15` and
`-np 2`. The launcher chooses `flow_gpu` on rank zero and `flow_gpu_cpu` on
other ranks. It clears `OMP_NUM_THREADS` so the rank-specific thread setting
takes effect. `FLOW_BINARY_DIR` overrides the executable directory.
`UCX_TLS=sysv,self` is a single-node workaround for this container's small
POSIX shared-memory mount. The rank launcher also avoids a crash in the
installed MPI launcher's colon-separated MPMD syntax.
The ROCm affinity setting makes runtime threads inherit the parent CPU mask;
the benchmark launcher supplies a separate mask for each MPI rank and records
runtime affinity samples.

All ranks must use the same deck, output directory, scalar precision,
timestep controls, convergence options, and Krylov solver. The tested solver
is double-precision BiCGSTAB with local DILU and host-staged halo exchanges.
Mixed CPR/AMG and device-direct MPI have not been validated here.

## MPI adaptations

Property evaluation and local convergence calculations enter the same error
collectives on CPU and GPU ranks. A GPU return before the CPU collective
would otherwise leave the MPI ranks executing different reductions.
The resident GPU rollback after a rejected timestep also matches the CPU
property-update error collective before either rank starts the retry.

GPU assembly and linear-solver selection do not perform GPU-only well-count
reductions. The resident matrix-to-solver path is retained for single-rank
execution. MPI uses host preparation so overlap matrix rows are replaced
with identity rows before GPU solver upload, as in the existing parallel
solver. Properties and assembly still run on the GPU. Tobias's resident
Newton update already restricts itself to single-rank execution; MPI uses
the existing host Newton update and halo synchronization.

CPU solver setup writes its local diagnostic through OpmLog without a
collective log gather. CPU and GPU ranks can rebuild their local solvers at
different times. GPU solution memory is pinned only for the duration of the
solve, so a registration cannot outlive the caller's temporary vector.

GPU thermal boundary fluxes use the GPU residual implementation regardless
of the enclosing host TypeTag's fluid system. Tobias's six-face thermal
boundary representation is retained. His persistent GPU buffers replace the
older per-iteration batching implementation.

The CPU companion is built with the host compiler so its property evaluation
and assembly use OpenMP. The default partitioner balances cells, without
CPU/GPU speed weights. An external partition can provide another distribution.

## Regression coverage and benchmark artifacts

`GpuOwnerOverlapCopyMixed` checks halo exchanges, dot products, and norms with
CPU peers, with both the first and last rank as the GPU participant.
`GpuPropertyTransfers` checks the production resident-state dispatcher on a
small grid and a 68,921-cell grid with varying porosity. `GpuSolutionPinLifetime`
checks that repeated solves release solution-vector registrations.

The experiments use the decks in `/workspaces/opm/newcases/SPE11C_cases` and
compare CPU MPI (16 ranks), CPU OpenMP (16 threads), GPU only, GPU plus 15 CPU
ranks, and GPU plus one 15-thread CPU rank. Grids and source definitions are
preserved. Diffusion and dispersion are removed identically from prepared
copies.

The scaling study repeats the earlier 26-day workload on the new grids: all
26 report intervals are one day, so the timing of source events is compressed.
The maximum timestep is one day in every configuration. Commands, frozen
binary/source provenance, input changes, logs, numerical validation, and plots
are under `/workspaces/opm/build_release/spe11c_scaling_20261019_26days`.

Separate full-schedule experiments for RSC_16, RSC_32, and RSC_48 preserve all
730,000 days and use a maximum allowed timestep of 365,000 days with adaptive
control. A tighter full-schedule RSC_16 numerical check uses a 365-day cap.
Those results are under `/workspaces/opm/build_release/spe11c_scaling_20261019`;
larger grids were not timed for that workload. The full and abbreviated
schedules are separate datasets.

The plots show total elapsed time, simulation time, assembly, linear solve,
properties/update, and pre/post processing against active-cell count, with
individual phase plots and a four-panel overview. Timing runs are sequential,
and only grids completing GPU-only execution for the plotted workload are
included. Runtime affinity samples accompany the measurements.
The phase plots use Flow's rank-zero wall timers. GPU property evaluation is
asynchronous, so its execution can be charged to a later phase that waits for
completion.

The earlier 26-day experiments and their original binaries remain under
`/workspaces/opm/build_release/spe11c_scaling_20261013`; their results describe
the previous source and workload and should not be mixed into the new curves.

## Continuing the full schedule unattended

`examples/run_spe11c_long.py` continues the full 730,000-day experiment on this
workspace. The convenience launcher is `/workspaces/opm/run_spe11c_week.sh`.
The default output directory is `/workspaces/opm/build_release/spe11c_week`.

```sh
cd /workspaces/opm
./run_spe11c_week.sh check
./run_spe11c_week.sh start
./run_spe11c_week.sh status
tail -f build_release/spe11c_week/manager.log
```

`start` detaches from the terminal, so the shell may be closed. The machine
and container must remain running. The script uses the frozen executables
and input hashes from `spe11c_scaling_20261019`; no rebuild is required.
It copies the executables and inputs into the new experiment directory.
Keep the baseline directory: its completed runs are reused through symlinks.

By default, the script imports the 15 completed RSC_16–48 configurations and
continues with 35 configurations for RSC_64–160. It runs GPU only, CPU MPI,
CPU OpenMP, mixed MPI, and mixed OpenMP sequentially, with the same 16-core
allocation and solver options as the full-schedule study. The original
schedule and grid sizes are preserved; diffusion and dispersion remain
disabled identically in the prepared decks. RSC_176 is excluded because its
previous GPU run exhausted VRAM. If a full-schedule GPU run also exhausts
VRAM, that grid and larger grids are excluded.

There is **no wall-time limit by default**. To impose one, add, for example,
`--timeout-hours 48`. This is a per-configuration limit, not a limit on the
whole experiment. A timed-out or failed configuration is recorded and the
queue continues. If GPU-only execution fails, the other modes for that grid
are skipped. Use `--retry-failed` on a later start to retry failures once;
previous attempts are archived.

```sh
# Interrupt the current configuration and stop the queue.
./run_spe11c_week.sh stop

# Resume: completed configurations are skipped.
./run_spe11c_week.sh start

# Retry failed or timed-out configurations as well.
./run_spe11c_week.sh start --retry-failed
```

Resume operates at configuration boundaries: an interrupted simulation starts
again from its beginning, rather than from an Eclipse restart checkpoint.
The previous attempt is retained under `attempts/`. A lock prevents duplicate
runners from using the same output directory. If less than 10 GiB of disk
space remains before a run, the queue stops with an error and can be resumed
after space is available. `--min-free-gib` changes that threshold.

`status.json` records the current phase, case, elapsed runtime, and recent
simulator output; `runs/CASE/MODE/console.log` contains the complete log.
The script updates `results.csv/json` after each run and regenerates the total,
simulation, assembly, linear-solve, properties/update, and pre/post plots after
each case. Only grids with all five modes complete appear in the plots.
Each plot is saved as PNG, PDF, and SVG.

After the queue finishes, it compares pressure, saturations, and temperature
at every saved report step and retains failures in `validation_results.json`
and `validation/`. These are the original default solver tolerances, whose
full-schedule numerical differences were already observed; completing a
simulation does not imply passing those comparisons. Simulation failures and
numerical comparison results are reported separately.

For a separate smaller experiment, use `--cases 16,32 --output /path/to/results`.
Repeat those options on resume. `--no-reuse-completed` starts those selected
configurations from scratch in the new output directory; `--no-plots` and
`--no-validate` disable their respective postprocessing steps.
