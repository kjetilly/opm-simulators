#!/usr/bin/env python3
# Copyright TODO ADD YEAR AND NAME OF AUTHOR
#
# This file is part of the Open Porous Media project (OPM).
#
# OPM is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OPM is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with OPM. If not, see <http://www.gnu.org/licenses/>.

"""Resume the full SPE11C experiment using its frozen binaries and input hashes."""

import argparse
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time

WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE = WORKSPACE / 'build_release/spe11c_scaling_20261019'
DEFAULT_OUTPUT = WORKSPACE / 'build_release/spe11c_week'
LAYOUTS = ['gpu', 'cpu_mpi', 'cpu_openmp', 'mixed_mpi', 'mixed_openmp']
STOP_REQUESTED = False
LOCK = None
POSTPROCESSING_FAILURES = []
OUT_OF_MEMORY = re.compile(r'GPU error was out of memory|hipErrorOutOfMemory|cudaErrorMemoryAllocation')


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_module(path, name):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def state(root, status, **details):
    write(root / 'status.json', dict(pid=os.getpid(), updated_utc=now(), status=status, **details))


def stopped(root):
    return STOP_REQUESTED or (root / 'STOP').exists()


def acquire_lock(root):
    inherited = os.environ.pop('SPE11C_WEEK_LOCK_FD', None)
    stream = os.fdopen(int(inherited), 'a+') if inherited else (root / '.lock').open('a+')
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        raise RuntimeError('This output directory is already running (or a simulation child is still alive).')
    return stream


def is_running(root):
    if not (root / '.lock').exists():
        return False
    with (root / '.lock').open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


def frozen_copy(source, target, expected=None):
    expected = expected or digest(source)
    if target.exists():
        if digest(target) != expected:
            raise RuntimeError(f'Frozen file changed: {target}')
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.tmp')
    shutil.copy2(source, temporary)
    if digest(temporary) != expected:
        raise RuntimeError(f'Input/binary hash differs from the original experiment: {source}')
    temporary.replace(target)


def selection(arguments):
    baseline = arguments.baseline
    metadata = read(baseline / 'environment.json')
    if metadata['schedule_modified'] or metadata['simulated_days'] != 730000:
        raise RuntimeError('The baseline must be the full 730,000-day experiment, not the 26-day study.')
    all_cases = read(baseline / 'cases.json')
    names = ({'RSC_' + value.strip().removeprefix('RSC_') for value in arguments.cases.split(',')}
             if arguments.cases else {case['name'] for case in all_cases if case['dimensions'][1] <= 160})
    unknown = names - {case['name'] for case in all_cases}
    if unknown or not names:
        raise RuntimeError(f'Invalid case selection: {sorted(unknown)}')
    if 'RSC_176' in names:
        raise RuntimeError('RSC_176 is excluded: its GPU-only run exhausted VRAM.')
    cases = sorted([case for case in all_cases if case['name'] in names], key=lambda case: case['cartesian_cells'])
    return metadata, cases


def preflight(arguments):
    metadata, cases = selection(arguments)
    for executable in ['mpirun', 'taskset']:
        if shutil.which(executable) is None:
            raise RuntimeError(f'Missing executable: {executable}')
    if not set(range(16)).issubset(os.sched_getaffinity(0)):
        raise RuntimeError('This experiment requires access to CPU cores 0 through 15.')
    for name, expected in metadata['binary_sha256'].items():
        if digest(arguments.baseline / 'bin' / name) != expected:
            raise RuntimeError(f'The baseline binary changed: {name}')
    for name in ['run_scaling.py', 'run_rank.py', 'plot_results.py', 'validate_results.py', 'original_input_hashes.json']:
        if not (arguments.baseline / name).is_file():
            raise RuntimeError(f'Missing baseline support file: {name}')
    if not arguments.no_plots:
        subprocess.run([str(arguments.baseline / 'plot_env/bin/python'), '-c', 'import matplotlib'], check=True)
    if not arguments.no_validate and not (WORKSPACE / 'build_release/opm-common/bin/compareECL').is_file():
        raise RuntimeError('Build compareECL with Ninja in build_release before enabling validation.')
    return metadata, cases


def initialize(arguments):
    root = arguments.output
    metadata, cases = preflight(arguments)
    identity = dict(baseline=str(arguments.baseline), cases=[case['name'] for case in cases],
                    binary_sha256=metadata['binary_sha256'], simulated_days=730000,
                    physics='no-diffusion-dispersion', maximum_timestep_days=365000)
    configuration = root / 'configuration.json'
    if configuration.exists() and read(configuration) != identity:
        raise RuntimeError('Case selection or baseline changed. Use a separate --output directory.')
    state(root, 'preparing', cases=identity['cases'])
    for name, expected in metadata['binary_sha256'].items():
        frozen_copy(arguments.baseline / 'bin' / name, root / 'bin' / name, expected)
    if not arguments.no_validate:
        frozen_copy(WORKSPACE / 'build_release/opm-common/bin/compareECL', root / 'bin/compareECL')
    support = root / 'support'
    support.mkdir(exist_ok=True)
    for name in ['run_scaling.py', 'plot_results.py', 'validate_results.py']:
        frozen_copy(arguments.baseline / name, support / name)
    frozen_copy(arguments.baseline / 'run_rank.py', root / 'run_rank.py')
    manifests = read(arguments.baseline / 'original_input_hashes.json')
    selected = {case['name'] for case in cases}
    for manifest in manifests:
        if manifest['case'] not in selected:
            continue
        for entry in manifest['files']:
            if stopped(root):
                raise KeyboardInterrupt
            source = Path(entry['path'])
            frozen_copy(source, root / 'source_inputs' / manifest['case'] / source.name, entry['sha256'])
    for case in cases:
        case['original_input'] = case['input']
        case['input'] = str(root / 'source_inputs' / case['name'] / Path(case['input']).name)
    write(root / 'cases.json', cases)
    write(configuration, identity)
    if not (root / 'environment.json').exists():
        metadata.update(workload='Full 2,000-year schedule; weekly continuation',
                        completed_cases=[], completed_timing_runs=0,
                        source_artifacts=str(arguments.baseline), input_files_frozen=True,
                        support_sha256={path.name: digest(path) for path in support.iterdir() if path.is_file()})
        for key in ['coverage', 'primary_scaling_artifacts', 'affinity_audit', 'validation']:
            metadata.pop(key, None)
        write(root / 'environment.json', metadata)
    if not arguments.no_reuse_completed:
        for case in cases:
            for layout in LAYOUTS:
                previous = arguments.baseline / 'runs' / case['name'] / layout
                destination = root / 'runs' / case['name'] / layout
                if not destination.exists() and (previous / 'result.json').exists():
                    if read(previous / 'result.json')['status'] == 'completed':
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        destination.symlink_to(previous, target_is_directory=True)
    module = load_module(support / 'run_scaling.py', 'scaling_support')
    module.ROOT, module.BUILD, module.write_json = root, WORKSPACE / 'build_release', write
    invocation = root / 'invocations' / datetime.datetime.now().strftime('%Y%m%dT%H%M%S%f')
    invocation.mkdir(parents=True)
    shutil.copy2(__file__, invocation / 'runner.py')
    write(invocation / 'options.json', {key: str(value) if isinstance(value, Path) else value
                                      for key, value in vars(arguments).items()})
    return module, cases


def terminate(process):
    # A process may exit between poll() and killpg(). Its MPI peers still need cleanup.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_case(arguments, module, case, layout):
    root = arguments.output
    output = root / 'runs' / case['name'] / layout
    record = output / 'result.json'
    if record.exists():
        previous = read(record)
        if previous['status'] == 'completed':
            print(f"SKIP completed {case['name']} {layout}", flush=True)
            return previous
        if previous['status'] != 'interrupted' and not arguments.retry_failed:
            print(f"SKIP {previous['status']} {case['name']} {layout}; use --retry-failed to retry", flush=True)
            return previous
    if stopped(root):
        raise KeyboardInterrupt
    if shutil.disk_usage(root).free < arguments.min_free_gib * 1024**3:
        raise RuntimeError(f'Less than {arguments.min_free_gib:g} GiB free; resume after freeing space.')
    if output.exists():
        archive = root / 'attempts' / case['name'] / layout / datetime.datetime.now().strftime('%Y%m%dT%H%M%S%f')
        archive.parent.mkdir(parents=True, exist_ok=True)
        output.rename(archive)
    deck = module.prepare_deck(case, 'no-diffusion-dispersion')
    changes = read(deck.parent / 'input_changes.json')
    if digest(deck) != changes['benchmark_sha256'] or changes['schedule_changed']:
        raise RuntimeError(f'The prepared full-schedule deck changed: {deck}')
    output.mkdir(parents=True)
    command = module.make_command(deck, layout, output)
    environment = dict(os.environ)
    environment.pop('OMP_NUM_THREADS', None)
    environment.update(UCX_TLS='sysv,self', OPENBLAS_NUM_THREADS='1', OMP_DYNAMIC='FALSE',
                       HSA_OVERRIDE_CPU_AFFINITY_DEBUG='0')
    (output / 'command.txt').write_text(shlex.join(command) + '\n')
    result = dict(case=case['name'], layout=layout, physics='no-diffusion-dispersion',
                  cartesian_cells=case['cartesian_cells'], command=command, started_utc=now(),
                  environment={key: environment[key] for key in ['UCX_TLS', 'OPENBLAS_NUM_THREADS',
                               'OMP_DYNAMIC', 'HSA_OVERRIDE_CPU_AFFINITY_DEBUG']})
    print(f"START {case['name']} {layout}", flush=True)
    start, last_heartbeat, peak_vram = time.monotonic(), 0, 0
    failure, process = None, None
    log_path = output / 'console.log'
    try:
        with log_path.open('w') as log:
            process = subprocess.Popen(command, cwd=module.BUILD, env=environment,
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, pass_fds=(LOCK.fileno(),))
            write(output / 'running.json', dict(pid=process.pid, started_utc=result['started_utc']))
            with log_path.open() as reader:
                tail = ''
                while process.poll() is None:
                    chunk = reader.read(65536)
                    new_output = tail[-1024:] + chunk
                    tail = (tail + chunk)[-16384:]
                    elapsed = time.monotonic() - start
                    if stopped(root):
                        failure = 'interrupted'
                    elif OUT_OF_MEMORY.search(new_output):
                        failure = 'gpu_out_of_memory'
                    elif arguments.timeout_hours and elapsed > arguments.timeout_hours * 3600:
                        failure = 'timeout'
                    if failure:
                        terminate(process)
                        break
                    try:
                        peak_vram = max(peak_vram, int(Path('/sys/class/drm/card1/device/mem_info_vram_used').read_text()))
                    except OSError:
                        pass
                    if elapsed - last_heartbeat >= 30 or last_heartbeat == 0:
                        state(root, 'running', case=case['name'], layout=layout, elapsed_seconds=elapsed,
                              console_log=str(log_path), last_output=tail.splitlines()[-8:])
                        last_heartbeat = elapsed
                    time.sleep(0.1)
            returncode = process.wait()
    finally:
        if process is not None and process.poll() is None:
            terminate(process)
    result.update(returncode=returncode, wall_seconds=time.monotonic() - start,
                  sampled_peak_device_vram_bytes=peak_vram)
    # Logs can grow for days. Parse them without loading the whole file into RAM.
    with log_path.open(errors='replace') as log:
        for line in log:
            if OUT_OF_MEMORY.search(line):
                failure = failure or 'gpu_out_of_memory'
            for key, expression in module.PATTERNS.items():
                match = re.search(expression, line)
                if match:
                    result[key] = float(match[1]) if key.endswith('_seconds') else int(match[1])
    result['status'] = failure or ('completed' if returncode == 0 and 'simulation_seconds' in result else 'failed')
    write(record, result)
    print(f"END {case['name']} {layout}: {result['status']}, wall={result['wall_seconds']:.1f}s", flush=True)
    return result


def postprocess(arguments, command, logfile):
    with (arguments.output / logfile).open('a') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True,
                                   pass_fds=(LOCK.fileno(),))
        try:
            while process.poll() is None:
                if stopped(arguments.output):
                    terminate(process)
                    raise KeyboardInterrupt
                time.sleep(0.2)
        finally:
            if process.poll() is None:
                terminate(process)
    if process.returncode:
        print(f'Postprocessing exited {process.returncode}; see {logfile}', flush=True)
    return process.returncode


def refresh_plots(arguments):
    if not arguments.no_plots:
        state(arguments.output, 'plotting')
        result = postprocess(arguments, [str(arguments.baseline / 'plot_env/bin/python'), str(Path(__file__).resolve()),
                                         'plot', '--output', str(arguments.output)], 'plot.log')
        if result:
            POSTPROCESSING_FAILURES.append('plot generation; see plot.log')


def plot(root):
    module = load_module(root / 'support/plot_results.py', 'plot_support')
    module.ROOT = root
    rows = read(root / 'results.json')
    cases = {row['case'] for row in rows}
    complete = {case for case in cases if {row['layout'] for row in rows
                if row['case'] == case and row['status'] == 'completed'} == set(LAYOUTS)}
    rows = [row for row in rows if row['case'] in complete and row['status'] == 'completed']
    if not rows:
        print('No case has all five completed configurations yet.')
        return

    def annotate(figure, field):
        if field == 'phases':
            figure.text(0.08, 0.99, 'SPE11C: runtime by phase', ha='left', va='top', fontsize=18)
        figure.text(0.08, 0.94 if field == 'phases' else 0.97,
                    'Ryzen 9 5950X: 16 physical cores | Radeon RX 9070 XT: 16 GB',
                    ha='left', va='top', fontsize=10)
        figure.text(0.08, 0.02,
                    f'Full 2,000-year schedule; {len(complete)} complete grids; one run per configuration.'
                    '\nDiffusion/dispersion disabled. Default tolerances; numerical differences may occur.'
                    '\nTotal includes setup/output. Phases use Flow rank-zero timers; GPU work may finish in a later phase.',
                    ha='left', va='bottom', fontsize=8.5, color='#4B5563')

    module.annotate = annotate
    for field in module.METRICS:
        module.make_plot(rows, field)
    module.make_plot(rows, 'wall_seconds', logarithmic=True)
    module.make_phase_overview(rows)
    print(f'Updated plots for {len(complete)} complete grids.', flush=True)


def report(arguments, excluded):
    root = arguments.output
    rows = read(root / 'results.json')
    complete = [row for row in rows if row['status'] == 'completed']
    metadata = read(root / 'environment.json')
    metadata.update(completed_timing_runs=len(complete),
                    completed_cases=sorted({row['case'] for row in complete
                        if sum(other['case'] == row['case'] for other in complete) == 5},
                        key=lambda name: int(name.split('_')[1])),
                    excluded_cases=excluded, sampled_vram_scope='Whole device, sampled every 0.1 seconds for new runs')
    if (root / 'validation_results.json').exists():
        checks = read(root / 'validation_results.json')
        metadata['validation'] = dict(passed=sum(row['returncode'] == 0 for row in checks), total=len(checks))
    write(root / 'environment.json', metadata)
    lines = ['Full-schedule SPE11C continuation', '',
             'Preserves all 730,000 simulated days and the original grid sizes.',
             'Diffusion/dispersion disabled; all five modes use the original full-study solver options.',
             'Default nonlinear tolerances and linear reduction 0.01; max timestep 365,000 days.',
             'Numerical agreement is not guaranteed: the earlier full default-tolerance runs had field mismatches.',
             'Frozen binaries/input copies are verified against the original experiment hashes.',
             'Completed baseline runs are reused through symlinks; their logs/commands are preserved.',
             'Interrupted runs restart from the beginning of that configuration. Earlier attempts are retained.',
             'Failed runs are skipped on resume unless --retry-failed is supplied.',
             'RSC_176 is excluded based on its previous GPU allocation failure.', '',
             f'Completed runs: {len(complete)}. Complete five-mode cases: {len(metadata["completed_cases"])}.',
             'See status.json and manager.log for progress; each runs/CASE/MODE/console.log has simulator output.',
             'results.csv/json contain total, simulation, assembly, linear solve, properties/update and pre/post times.',
             'The PNG/PDF/SVG plots include only grids with all five runs completed.',
             'Phase timers are rank-zero wall timers; asynchronous GPU work may be charged to a later phase.',
             'validation_results.json and validation/ retain all numerical comparison failures.', '',
             'Case / mode / status / elapsed seconds']
    lines.extend(f"{row['case']} / {row['layout']} / {row['status']} / {row['wall_seconds']:.2f}" for row in rows)
    lines.extend(['', 'Excluded cases:', json.dumps(excluded, indent=2), ''])
    (root / 'README.txt').write_text('\n'.join(lines))


def run(arguments):
    global LOCK
    root = arguments.output
    inherited_lock = 'SPE11C_WEEK_LOCK_FD' in os.environ
    LOCK = acquire_lock(root)
    (root / '.spe11c-week').write_text('Full-schedule SPE11C runner, format 1\n')
    if not inherited_lock:
        (root / 'STOP').unlink(missing_ok=True)
    for signum in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(signum, request_stop)
    try:
        module, cases = initialize(arguments)
        module.summarize()
        excluded = [dict(case='RSC_176', reason='GPU out of memory in the previous 26-day experiment', attempted=False)]
        first_oversized = None
        for case in cases:
            if stopped(root):
                raise KeyboardInterrupt
            if first_oversized:
                excluded.append(dict(case=case['name'], reason='larger than full-schedule GPU OOM case',
                                     first_oversized_case=first_oversized, attempted=False))
                continue
            for layout in LAYOUTS:
                result = run_case(arguments, module, case, layout)
                module.summarize()
                if result['status'] == 'interrupted':
                    raise KeyboardInterrupt
                if layout == 'gpu' and result['status'] != 'completed':
                    if result['status'] == 'gpu_out_of_memory':
                        first_oversized = case['name']
                        excluded.append(dict(case=case['name'], reason='full-schedule GPU out of memory', attempted=True))
                    break
            write(root / 'excluded_cases.json', excluded)
            report(arguments, excluded)
            refresh_plots(arguments)
        write(root / 'excluded_cases.json', excluded)
        module.summarize()
        if not arguments.no_validate:
            state(root, 'validating')
            comparison_status = postprocess(arguments, [sys.executable, str(Path(__file__).resolve()), 'validate',
                                                       '--output', str(root)], 'validation.log')
            rows = read(root / 'results.json')
            gpu_cases = {row['case'] for row in rows if row['layout'] == 'gpu' and row['status'] == 'completed'}
            expected_checks = 4 * sum(row['case'] in gpu_cases and row['layout'] != 'gpu'
                                     and row['status'] == 'completed' for row in rows)
            checks = read(root / 'validation_results.json') if (root / 'validation_results.json').exists() else []
            if len(checks) != expected_checks or comparison_status not in [0, 1]:
                POSTPROCESSING_FAILURES.append('incomplete field comparisons; see validation.log')
        report(arguments, excluded)
        failures = [row for row in read(root / 'results.json') if row['status'] not in ['completed', 'gpu_out_of_memory']]
        validation = read(root / 'environment.json').get('validation', {})
        state(root, 'finished_with_failures' if failures or POSTPROCESSING_FAILURES else 'finished',
              failed_runs=len(failures), postprocessing_errors=POSTPROCESSING_FAILURES,
              numerical_checks=validation)
        print(f'Done. Results: {root}', flush=True)
        return 1 if failures or POSTPROCESSING_FAILURES else 0
    except KeyboardInterrupt:
        state(root, 'stopped', note='Resume with start; the interrupted configuration will restart.')
        print('Stopped. Completed runs and interrupted logs are retained.', flush=True)
        return 130
    except Exception as error:
        state(root, 'error', message=str(error))
        raise
    finally:
        LOCK.close()


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', nargs='?', default='start',
                        choices=['start', 'run', 'status', 'stop', 'check', 'plot', 'validate'])
    parser.add_argument('--baseline', type=Path, default=DEFAULT_BASELINE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--cases', help='Comma-separated sizes, default 16,32,...,160; keep the same list on resume')
    parser.add_argument('--timeout-hours', type=float, default=0, help='Per-run wall limit; 0 means unlimited (default)')
    parser.add_argument('--min-free-gib', type=float, default=10, help='Pause before a run if disk space is below this value')
    parser.add_argument('--retry-failed', action='store_true', help='Archive and retry failed/timed-out runs once')
    parser.add_argument('--no-reuse-completed', action='store_true', help='Do not import the 15 completed baseline runs')
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--no-validate', action='store_true', help='Skip end-of-queue field comparisons')
    arguments = parser.parse_args()
    arguments.output, arguments.baseline = arguments.output.resolve(), arguments.baseline.resolve()
    if arguments.timeout_hours < 0 or arguments.min_free_gib < 0:
        parser.error('Timeout and disk-space thresholds must be nonnegative.')
    root = arguments.output
    if arguments.command == 'status':
        print(f'Active: {is_running(root)}; output: {root}')
        if (root / 'status.json').exists():
            print(json.dumps(read(root / 'status.json'), indent=2))
        if (root / 'results.json').exists():
            rows = read(root / 'results.json')
            print(f"Completed: {sum(row['status'] == 'completed' for row in rows)} / {len(read(root / 'cases.json')) * 5} runs")
        return 0
    if arguments.command == 'stop':
        if not is_running(root):
            print('No active runner.')
            return 0
        (root / 'STOP').touch()
        print('Stop requested. The current configuration will be interrupted; completed runs are retained.')
        return 0
    if arguments.command == 'plot':
        plot(root)
        return 0
    if arguments.command == 'validate':
        module = load_module(root / 'support/validate_results.py', 'validation_support')
        module.ROOT, module.COMPARATOR = root, root / 'bin/compareECL'
        module.main()
        return 0
    if arguments.command == 'check':
        metadata, cases = preflight(arguments)
        completed = sum((arguments.baseline / 'runs' / case['name'] / layout / 'result.json').exists()
                        and read(arguments.baseline / 'runs' / case['name'] / layout / 'result.json')['status'] == 'completed'
                        for case in cases for layout in LAYOUTS)
        print(f"Full schedule: {metadata['simulated_days']:g} days; {len(cases)} grids, {len(cases) * 5} configurations.")
        print(f'Reusable completed runs: {0 if arguments.no_reuse_completed else completed}. RSC_176 excluded.')
        print(f'Per-run timeout: {arguments.timeout_hours:g} hours (0 = unlimited). Output: {root}')
        return 0
    if root == arguments.baseline or root == WORKSPACE or root == WORKSPACE / 'build_release':
        parser.error('Use a separate experiment output directory.')
    if (root.exists() and not (root / '.spe11c-week').exists()
            and not (root / 'configuration.json').exists() and any(root.iterdir())):
        parser.error('The output directory contains another experiment. Choose an empty directory.')
    root.mkdir(parents=True, exist_ok=True)
    if arguments.command == 'run':
        return run(arguments)
    lock = acquire_lock(root)
    try:
        (root / '.spe11c-week').write_text('Full-schedule SPE11C runner, format 1\n')
        (root / 'STOP').unlink(missing_ok=True)
        environment = dict(os.environ, SPE11C_WEEK_LOCK_FD=str(lock.fileno()))
        forwarded = [str(Path(__file__).resolve()), 'run', '--output', str(root), '--baseline', str(arguments.baseline),
                     '--timeout-hours', str(arguments.timeout_hours), '--min-free-gib', str(arguments.min_free_gib)]
        if arguments.cases:
            forwarded += ['--cases', arguments.cases]
        for key in ['retry_failed', 'no_reuse_completed', 'no_plots', 'no_validate']:
            if getattr(arguments, key):
                forwarded.append('--' + key.replace('_', '-'))
        with (root / 'manager.log').open('a') as log:
            process = subprocess.Popen([sys.executable, '-u', *forwarded], env=environment,
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, pass_fds=(lock.fileno(),))
        print(f'Started PID {process.pid}; survives terminal disconnect. Log: {root / "manager.log"}')
        print('Use status for progress, or stop to interrupt the current configuration.')
    finally:
        lock.close()
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f'ERROR: {error}', file=sys.stderr, flush=True)
        raise SystemExit(1)
