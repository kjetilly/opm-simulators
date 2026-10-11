#!/bin/sh
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with OPM.  If not, see <http://www.gnu.org/licenses/>.

# Launch this script on every MPI rank, forwarding the same deck and solver
# options to both executables. Rank zero runs the GPU implementation.
set -eu
rank=${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-${PMIX_RANK:-}}}
if [ -z "$rank" ]; then
    echo "Run this script with mpirun or an MPI launcher providing a rank variable." >&2
    exit 1
fi
binary_directory=${FLOW_BINARY_DIR:-./opm-simulators/bin}
# OPM gives this environment variable precedence over --threads-per-process.
unset OMP_NUM_THREADS
if [ "$rank" -eq 0 ]; then
    exec "$binary_directory/flow_gpu" "$@" \
        --experimental-compute-properties-on-gpu=true \
        --linear-solver-accelerator=gpu --threads-per-process=1
else
    exec "$binary_directory/flow_gpu_cpu" "$@" \
        --linear-solver-accelerator=cpu --threads-per-process="${CPU_THREADS:-1}"
fi
