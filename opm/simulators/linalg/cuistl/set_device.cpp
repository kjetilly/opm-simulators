/*
  Copyright SINTEF AS 2022

  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.
*/
#include <opm/simulators/linalg/cuistl/set_device.hpp>

namespace Opm::cuistl {
void setDevice(const size_t mpiRank, const size_t numberOfMpiRanks) {

    int deviceCount = -1;
    OPM_CUDA_SAFE_CALL(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        // If they have CUDA enabled, this will fail later down the line.
        // At this point in the simulator, we can not determine if CUDA is enabled, so we can only
        // issue a warning.
        OpmLog::warning("Could not find any CUDA devices.");
    }

         // Now do a round robin kind of assignment
    const auto deviceId = mpiRank % deviceCount;
    OPM_CUDA_SAFE_CALL(cudaDeviceReset());
    OPM_CUDA_SAFE_CALL(cudaThreadExit());
    OPM_CUDA_SAFE_CALL(cudaSetDevice(deviceId));
}
}
