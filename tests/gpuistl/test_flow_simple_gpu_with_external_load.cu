/*
  Copyright 2024, SINTEF AS

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

#include "config.h"
#include <tests/common_type_tag.hpp>
#include <tests/load_data.hpp>
#include <fmt/core.h>
#if USE_HIP
#include <opm/simulators/linalg/gpuistl_hip/GpuBuffer.hpp>
#include <opm/simulators/linalg/gpuistl_hip/DualBuffer.hpp>
#else
#include <opm/simulators/linalg/gpuistl/GpuBuffer.hpp>
#include <opm/simulators/linalg/gpuistl/DualBuffer.hpp
#endif

#include <opm/simulators/flow/FlowProblemBlackoilGpu.hpp>


int main(int argc, char** argv)
{
    using TypeTag = Opm::Properties::TTag::FlowSimpleProblem;

    loadData<TypeTag>(argc, argv, [](Opm::GetPropType<TypeTag, Opm::Properties::Problem>& problem) {
      fmt::println("From callback");
      auto problemGpuBuf = Opm::gpuistl::copy_to_gpu<double, Opm::gpuistl::GpuBuffer, Opm::gpuistl::DualBuffer, TypeTag, TypeTag>(problem);
      fmt::println("Copied to GPU");

    });
    return 0;
//    return Opm::start<TypeTag>(argc, argv);
}
