/*
  Copyright TODO ADD YEAR AND NAME OF AUTHOR

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
#include <config.h>

#include <opm/simulators/flow/FlowGasWaterEnergyTypeTag.hpp>
#include <opm/simulators/flow/Main.hpp>
#include <opm/simulators/flow/SimulatorFullyImplicit.hpp>

#include <memory>
#include <tuple>

namespace Opm::Properties
{
namespace TTag
{
    // Keep exactly the GPU model's physics, including disabled diffusion and
    // dispersion, while compiling CPU assembly and properties with host OpenMP.
    struct FlowGasWaterEnergyProblemCPU {
        using InheritsFrom = std::tuple<FlowGasWaterEnergyProblemGPU>;
    };
} // namespace TTag

template <class TypeTag>
struct RunAssemblyOnGpu<TypeTag, TTag::FlowGasWaterEnergyProblemCPU> {
    static constexpr bool value = false;
};

template <class TypeTag>
struct Scalar<TypeTag, TTag::FlowGasWaterEnergyProblemCPU> {
    using type = double;
};
} // namespace Opm::Properties

int
main(int argc, char** argv)
{
    using TypeTag = Opm::Properties::TTag::FlowGasWaterEnergyProblemCPU;
    auto mainObject = std::make_unique<Opm::Main>(argc, argv);
    const auto result = mainObject->runStatic<TypeTag>();
    // Finalize MPI after destroying the simulator.
    mainObject.reset();
    return result;
}
