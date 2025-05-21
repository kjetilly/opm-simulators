#include "config.h"
#include "opm/simulators/flow/FlowProblemBlackoil.hpp"
#include <tests/common_type_tag.hpp>
#if USE_HIP
#include <opm/simulators/linalg/gpuistl_hip/GpuBuffer.hpp>
#include <opm/simulators/linalg/gpuistl_hip/DualBuffer.hpp>
#else
#include <opm/simulators/linalg/gpuistl/GpuBuffer.hpp>
#include <opm/simulators/linalg/gpuistl/DualBuffer.hpp
#endif

#include <opm/simulators/flow/FlowProblemBlackoilGpu.hpp>
#include <fmt/core.h>

#include <tests/load_data.hpp>

//
template <class TypeTag>
void loadData(int argc, char **argv, std::function<void(typename Opm::GetPropType<TypeTag, Opm::Properties::Problem>&)> callback) {
    auto mainObject = Opm::Main(argc, argv);
    //mainObject.runStatic<TypeTag>();
    auto mainFlow = mainObject.gimmeFlowMain<TypeTag>();
    mainFlow->execute();

    auto simulator = mainFlow->getSimulator();

    auto& problem = simulator->problem();

    fmt::println("Calling callback");
    callback(problem);
    fmt::println("Callback executed");
}
using namespace Opm::Properties::TTag;
template void loadData<FlowSimpleProblem>(int argc, char **argv, std::function<void(Opm::GetPropType<FlowSimpleProblem, Opm::Properties::Problem>&)>);