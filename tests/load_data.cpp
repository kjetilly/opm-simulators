#undef NDEBUG
#include "config.h"
#undef NDEBUG
#include <iostream>
#if USE_HIP
#include <opm/simulators/linalg/gpuistl_hip/GpuBuffer.hpp>
#include <opm/simulators/linalg/gpuistl_hip/DualBuffer.hpp>
#else
#include <opm/simulators/linalg/gpuistl/GpuBuffer.hpp>
#include <opm/simulators/linalg/gpuistl/DualBuffer.hpp
#endif
#include "opm/simulators/flow/FlowProblemBlackoil.hpp"
#include <tests/common_type_tag.hpp>


#include <opm/simulators/flow/FlowProblemBlackoilGpu.hpp>
#include <fmt/core.h>

#include <tests/load_data.hpp>

//
template <class TypeTag>
void loadData(int argc, char **argv, const CallbackType<TypeTag>& callback) {
    std::cout << "Loading data" << std::endl;
    auto mainObject = Opm::Main(argc, argv);
    //mainObject.runStatic<TypeTag>();
    auto mainFlow = mainObject.gimmeFlowMain<TypeTag>();
    std::cout << "Got mainFlow" << std::endl;
    mainFlow->execute();
    std::cout << "Executed mainFlow" << std::endl;
    auto simulator = mainFlow->getSimulator();
    std::cout << "Got simulator" << std::endl;
    auto& problem = simulator->problem();
    std::cout << "Got problem" << std::endl;
    auto problemGpuBuf = Opm::gpuistl::
       copy_to_gpu<double, Opm::gpuistl::GpuBuffer, Opm::gpuistl::DualBuffer, TypeTag, TypeTag>(problem);
    std::cout << "Calling callback" << std::endl;
    callback(problemGpuBuf);
    std::cout << "Callback executed" << std::endl;
}
using namespace Opm::Properties::TTag;
template void loadData<FlowSimpleProblem>(int argc, char **argv, const CallbackType<FlowSimpleProblem>&);