/*
  Copyright 2025 SINTEF AS
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
// #undef NDEBUG
#include <config.h>
// #undef NDEBUG

#include <string>

#define BOOST_TEST_MODULE TestFlowProblemGpu

#include <boost/test/unit_test.hpp>
#include <opm/material/densead/Evaluation.hpp>
#include <opm/material/fluidmatrixinteractions/EclMaterialLawManagerSimple.hpp>
#include <opm/simulators/linalg/gpuistl/DualBuffer.hpp>
#include <opm/simulators/linalg/gpuistl/GpuBuffer.hpp>

#include <opm/models/blackoil/blackoilmodel.hh>
#include <opm/models/discretization/common/tpfalinearizer.hh>
#include <opm/models/utils/simulator.hh>

#include <opm/simulators/flow/FlowProblemBlackoil.hpp>
#include <opm/simulators/flow/FlowProblemBlackoilGpu.hpp>
#include <opm/simulators/flow/FlowProblemBlackoilProperties.hpp>
#include <opm/simulators/utils/moduleVersion.hpp>

#include <opm/simulators/flow/BlackoilModelParameters.hpp>
#include <opm/simulators/flow/FlowGenericVanguard.hpp>
#include <opm/simulators/flow/FlowProblemBlackoil.hpp>
#include <opm/simulators/flow/FlowProblemBlackoilProperties.hpp>
#include <opm/simulators/flow/equil/EquilibrationHelpers.hpp>
#include <opm/simulators/linalg/gpuistl/DualBuffer.hpp>
#include <opm/simulators/linalg/gpuistl/GpuBuffer.hpp>
#include <opm/simulators/linalg/gpuistl/GpuView.hpp>
#include <opm/simulators/linalg/gpuistl/detail/gpu_safe_call.hpp>
#include <opm/simulators/linalg/gpuistl/gpu_smart_pointer.hpp>
#include <opm/simulators/linalg/parallelbicgstabbackend.hh>
#include <opm/simulators/wells/BlackoilWellModel.hpp>

#include <utility>

#include <cuda_runtime.h>

/*
Functionality requested for the blackoil flow problem on gpu:
[X] indicates that the functionality is added and verified with unit test
[-] indicates that the functionality does not seem to be used in spe11
[ ] indicates that the functionality is not added yet

[X] - problem.model().linearizer().getLinearizationType()
[X] - problem.satnumRegionIndex(globalSpaceIdx)
[X] - problem.materialLawParams(globalSpaceIdx)
[X] - problem.rockCompressibility(globalSpaceIdx)
[X] - problem.rockReferencePressure(globalSpaceIdx)
[X] - problem.porosity(globalSpaceIdx, timeIdx)
[-] - problem.maxOilVaporizationFactor(timeIdx, globalSpaceIdx)
[-] - problem.maxGasDissolutionFactor(timeIdx, globalSpaceIdx)
[-] - problem.maxOilSaturation(globalSpaceIdx)
[-] - problem.template rockCompPoroMultiplier<Evaluation>(*this, globalSpaceIdx)
[X] - problem.updateRelperms(mobility_, dirMob_, fluidState_, globalSpaceIdx)
[X] - problem.template rockCompTransMultiplier<Evaluation>(*this, globalSpaceIdx)

*/

#include <opm/material/fluidstates/BlackOilFluidState.hpp>
#include <opm/material/fluidsystems/BlackOilFluidSystem.hpp>
#include <opm/material/fluidsystems/BlackOilFluidSystemNonStatic.hpp>

#include <opm/input/eclipse/Deck/Deck.hpp>
#include <opm/input/eclipse/EclipseState/EclipseState.hpp>
#include <opm/input/eclipse/Parser/Parser.hpp>
#include <opm/input/eclipse/Python/Python.hpp>
#include <opm/input/eclipse/Schedule/Schedule.hpp>
#include <tests/common_type_tag.hpp>
// #include <tests/load_data.hpp>
// #include <tests/load_data.cpp>

#include <iostream>
#include <memory>
#include <type_traits>
// #include <dune/common/mpihelper.hh>
#include <dune/common/parallel/mpihelper.hh>
#include <opm/models/utils/start.hh>

#include <opm/simulators/linalg/gpuistl/gpu_smart_pointer.hpp>

template <class ProblemView>
__global__ void
satnumFromFlowProblemBlackoilGpu(ProblemView prob, unsigned short* res)
{
    *res = prob.satnumRegionIndex(0);
}

template <class ProblemView>
__global__ void
linTypeFromFlowProblemBlackoilGpu(ProblemView prob, Opm::LinearizationType* res)
{
    *res = prob.model().linearizer().getLinearizationType();
}

template <class ProblemView>
__global__ void
rockCompressibilityFromFlowProblemBlackoilGpu(ProblemView prob, double* res)
{
    *res = prob.rockCompressibility(0);
}

template <class ProblemView>
__global__ void
porosityFromFlowProblemBlackoilGpu(ProblemView prob, double* res)
{
    *res = prob.porosity(0, 0);
}

template <class ProblemView>
__global__ void
rockReferencePressureFromFlowProblemBlackoilGpu(ProblemView prob, double* res)
{
    *res = prob.rockReferencePressure(0);
}

template <class ProblemView>
__global__ void
materialLawParamsCallable(ProblemView prob)
{
    auto matLawParams = prob.materialLawParams(0);
}

template <class DirMobPtr, class ProblemView, class MobArr, class FluidState>
__global__ void
updateRelPermsFromFlowProblemBlackoilGpu(ProblemView prob, MobArr mob, FluidState fs)
{
    auto dirPtr = DirMobPtr(); // produces nullptr, this value is not used in the function, but should match signature
    prob.updateRelperms(mob, dirPtr, fs, 0);
}


BOOST_AUTO_TEST_CASE(TestInstantiateGpuFlowProblem)
{
    std::cout << __LINE__ << std::endl;
    using TypeTag = Opm::Properties::TTag::FlowSimpleProblem;
    std::cout << __LINE__ << std::endl;
    // TODO: will this actually refer to the very_simple_deck.DATA inside the gpuistl folder,
    // TODO: do we need to keep track of the path since it can be hipified?
    const std::string filename = "very_simple_deck.DATA";
        std::cout << __LINE__ << std::endl;

    std::vector<std::string> args = {filename, "very_simple_deck.DATA", ""};
        std::cout << __LINE__ << std::endl;

    std::vector<char*> argv2;
        std::cout << __LINE__ << std::endl;

    for (auto& arg : args) {
            std::cout << __LINE__ << std::endl;

        argv2.push_back(static_cast<char*>(arg.data()));
            std::cout << __LINE__ << std::endl;

    }
        std::cout << __LINE__ << std::endl;

    using TypeTag = Opm::Properties::TTag::FlowSimpleProblem;
        std::cout << __LINE__ << std::endl;

    // auto mainObject = Opm::Main(argv2.size() - 1, argv2.data());
    auto mainObject = Opm::Main(filename, false, false);
        std::cout << __LINE__ << std::endl;

    // mainObject.runStatic<TypeTag>();
    auto mainFlow = mainObject.gimmeFlowMain<TypeTag>();
        std::cout << __LINE__ << std::endl;

    std::cout << "Got mainFlow" << std::endl;
    mainFlow->execute();
    std::cout << "Executed mainFlow" << std::endl;
    auto simulator = mainFlow->getSimulator();
    std::cout << "Got simulator" << std::endl;
    auto& problem = simulator->problem();
    std::cout << "Got problem" << std::endl;
    #if 0
    auto problemGpuBuf
        = Opm::gpuistl::copy_to_gpu<double, Opm::gpuistl::GpuBuffer, Opm::gpuistl::DualBuffer, TypeTag, TypeTag>(
            problem);

    fmt::println("From callback");
    // auto problemGpuBuf = Opm::gpuistl::
    //     copy_to_gpu<double, Opm::gpuistl::GpuBuffer, Opm::gpuistl::DualBuffer, TypeTag, TypeTag>(problem);
    fmt::println("Copied to GPU");
    auto problemGpuView = Opm::gpuistl::make_view<Opm::gpuistl::GpuView, Opm::gpuistl::ValueAsPointer>(problemGpuBuf);

    fmt::println("At line {}", __LINE__);

    auto satNumOnGpu = Opm::gpuistl::make_gpu_unique_ptr<unsigned short>(0);
    satnumFromFlowProblemBlackoilGpu<<<1, 1>>>(problemGpuView, satNumOnGpu.get());
    const auto satNumOnCpu = Opm::gpuistl::copyFromGPU(satNumOnGpu);
    BOOST_CHECK_EQUAL(satNumOnCpu, problem.satnumRegionIndex(0));
    fmt::println("At line {}", __LINE__);

    auto linTypeOnGpu = Opm::gpuistl::make_gpu_unique_ptr<Opm::LinearizationType>(Opm::LinearizationType{});
    linTypeFromFlowProblemBlackoilGpu<<<1, 1>>>(problemGpuView, linTypeOnGpu.get());
    const auto linTypeOnCpu = Opm::gpuistl::copyFromGPU(linTypeOnGpu);
    auto linTypeFromCPUSimulator = problem.model().linearizer().getLinearizationType();
    BOOST_CHECK_EQUAL(linTypeOnCpu.type, linTypeFromCPUSimulator.type);
    fmt::println("At line {}", __LINE__);

    auto rockCompressibilityOnGpu = Opm::gpuistl::make_gpu_unique_ptr<double>(0.0);
    rockCompressibilityFromFlowProblemBlackoilGpu<<<1, 1>>>(problemGpuView, rockCompressibilityOnGpu.get());
   
    const auto rocmCompressibilityOnCpu = Opm::gpuistl::copyFromGPU(rockCompressibilityOnGpu);
    BOOST_CHECK_EQUAL(rocmCompressibilityOnCpu, problem.rockCompressibility(0));
    fmt::println("At line {}", __LINE__);

    auto  porosityOnGpu = Opm::gpuistl::make_gpu_unique_ptr<double>(0.0);
    porosityFromFlowProblemBlackoilGpu<<<1, 1>>>(problemGpuView, porosityOnGpu.get());
    const auto porosityOnCpu = Opm::gpuistl::copyFromGPU(porosityOnGpu);
    BOOST_CHECK_EQUAL(porosityOnCpu, problem.porosity(0, 0));

    auto referencePressureOnGpu = Opm::gpuistl::make_gpu_unique_ptr<double>(0.0);
    rockReferencePressureFromFlowProblemBlackoilGpu<<<1, 1>>>(problemGpuView, referencePressureOnGpu.get());
    fmt::println("At line {}", __LINE__);
    const auto referencePressureOnCpu = Opm::gpuistl::copyFromGPU(referencePressureOnGpu);
    BOOST_CHECK_EQUAL(referencePressureOnCpu, problem.rockReferencePressure(0));
    
    materialLawParamsCallable<<<1, 1>>>(problemGpuView);
    fmt::println("At line {}", __LINE__);

    using FluidSystem = Opm::BlackOilFluidSystem<double>;
    using Evaluation = Opm::DenseAd::Evaluation<double, 2>;
    using Scalar = double;
    // using DirectionalMobilityPtr = Utility::CopyablePtr<DirectionalMobility<TypeTag, Evaluation>>;
    using DirectionalMobilityPtr = Opm::Utility::CopyablePtr<Opm::DirectionalMobility<TypeTag>>;
    fmt::println("At line {}", __LINE__);


    // Create the fluid system
    std::string deckString1;
    {
        std::ifstream deckFile(filename);
        if (!deckFile) {
            throw std::runtime_error("Failed to open deck file: " + filename);
        }
        std::stringstream buffer;
        buffer << deckFile.rdbuf();
        deckString1 = buffer.str();
    }
    Opm::Parser parser;
    auto deck = parser.parseString(deckString1);
    auto python = std::make_shared<Opm::Python>();
    Opm::EclipseState eclState(deck);
    Opm::Schedule schedule(deck, eclState, python);
    fmt::println("At line {}", __LINE__);

    FluidSystem::initFromState(eclState, schedule);
    auto& dynamicFluidSystem = FluidSystem::getNonStaticInstance();
    auto dynamicGpuFluidSystemBuffer
        = ::Opm::gpuistl::copy_to_gpu<::Opm::gpuistl::GpuBuffer, double>(dynamicFluidSystem);
    auto dynamicGpuFluidSystemView = ::Opm::gpuistl::make_view<::Opm::gpuistl::GpuView, ::Opm::gpuistl::ValueAsPointer>(
        dynamicGpuFluidSystemBuffer);
    auto gpufluidstate
        = Opm::BlackOilFluidState<double, decltype(dynamicGpuFluidSystemView)>(dynamicGpuFluidSystemView);
    // Create MobArr
    double testValue = 0.5;
    // Create an array of Evaluations on CPU
    using MobArr = std::array<Evaluation, 2>;
    MobArr cpuMobArray;
    cpuMobArray[0] = Evaluation(testValue, 0);
    cpuMobArray[1] = Evaluation(testValue, 1);
    fmt::println("At line {}", __LINE__);
    fmt::println("At line {}", __LINE__);

    updateRelPermsFromFlowProblemBlackoilGpu<DirectionalMobilityPtr>
         <<<1, 1>>>(problemGpuView, cpuMobArray, gpufluidstate);
    fmt::println("At line {}", __LINE__);

    OPM_GPU_SAFE_CALL(cudaDeviceSynchronize());
    fmt::println("At line {}", __LINE__);
    #endif
}
