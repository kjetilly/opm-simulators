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
#include <opm/models/utils/start.hh>
#include <opm/simulators/flow/SimulatorFullyImplicitBlackoil.cpp>
#include <opm/simulators/flow/FlowUtils.cpp>
#include <opm/simulators/flow/BlackoilModelParameters.cpp>
#include <opm/simulators/flow/FlowGenericProblem.cpp>
#include <opm/models/parallel/threadmanager.cpp>
#include <opm/simulators/flow/SolutionContainers.cpp>
#include <opm/simulators/flow/MICPContainer.cpp>
#include <opm/simulators/timestepping/EclTimeSteppingParams.cpp>
#include <opm/models/utils/parametersystem.cpp>
#include <opm/simulators/flow/GenericOutputBlackoilModule.cpp>
#include <opm/simulators/flow/LogOutputHelper.cpp>
#include <opm/simulators/flow/MechContainer.cpp>
#include <opm/simulators/flow/FIPContainer.cpp>
#include <opm/simulators/flow/InterRegFlows.cpp>
#include <opm/simulators/flow/FlowsContainer.cpp>
#include <opm/simulators/flow/ExtboContainer.cpp>
#include <opm/simulators/flow/RFTContainer.cpp>
#include <opm/simulators/flow/RSTConv.cpp>
#include <opm/simulators/flow/TracerContainer.cpp>
#include <opm/simulators/flow/NonlinearSolver.cpp>
#include <opm/models/utils/timer.cpp>
#include <opm/models/io/vtkmultiphaseparams.cpp>
#include <opm/simulators/utils/VoigtArray.cpp>
#include <opm/models/io/vtkcompositionparams.cpp>
#include <opm/models/io/vtkblackoilparams.cpp>
#include <opm/models/io/vtktemperatureparams.cpp>
#include <opm/simulators/linalg/setupPropertyTree.cpp>
#include <opm/simulators/utils/PressureAverage.cpp>
#include <opm/simulators/timestepping/AdaptiveTimeStepping.cpp>
#include <opm/simulators/timestepping/TimeStepControl.cpp>
#include <opm/simulators/flow/FlowGenericVanguard.cpp>
#include <opm/simulators/flow/RegionPhasePVAverage.cpp>
#include <opm/simulators/utils/ParallelFileMerger.cpp>
#include <opm/simulators/wells/BlackoilWellModelRestart.cpp>
#include <opm/simulators/wells/WellProdIndexCalculator.cpp>
#include <opm/simulators/wells/BlackoilWellModelGeneric.cpp>
#include <opm/output/eclipse/LinearisedOutputTable.cpp>
#include <opm/simulators/wells/BlackoilWellModelGuideRates.cpp>
#include <opm/simulators/wells/MSWellHelpers.cpp>
#include <opm/simulators/wells/BlackoilWellModelWBP.cpp>
#include <opm/simulators/linalg/gpuistl_hip/detail/preconditionerKernels/JacKernels.hip>

#include <opm/output/eclipse/AggregateGroupData.cpp>
#include <opm/simulators/wells/WGState.cpp>
#include <opm/output/eclipse/LoadRestart.cpp>
#include <opm/output/eclipse/EclipseIO.cpp>
#include <opm/output/eclipse/ActiveIndexByColumns.cpp>
#include <opm/simulators/wells/ParallelWellInfo.cpp>
#include <opm/simulators/wells/WellGroupConstraints.cpp>

#include <opm/simulators/wells/GroupState.cpp>
#include <opm/simulators/wells/MultisegmentWell_impl.hpp>
#include <opm/simulators/wells/GroupEconomicLimitsChecker.cpp>
#include <opm/simulators/wells/WellGroupHelpers.cpp>
#include <opm/output/eclipse/Summary.cpp>

#include <opm/io/eclipse/EclOutput.cpp>
#include <opm/json/JsonObject.cpp>
#include <opm/simulators/wells/GasLiftStage2.cpp>
#include <opm/simulators/wells/WellState.cpp>
#include <opm/simulators/wells/TargetCalculator.cpp>

///////////
// Add these .cpp includes after your existing ones (around line 69):

// For ConvergenceOutputConfiguration errors:
#include <opm/simulators/flow/ConvergenceOutputConfiguration.cpp>

// For SimulatorReport error
// For getTtyWidth and breakLines errors:
#include <opm/common/utility/OpmInputError.cpp>
#include <opm/common/utility/String.cpp>

// For PropertyTree errors:
#include <opm/simulators/linalg/PropertyTree.cpp>

// For AdaptiveSimulatorTimer errors:
#include <opm/simulators/timestepping/AdaptiveSimulatorTimer.cpp>

// For Main class errors:
#include <opm/models/utils/start.hh>

// For VtkDiffusionParams error:
#include <opm/models/io/vtkdiffusionparams.cpp>

// For registerFlowProblemParameters error:
#include <opm/simulators/flow/FlowProblemParameters.cpp>

// For FlowLinearSolverParameters error:
#include <opm/simulators/linalg/FlowLinearSolverParameters.cpp>

// For NewtonMethodParams error:
#include <opm/models/nonlinear/newtonmethodparams.cpp>
////


#include <opm/simulators/flow/GenericTracerModel_impl.hpp>

#include <opm/simulators/flow/MixingRateControls.cpp>

// Add these .cpp includes after your existing ones (around line 69):
// For getTtyWidth and breakLines errors:
#include <opm/common/utility/OpmInputError.cpp>
#include <opm/common/utility/String.cpp>

// For GenericCpGridVanguard errors:
#include <opm/simulators/flow/GenericCpGridVanguard.cpp>

// For Transmissibility errors:
#include <opm/simulators/flow/Transmissibility.cpp>

// For EclGenericWriter errors:
#include <opm/simulators/flow/EclGenericWriter.cpp>
#include <opm/simulators/flow/ValidationFunctions.cpp>
#include <opm/simulators/utils/PartiallySupportedFlowKeywords.cpp>
#include <opm/simulators/flow/KeywordValidation.cpp>


#include <opm/models/parallel/tasklets.cpp>
#include <opm/simulators/utils/MPIPacker.cpp>
#include <opm/models/blackoil/blackoilnewtonmethodparams.cpp>
#include <opm/models/utils/simulatorutils.cpp>
#include <opm/simulators/linalg/ISTLSolver.cpp>
#include <opm/simulators/linalg/ExtractParallelGridInformationToISTL.cpp>
#include <opm/simulators/utils/DeferredLogger.cpp>
#include <opm/simulators/linalg/FlexibleSolver_impl.hpp>
#include <opm/simulators/linalg/PreconditionerFactory_impl.hpp>
#include <opm/simulators/linalg/ParallelOverlappingILU0_impl.hpp>
#include <opm/simulators/linalg/MILU.cpp>
#include <opm/simulators/timestepping/SimulatorReport.cpp>
#include <opm/models/utils/terminal.cpp>

#include <opm/simulators/utils/UnsupportedFlowKeywords.cpp>
#include <opm/simulators/utils/FullySupportedFlowKeywords.cpp>

#include <opm/simulators/utils/readDeck.cpp>
#include <opm/simulators/linalg/ParallelIstlInformation.cpp>
#include <opm/simulators/flow/Main.cpp>
#include <opm/simulators/utils/SetupPartitioningParams.cpp>
#include <opm/simulators/utils/ParallelEclipseState.cpp>
#include <opm/simulators/utils/ParallelSerialization.cpp>
#include <opm/simulators/flow/CollectDataOnIORank.cpp>
#include <opm/simulators/utils/gatherDeferredLogger.cpp>
#include <opm/simulators/linalg/gpuistl_hip/GpuVector.cpp>
#include <opm/simulators/linalg/gpuistl_hip/GpuSeqILU0.cpp>
#include <opm/simulators/linalg/gpuistl_hip/GpuDILU.cpp>
#include <opm/simulators/linalg/gpuistl_hip/OpmGpuILU0.cpp>

#include <opm/simulators/linalg/gpuistl_hip/detail/vector_operations.hip>
#include <opm/simulators/linalg/gpuistl_hip/detail/CuBlasHandle.cpp>
#include <opm/simulators/linalg/gpuistl_hip/detail/CuSparseHandle.cpp>

#include <opm/simulators/linalg/gpuistl_hip/device_management.cpp>
#include <opm/simulators/linalg/gpuistl_hip/set_device.cpp>
#include <opm/simulators/linalg/gpuistl_hip/detail/preconditionerKernels/ILU0Kernels.hip>
#include <opm/simulators/linalg/gpuistl_hip/detail/preconditionerKernels/DILUKernels.hip>
#include <opm/simulators/linalg/gpuistl_hip/detail/gpusparse_matrix_operations.hip>
#include <opm/simulators/linalg/gpuistl_hip/GpuJac.cpp>
#include <opm/models/io/vtkprimaryvarsparams.cpp>
#include <opm/models/blackoil/blackoilbrineparams.cpp>
#include <opm/simulators/utils/satfunc/RelpermDiagnostics.cpp>
#include <opm/simulators/utils/phaseUsageFromDeck.cpp>
#include <opm/models/blackoil/blackoilsolventparams.cpp>
#include <opm/models/blackoil/blackoilpolymerparams.cpp>
#include <opm/models/blackoil/blackoilmicpparams.cpp>
#include <opm/models/blackoil/blackoilfoamparams.cpp>
#include <opm/models/blackoil/blackoilextboparams.cpp>
#include <opm/simulators/flow/ActionHandler.cpp>
#include <opm/simulators/flow/GenericThresholdPressure_impl.hpp>
#include <opm/simulators/flow/Banners.cpp>


#include <opm/simulators/wells/WellInterfaceGeneric.cpp>
#include <opm/simulators/wells/GlobalWellInfo.cpp>
#include <opm/simulators/wells/BlackoilWellModelConstraints.cpp>
#include <opm/simulators/wells/ALQState.cpp>
#include <opm/simulators/wells/PerfData.cpp>
#include <opm/simulators/wells/WellFilterCake.cpp>
#include <opm/simulators/wells/SingleWellState.cpp>
#include <opm/simulators/wells/ParallelWBPCalculation.cpp>
#include <opm/simulators/wells/StandardWellEquations.cpp>
#include <opm/simulators/wells/WellBhpThpCalculator.cpp>
#include <opm/simulators/wells/RatioCalculator.cpp>
#include <opm/simulators/wells/MultisegmentWellGeneric.cpp>
#include <opm/simulators/wells/MultisegmentWellEquations.cpp>
#include <opm/simulators/wells/SegmentState.cpp>
#include <opm/simulators/wells/MultisegmentWellAssemble.cpp>
#include <opm/simulators/wells/VFPInjProperties.cpp>
#include <opm/simulators/wells/VFPProdProperties.cpp>
#include <opm/simulators/utils/satfunc/SatfuncConsistencyCheckManager.cpp>
#include <opm/simulators/flow/SimulatorSerializer.cpp>
#include <opm/simulators/flow/SimulatorConvergenceOutput.cpp>
#include <opm/simulators/utils/HDF5File.cpp>
#include <opm/simulators/timestepping/SimulatorTimer.cpp>
#include <opm/simulators/flow/ReservoirCouplingSlave.cpp>
#include <opm/simulators/flow/ReservoirCouplingMaster.cpp>
#include <opm/simulators/flow/BlackoilModelConvergenceMonitor.cpp>
#include <opm/simulators/timestepping/AdaptiveTimeStepping_impl.hpp>
#include <opm/simulators/wells/GasLiftSingleWellGeneric.cpp>
#include <opm/simulators/wells/GasLiftGroupInfo.cpp>
#include <opm/simulators/wells/BlackoilWellModelNldd.cpp>
#include <opm/simulators/wells/BlackoilWellModelGasLift.cpp>
#include <opm/simulators/wells/WellGroupControls.cpp>
#include <opm/simulators/timestepping/SimulatorTimerInterface.cpp>

#include <opm/simulators/linalg/gpuistl_hip/GpuSparseMatrix.cpp>
#include <opm/simulators/flow/SimulatorReportBanners.cpp>
#include <opm/simulators/flow/NlddReporting.cpp>
#include <opm/simulators/utils/ParallelRestart.cpp>
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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Automatically added includes for undefined symbols
// Added at datetime: 2025-05-29 14:02:31.923833
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include <opm/simulators/utils/HDF5Serializer.cpp>
#include <opm/simulators/flow/ExtraConvergenceOutputThread.cpp>
#include <opm/simulators/flow/ReservoirCouplingSpawnSlaves.cpp>
#include <opm/simulators/wells/ParallelPAvgCalculator.cpp>
#include <opm/simulators/wells/WellTest.cpp>
#include <opm/simulators/wells/ConnFiltrateData.cpp>
#include <opm/simulators/wells/ParallelPAvgDynamicSourceData.cpp>
#include <opm/simulators/wells/WellHelpers.cpp>
#include <opm/simulators/wells/VFPHelpers.cpp>
#include <opm/simulators/utils/satfunc/SatfuncConsistencyChecks.cpp>
#include <opm/simulators/wells/GasLiftCommon.cpp>
#include <opm/simulators/wells/FractionCalculator.cpp>
#include <opm/common/utility/MemPacker.cpp>
#include <opm/simulators/utils/satfunc/UnscaledSatfuncCheckPoint.cpp>
#include <opm/simulators/utils/satfunc/ScaledSatfuncCheckPoint.cpp>
#include <opm/simulators/utils/satfunc/PhaseCheckBase.cpp>
#include <opm/simulators/utils/satfunc/OilPhaseConsistencyChecks.cpp>
#include <opm/simulators/utils/satfunc/GasPhaseConsistencyChecks.cpp>
#include <opm/simulators/utils/satfunc/WaterPhaseConsistencyChecks.cpp>
#include <opm/simulators/utils/satfunc/ThreePointHorizontalConsistencyChecks.cpp>
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Automatically added includes for undefined symbols
// Added at datetime: 2025-05-29 18:41:27.301234
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include <opm/simulators/wells/BlackoilWellModel_impl.hpp>
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Automatically added includes for undefined symbols
// Added at datetime: 2025-05-29 19:48:19.555860
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include <opm/simulators/utils/BlackoilPhases.cpp>


// Sonnet
// Time-related implementations
#include "opm/common/utility/TimeService.cpp"
#include "opm/grid/utility/StopWatch.cpp"

// Comparison utilities
#include "opm/common/utility/numeric/cmp.hpp"
#include "opm/input/eclipse/Schedule/UDQ/UDQFunction.cpp"

// Error handling
#include "opm/simulators/utils/DeferredLoggingErrorHelpers.hpp"

// Reservoir coupling
#include "opm/simulators/flow/ReservoirCoupling.cpp"
#include "opm/input/eclipse/Schedule/ResCoup/MasterGroup.cpp"
#include "opm/input/eclipse/Schedule/ResCoup/Slaves.cpp"

// Scheduling
#include "opm/input/eclipse/Schedule/Schedule.cpp"


///////////
#include <opm/simulators/wells/WellInterfaceFluidSystem.cpp>
#include <opm/simulators/flow/partitionCells.cpp>
#include <opm/simulators/utils/ComponentName.cpp>
#include <opm/simulators/flow/equil/InitStateEquil_impl.hpp>

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

#define BOOST_CHECK_EQUAL(a, b) \
    if (!((a) == (b))) { \
        std::cerr << "Check failed: " << #a << " == " << #b << " at line " << __LINE__ << std::endl; \
        std::exit(EXIT_FAILURE); \
    }

int main(int argc, char** argv)
{
    MPIHelper::instance(argc,argv);
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
    }
        std::cout << __LINE__ << std::endl;

    using TypeTag = Opm::Properties::TTag::FlowSimpleProblem;
        std::cout << __LINE__ << std::endl;

//    auto mainObject = Opm::Main(argv2.size() - 1, argv2.data());
        auto mainObject = Opm::Main(filename, false, false);

        std::cout << __LINE__ << std::endl;

    // mainObject.runStatic<TypeTag>();
    auto mainFlow = mainObject.gimmeFlowMain<TypeTag>();
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
    return 0;
}////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Automatically added includes for undefined symbols
// Added at datetime: 2025-05-30 15:48:02.238381
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include <opm/simulators/timestepping/gatherConvergenceReport.cpp>
#include <opm/simulators/utils/SerializationPackers.cpp>////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Automatically added includes for undefined symbols
// Added at datetime: 2025-05-31 20:49:43.015556
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include <opm/simulators/utils/compressPartition.cpp>
#include <opm/simulators/utils/ParallelNLDDPartitioningZoltan.cpp>
#include <opm/simulators/wells/WellConstraints.cpp>
#include <opm/simulators/flow/equil/EquilibrationHelpers_impl.hpp>
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Automatically added includes for undefined symbols
// Added at datetime: 2025-06-01 13:48:16.824583
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include <opm/simulators/wells/WellAssemble.cpp>
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Automatically added includes for undefined symbols
// Added at datetime: 2025-06-01 14:39:08.249201
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include <opm/simulators/timestepping/ConvergenceReport.cpp>
#include <opm/simulators/wells/MultisegmentWellPrimaryVariables.cpp>
#include <opm/simulators/wells/WellInterfaceIndices.cpp>
#include <opm/simulators/wells/RateConverter.cpp>
#include <opm/simulators/wells/StandardWellEval.cpp>
#include <opm/simulators/wells/StandardWellPrimaryVariables.cpp>
#include <opm/simulators/wells/StandardWellConnections.cpp>
#include <opm/simulators/wells/StandardWellAssemble.cpp>
#include <opm/simulators/wells/MultisegmentWellEval.cpp>
#include <opm/simulators/wells/MultisegmentWellSegments.cpp>
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Automatically added includes for undefined symbols
// Added at datetime: 2025-06-01 14:57:11.752623
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include <opm/simulators/wells/WellConvergence.cpp>
