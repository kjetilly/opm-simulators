/*
  Copyright 2023 SINTEF AS

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

#define BOOST_TEST_MODULE TestGpuOwnerOverlapCopy
#define BOOST_TEST_NO_MAIN

#include <boost/test/unit_test.hpp>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/owneroverlapcopy.hh>
#include <memory>
#include <opm/simulators/linalg/gpuistl/GpuOwnerOverlapCopy.hpp>
#include <opm/simulators/linalg/gpuistl/GpuVector.hpp>
#include <opm/simulators/linalg/gpuistl/detail/gpu_safe_call.hpp>
#include <opm/simulators/linalg/gpuistl/set_device.hpp>
#include <random>
#include <mpi.h>

bool
init_unit_test_func()
{
    return true;
}

int
main(int argc, char** argv)
{
    [[maybe_unused]] const auto& helper = Dune::MPIHelper::instance(argc, argv);
    int rank, totalRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &totalRanks);
    Opm::gpuistl::setDevice(rank, totalRanks);
    return boost::unit_test::unit_test_main(&init_unit_test_func, argc, argv);
}

BOOST_AUTO_TEST_CASE(TestProject)
{

    // We're going to have three points: Centered is owned, left and right is copied. We assume a periodic domain
    // ([0,1]/0~1)
    auto indexInfo = Dune::IndexInfoFromGrid<int, int>();
    indexInfo.addLocalIndex(std::make_tuple(0, 0, Dune::OwnerOverlapCopyAttributeSet::copy));
    indexInfo.addLocalIndex(std::make_tuple(1, 1, Dune::OwnerOverlapCopyAttributeSet::owner));
    indexInfo.addLocalIndex(std::make_tuple(2, 2, Dune::OwnerOverlapCopyAttributeSet::copy));

    auto ownerOverlapCopy = Dune::OwnerOverlapCopyCommunication<int>(indexInfo, MPI_COMM_WORLD);
    auto xCPU = std::vector<double> {{1.0, 2.0, 3.0}};
    auto xGPU = Opm::gpuistl::GpuVector<double>(xCPU);

    auto gpuComm = std::make_shared<Opm::gpuistl::GPUObliviousMPISender<double, 1, Dune::OwnerOverlapCopyCommunication<int>>>(ownerOverlapCopy);

    auto GpuOwnerOverlapCopy
        = Opm::gpuistl::GpuOwnerOverlapCopy<double, Dune::OwnerOverlapCopyCommunication<int>>(gpuComm);

    GpuOwnerOverlapCopy.project(xGPU);

    auto resultOfProject = xGPU.asStdVector();

    BOOST_CHECK_EQUAL(0.0, resultOfProject[0]);
    BOOST_CHECK_EQUAL(2.0, resultOfProject[1]);
    BOOST_CHECK_EQUAL(0.0, resultOfProject[2]);
}


BOOST_AUTO_TEST_CASE(TestDot)
{

    // We're going to have three points: Centered is owned, left and right is copied. We assume a periodic domain
    // ([0,1]/0~1)
    auto indexInfo = Dune::IndexInfoFromGrid<int, int>();
    indexInfo.addLocalIndex(std::make_tuple(0, 0, Dune::OwnerOverlapCopyAttributeSet::copy));
    indexInfo.addLocalIndex(std::make_tuple(1, 1, Dune::OwnerOverlapCopyAttributeSet::owner));
    indexInfo.addLocalIndex(std::make_tuple(2, 2, Dune::OwnerOverlapCopyAttributeSet::copy));

    indexInfo.addRemoteIndex(std::make_tuple(0, 0, Dune::OwnerOverlapCopyAttributeSet::copy));
    indexInfo.addRemoteIndex(std::make_tuple(0, 1, Dune::OwnerOverlapCopyAttributeSet::owner));
    indexInfo.addRemoteIndex(std::make_tuple(0, 2, Dune::OwnerOverlapCopyAttributeSet::copy));
    auto ownerOverlapCopy = Dune::OwnerOverlapCopyCommunication<int>(indexInfo, MPI_COMM_WORLD);
    auto xCPU = std::vector<double> {{1.0, 2.0, 3.0}};
    auto xGPU = Opm::gpuistl::GpuVector<double>(xCPU);

    auto gpuComm = std::make_shared<Opm::gpuistl::GPUObliviousMPISender<double, 1, Dune::OwnerOverlapCopyCommunication<int>>>(ownerOverlapCopy);

    auto GpuOwnerOverlapCopy
        = Opm::gpuistl::GpuOwnerOverlapCopy<double, Dune::OwnerOverlapCopyCommunication<int>>(gpuComm);

    double outputDune = -1.0;
    auto xDune = xGPU.asDuneBlockVector<1>();
    ownerOverlapCopy.dot(xDune, xDune, outputDune);

    double output = -1.0;
    GpuOwnerOverlapCopy.dot(xGPU, xGPU, output);


    BOOST_CHECK_EQUAL(outputDune, output);
}

namespace {

void checkMixedCpuGpuCommunication(Dune::OwnerOverlapCopyCommunication<int>& communication,
                                  const int gpuRank,
                                  const double expectedDot)
{
    using Communication = Dune::OwnerOverlapCopyCommunication<int>;
    using CpuVector = Dune::BlockVector<Dune::FieldVector<double, 3>>;
    using GpuVector = Opm::gpuistl::GpuVector<double>;
    const auto rank = communication.communicator().rank();
    const auto ranks = communication.communicator().size();
    auto values = CpuVector(ranks);
    values = -1.0;
    for (auto component = 0; component < 3; ++component) {
        values[rank][component] = 3 * rank + component + 1;
    }
    auto dot = 0.0;
    auto norm = 0.0;
    if (rank == gpuRank) {
        auto sender = std::make_shared<Opm::gpuistl::GPUObliviousMPISender<double, 3, Communication>>(
            communication);
        const auto gpuCommunication = Opm::gpuistl::GpuOwnerOverlapCopy<double, Communication>(sender);
        auto gpuValues = GpuVector(values);
        gpuCommunication.copyOwnerToAll(gpuValues, gpuValues);
        gpuCommunication.dot(gpuValues, gpuValues, dot);
        norm = gpuCommunication.norm(gpuValues);
        gpuValues.copyToHost(values);
    } else {
        communication.copyOwnerToAll(values, values);
        communication.dot(values, values, dot);
        norm = communication.norm(values);
    }
    BOOST_CHECK_CLOSE(dot, expectedDot, 1e-10);
    BOOST_CHECK_CLOSE(norm * norm, expectedDot, 1e-10);
    for (auto cell = 0; cell < ranks; ++cell) {
        for (auto component = 0; component < 3; ++component) {
            BOOST_CHECK_EQUAL(values[cell][component], 3 * cell + component + 1);
        }
    }
}

} // namespace

// Use three-component blocks, as in the thermal gas-water model, with each
// rank owning one block and holding copies of every other block. Exercise
// host-staged exchanges and reductions with both choices of the GPU rank.
BOOST_AUTO_TEST_CASE(TestMixedCpuGpuCommunication)
{
    using Communication = Dune::OwnerOverlapCopyCommunication<int>;
    using Attributes = Dune::OwnerOverlapCopyAttributeSet;
    const auto communicator = Dune::Communication<MPI_Comm>(MPI_COMM_WORLD);
    const auto rank = communicator.rank();
    const auto ranks = communicator.size();
    if (ranks < 2) {
        BOOST_TEST_MESSAGE("Mixed CPU/GPU communication requires at least two MPI ranks.");
        return;
    }

    auto indexInfo = Dune::IndexInfoFromGrid<int, int>();
    for (auto cell = 0; cell < ranks; ++cell) {
        indexInfo.addLocalIndex({cell, cell, cell == rank ? Attributes::owner : Attributes::copy});
        for (auto remoteRank = 0; remoteRank < ranks; ++remoteRank) {
            if (remoteRank != rank) {
                indexInfo.addRemoteIndex({remoteRank, cell,
                                         cell == remoteRank ? Attributes::owner : Attributes::copy});
            }
        }
    }
    auto communication = Communication(indexInfo, MPI_COMM_WORLD);
    auto expectedDot = 0.0;
    for (auto cell = 0; cell < ranks; ++cell) {
        for (auto component = 0; component < 3; ++component) {
            const auto value = static_cast<double>(3 * cell + component + 1);
            expectedDot += value * value;
        }
    }

    checkMixedCpuGpuCommunication(communication, 0, expectedDot);
    checkMixedCpuGpuCommunication(communication, ranks - 1, expectedDot);
}
