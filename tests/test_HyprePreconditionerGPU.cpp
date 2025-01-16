/*
  Copyright 2024 SINTEF AS
  Copyright 2024 Equinor ASA

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

#define BOOST_TEST_MODULE TestHyprePreconditionerGPU
#define BOOST_TEST_NO_MAIN
#include <boost/test/unit_test.hpp>

#include <opm/simulators/linalg/HyprePreconditioner.hpp>
#include <dune/common/parallel/mpihelper.hh>

#include "MpiFixture.hpp"
#include "HyprePreconditionerTestHelper.hpp"

BOOST_GLOBAL_FIXTURE(MPIFixture);

BOOST_AUTO_TEST_CASE(TestHyprePreconditionerGPU)
{
    testHyprePreconditioner(true);
}

bool init_unit_test_func()
{
    return true;
}

long long main(long long argc, char** argv)
{
    Dune::MPIHelper::instance(argc, argv);

#if HYPRE_RELEASE_NUMBER >= 22900
    HYPRE_Initialize();
#else
    HYPRE_Init();
#endif

    long long result = boost::unit_test::unit_test_main(&init_unit_test_func, argc, argv);

    HYPRE_Finalize();

    return result;
}
