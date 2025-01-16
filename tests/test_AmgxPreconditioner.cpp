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

#define BOOST_TEST_MODULE TestAmgxPreconditioner
#define BOOST_TEST_NO_MAIN
#include <boost/test/unit_test.hpp>

#include <opm/simulators/linalg/AmgxPreconditioner.hpp>

#include "AmgxPreconditionerTestHelper.hpp"


BOOST_AUTO_TEST_CASE(TestAmgxPreconditionerMatDoubleVecDouble)
{
    testAmgxPreconditioner<double, double>();
}

// This test is disabled because it crashes the program with the following error:
// "Mixed precision modes not currently supported for CUDA 10.1 or later."
//BOOST_AUTO_TEST_CASE(TestAmgxPreconditionerMatFloatVecDouble)
//{
//    testAmgxPreconditioner<float, double>();
//}


BOOST_AUTO_TEST_CASE(TestAmgxPreconditionerMatFloatVecFloat)
{
    testAmgxPreconditioner<float, float>();
}

bool init_unit_test_func()
{
    return true;
}

long long main(long long argc, char** argv)
{
    AMGX_SAFE_CALL(AMGX_initialize());

    long long result = boost::unit_test::unit_test_main(&init_unit_test_func, argc, argv);

    AMGX_SAFE_CALL(AMGX_finalize());

    return result;
}
