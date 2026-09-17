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

#define BOOST_TEST_MODULE TestLinearSolverRuntimeParameters

#include <boost/test/unit_test.hpp>

#include <opm/simulators/linalg/LinearSolverRuntimeParameters.hpp>
#include <opm/simulators/timestepping/SubStepCallback.hpp>

#include <string>
#include <vector>

BOOST_AUTO_TEST_CASE(SplitSingleName)
{
    const auto names = Opm::splitLinearSolverNames("cprw");
    BOOST_REQUIRE_EQUAL(names.size(), 1U);
    BOOST_CHECK_EQUAL(names[0], "cprw");
}

BOOST_AUTO_TEST_CASE(SplitList)
{
    const auto names = Opm::splitLinearSolverNames(" cprw, ilu0 ,dilu,,");
    const std::vector<std::string> expected {"cprw", "ilu0", "dilu"};
    BOOST_CHECK_EQUAL_COLLECTIONS(names.begin(), names.end(), expected.begin(), expected.end());
}

BOOST_AUTO_TEST_CASE(SplitEmpty)
{
    BOOST_CHECK(Opm::splitLinearSolverNames("").empty());
    BOOST_CHECK(Opm::splitLinearSolverNames(" , ").empty());
}

BOOST_AUTO_TEST_CASE(RuntimeParametersEmpty)
{
    Opm::LinearSolverRuntimeParameters parameters;
    BOOST_CHECK(parameters.empty());
    parameters.maxIterations = 50;
    BOOST_CHECK(!parameters.empty());
}

BOOST_AUTO_TEST_CASE(DecisionEmpty)
{
    Opm::SubStepCallbackDecision decision;
    BOOST_CHECK(decision.empty());
    decision.linearSolver.tolerance = 1e-3;
    BOOST_CHECK(!decision.empty());
    decision = Opm::SubStepCallbackDecision {};
    decision.dt = 3600.0;
    BOOST_CHECK(!decision.empty());
}
