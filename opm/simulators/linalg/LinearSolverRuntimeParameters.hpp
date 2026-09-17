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
#ifndef OPM_LINEAR_SOLVER_RUNTIME_PARAMETERS_HPP
#define OPM_LINEAR_SOLVER_RUNTIME_PARAMETERS_HPP

#include <optional>
#include <string>
#include <vector>

namespace Opm
{

/// \brief Linear solver settings that may be changed while a simulation is running.
///
/// Applied to the currently active solver setup through
/// \c AbstractISTLSolver::setRuntimeParameters().  Unset fields are left untouched.
struct LinearSolverRuntimeParameters {
    std::optional<double> tolerance; //!< required residual reduction
    std::optional<double> relaxedTolerance; //!< reduction accepted when the solver did not converge
    std::optional<int> maxIterations; //!< maximal number of linear iterations
    std::optional<int> cprReuseSetup; //!< --cpr-reuse-setup semantics
    std::optional<int> cprReuseInterval; //!< --cpr-reuse-interval semantics

    [[nodiscard]] bool empty() const
    {
        return !tolerance && !relaxedTolerance && !maxIterations && !cprReuseSetup
            && !cprReuseInterval;
    }
};

/// \brief Split a comma separated \c --linear-solver value into individual solver names.
///
/// Whitespace around the names is removed and empty entries are dropped.  A value without commas
/// yields a single entry.
[[nodiscard]] std::vector<std::string> splitLinearSolverNames(const std::string& value);

} // namespace Opm

#endif // OPM_LINEAR_SOLVER_RUNTIME_PARAMETERS_HPP
