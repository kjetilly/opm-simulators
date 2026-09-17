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
#ifndef OPM_SUB_STEP_CALLBACK_HPP
#define OPM_SUB_STEP_CALLBACK_HPP

#include <opm/simulators/linalg/LinearSolverRuntimeParameters.hpp>

#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace Opm
{

/// \brief State handed to a \ref SubStepCallback right before a substep is attempted.
///
/// All times are in seconds.  "Last" quantities describe the previous substep attempt of the
/// same report step (converged or not); they are zero/false for the first attempt of a run.
/// "Total" quantities accumulate over the whole simulation so far.
struct SubStepCallbackInfo {
    int reportStep = 0; //!< index of the current report step
    int subStep = 0; //!< number of substep attempts made in this report step
    int restarts = 0; //!< consecutive failed attempts preceding this one
    double time = 0.0; //!< simulation time at the start of the substep
    double reportStepStart = 0.0; //!< start time of the current report step
    double reportStepEnd = 0.0; //!< end time of the current report step
    double totalTime = 0.0; //!< end time of the whole simulation
    double proposedDt = 0.0; //!< substep length the time stepper is about to use
    double suggestedNextDt = 0.0; //!< controller suggestion stored for the next substep

    bool lastConverged = false;
    bool lastTimeStepRejected = false;
    double lastDt = 0.0;
    int lastNewtonIterations = 0;
    int lastLinearIterations = 0;
    int lastWellIterations = 0;
    double lastSolverTime = 0.0; //!< wall time of the last attempt (seconds)
    double lastAssembleTime = 0.0;
    double lastLinearSolveTime = 0.0;
    double lastLinearSolveSetupTime = 0.0;
    double lastUpdateTime = 0.0;
    std::string lastFailureCause;

    long totalNewtonIterations = 0;
    long totalLinearIterations = 0;
    long totalWastedNewtonIterations = 0; //!< iterations of failed attempts
    long totalWastedLinearIterations = 0;
    long totalSubSteps = 0; //!< converged substeps
    long totalFailedSubSteps = 0;
    double totalSolverTime = 0.0;

    int activeLinearSolver = 0;
    std::vector<std::string> linearSolvers; //!< names of the configured linear solver setups
    double linearSolverTolerance = 0.0;
    int linearSolverMaxIterations = 0;
    int newtonMaxIterations = 0;
    int newtonMinIterations = 0;
    double growthFactor = 0.0;
    double maxGrowth = 0.0;
    double restartFactor = 0.0;
    double maxTimeStep = 0.0;
};

/// \brief Overrides returned by a \ref SubStepCallback.  Unset fields keep the current values.
struct SubStepCallbackDecision {
    std::optional<double> dt; //!< substep length to attempt [s] (clamped to the report step)
    std::optional<int> linearSolverIndex; //!< index into SubStepCallbackInfo::linearSolvers
    LinearSolverRuntimeParameters linearSolver; //!< tolerance / max iterations / CPR reuse
    std::optional<int> newtonMaxIterations;
    std::optional<int> newtonMinIterations;
    std::optional<double> growthFactor; //!< time step growth factor after a recovered failure
    std::optional<double> maxGrowth; //!< maximal growth factor between substeps
    std::optional<double> restartFactor; //!< chop factor applied on failure
    std::optional<double> maxTimeStep; //!< maximal substep length [s]

    [[nodiscard]] bool empty() const
    {
        return !dt && !linearSolverIndex && linearSolver.empty() && !newtonMaxIterations
            && !newtonMinIterations && !growthFactor && !maxGrowth && !restartFactor
            && !maxTimeStep;
    }
};

/// \brief Callback invoked by \c AdaptiveTimeStepping before every substep attempt.
///
/// This is the hook used to steer solver and time stepping parameters at substep granularity from
/// outside the simulator (e.g. from Python through the pybind11 bindings).  In MPI runs it is
/// invoked on every rank and must return identical decisions on all ranks.
using SubStepCallback = std::function<SubStepCallbackDecision(const SubStepCallbackInfo&)>;

} // namespace Opm

#endif // OPM_SUB_STEP_CALLBACK_HPP
