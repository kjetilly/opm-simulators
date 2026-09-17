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
#ifndef OPM_PY_SUB_STEP_CALLBACK_HEADER_INCLUDED
#define OPM_PY_SUB_STEP_CALLBACK_HEADER_INCLUDED

#include <opm/simulators/flow/python/Pybind11Exporter.hpp>
#include <opm/simulators/timestepping/SubStepCallback.hpp>

#include <optional>
#include <string>

namespace Opm::Pybind
{

/// Build a \c SubStepCallbackDecision from the object a Python callback returned: \c None (no
/// change), a \c SubStepDecision instance, or a \c dict using the SubStepDecision property names.
inline SubStepCallbackDecision
subStepDecisionFromPython(const py::object& result)
{
    if (result.is_none()) {
        return {};
    }
    if (py::isinstance<SubStepCallbackDecision>(result)) {
        return result.cast<SubStepCallbackDecision>();
    }
    if (py::isinstance<py::dict>(result)) {
        SubStepCallbackDecision decision;
        const auto dict = result.cast<py::dict>();
        auto get = [&dict]<class T>(const char* key, std::optional<T>& target) {
            if (dict.contains(key) && !dict[key].is_none()) {
                target = dict[key].cast<T>();
            }
        };
        get("dt", decision.dt);
        get("linear_solver_index", decision.linearSolverIndex);
        get("linear_solver_tolerance", decision.linearSolver.tolerance);
        get("relaxed_linear_solver_tolerance", decision.linearSolver.relaxedTolerance);
        get("linear_solver_max_iterations", decision.linearSolver.maxIterations);
        get("cpr_reuse_setup", decision.linearSolver.cprReuseSetup);
        get("cpr_reuse_interval", decision.linearSolver.cprReuseInterval);
        get("newton_max_iterations", decision.newtonMaxIterations);
        get("newton_min_iterations", decision.newtonMinIterations);
        get("growth_factor", decision.growthFactor);
        get("max_growth", decision.maxGrowth);
        get("restart_factor", decision.restartFactor);
        get("max_time_step", decision.maxTimeStep);
        return decision;
    }
    throw py::type_error("substep callback must return None, a SubStepDecision or a dict");
}

/// Register the \c SubStepInfo and \c SubStepDecision Python classes in module \p m.
///
/// Registered module-local because every simulator module (BlackOil, GasWater, ...) exports them.
inline void
exportSubStepCallbackTypes(py::module& m)
{
    py::class_<SubStepCallbackInfo>(m,
                                    "SubStepInfo",
                                    py::module_local(),
                                    "State of the adaptive time stepper right before a substep "
                                    "attempt. All times are in seconds.")
        .def_readonly("report_step", &SubStepCallbackInfo::reportStep)
        .def_readonly("sub_step", &SubStepCallbackInfo::subStep)
        .def_readonly("restarts", &SubStepCallbackInfo::restarts)
        .def_readonly("time", &SubStepCallbackInfo::time)
        .def_readonly("report_step_start", &SubStepCallbackInfo::reportStepStart)
        .def_readonly("report_step_end", &SubStepCallbackInfo::reportStepEnd)
        .def_readonly("total_time", &SubStepCallbackInfo::totalTime)
        .def_readonly("proposed_dt", &SubStepCallbackInfo::proposedDt)
        .def_readonly("suggested_next_dt", &SubStepCallbackInfo::suggestedNextDt)
        .def_readonly("last_converged", &SubStepCallbackInfo::lastConverged)
        .def_readonly("last_time_step_rejected", &SubStepCallbackInfo::lastTimeStepRejected)
        .def_readonly("last_dt", &SubStepCallbackInfo::lastDt)
        .def_readonly("last_newton_iterations", &SubStepCallbackInfo::lastNewtonIterations)
        .def_readonly("last_linear_iterations", &SubStepCallbackInfo::lastLinearIterations)
        .def_readonly("last_well_iterations", &SubStepCallbackInfo::lastWellIterations)
        .def_readonly("last_solver_time", &SubStepCallbackInfo::lastSolverTime)
        .def_readonly("last_assemble_time", &SubStepCallbackInfo::lastAssembleTime)
        .def_readonly("last_linear_solve_time", &SubStepCallbackInfo::lastLinearSolveTime)
        .def_readonly("last_linear_solve_setup_time",
                      &SubStepCallbackInfo::lastLinearSolveSetupTime)
        .def_readonly("last_update_time", &SubStepCallbackInfo::lastUpdateTime)
        .def_readonly("last_failure_cause", &SubStepCallbackInfo::lastFailureCause)
        .def_readonly("total_newton_iterations", &SubStepCallbackInfo::totalNewtonIterations)
        .def_readonly("total_linear_iterations", &SubStepCallbackInfo::totalLinearIterations)
        .def_readonly("total_wasted_newton_iterations",
                      &SubStepCallbackInfo::totalWastedNewtonIterations)
        .def_readonly("total_wasted_linear_iterations",
                      &SubStepCallbackInfo::totalWastedLinearIterations)
        .def_readonly("total_sub_steps", &SubStepCallbackInfo::totalSubSteps)
        .def_readonly("total_failed_sub_steps", &SubStepCallbackInfo::totalFailedSubSteps)
        .def_readonly("total_solver_time", &SubStepCallbackInfo::totalSolverTime)
        .def_readonly("active_linear_solver", &SubStepCallbackInfo::activeLinearSolver)
        .def_readonly("linear_solvers", &SubStepCallbackInfo::linearSolvers)
        .def_readonly("linear_solver_tolerance", &SubStepCallbackInfo::linearSolverTolerance)
        .def_readonly("linear_solver_max_iterations",
                      &SubStepCallbackInfo::linearSolverMaxIterations)
        .def_readonly("newton_max_iterations", &SubStepCallbackInfo::newtonMaxIterations)
        .def_readonly("newton_min_iterations", &SubStepCallbackInfo::newtonMinIterations)
        .def_readonly("growth_factor", &SubStepCallbackInfo::growthFactor)
        .def_readonly("max_growth", &SubStepCallbackInfo::maxGrowth)
        .def_readonly("restart_factor", &SubStepCallbackInfo::restartFactor)
        .def_readonly("max_time_step", &SubStepCallbackInfo::maxTimeStep)
        .def("__repr__", [](const SubStepCallbackInfo& info) {
            return "SubStepInfo(report_step=" + std::to_string(info.reportStep) + ", sub_step="
                + std::to_string(info.subStep) + ", time=" + std::to_string(info.time)
                + ", proposed_dt=" + std::to_string(info.proposedDt) + ")";
        });

    py::class_<SubStepCallbackDecision>(m,
                                        "SubStepDecision",
                                        py::module_local(),
                                        "Overrides applied to the next substep attempt. Attributes "
                                        "left as None keep the current values.")
        .def(py::init<>())
        .def_readwrite("dt", &SubStepCallbackDecision::dt)
        .def_readwrite("linear_solver_index", &SubStepCallbackDecision::linearSolverIndex)
        .def_property(
            "linear_solver_tolerance",
            [](const SubStepCallbackDecision& d) { return d.linearSolver.tolerance; },
            [](SubStepCallbackDecision& d, std::optional<double> v) {
                d.linearSolver.tolerance = v;
            })
        .def_property(
            "relaxed_linear_solver_tolerance",
            [](const SubStepCallbackDecision& d) { return d.linearSolver.relaxedTolerance; },
            [](SubStepCallbackDecision& d, std::optional<double> v) {
                d.linearSolver.relaxedTolerance = v;
            })
        .def_property(
            "linear_solver_max_iterations",
            [](const SubStepCallbackDecision& d) { return d.linearSolver.maxIterations; },
            [](SubStepCallbackDecision& d, std::optional<int> v) {
                d.linearSolver.maxIterations = v;
            })
        .def_property(
            "cpr_reuse_setup",
            [](const SubStepCallbackDecision& d) { return d.linearSolver.cprReuseSetup; },
            [](SubStepCallbackDecision& d, std::optional<int> v) {
                d.linearSolver.cprReuseSetup = v;
            })
        .def_property(
            "cpr_reuse_interval",
            [](const SubStepCallbackDecision& d) { return d.linearSolver.cprReuseInterval; },
            [](SubStepCallbackDecision& d, std::optional<int> v) {
                d.linearSolver.cprReuseInterval = v;
            })
        .def_readwrite("newton_max_iterations", &SubStepCallbackDecision::newtonMaxIterations)
        .def_readwrite("newton_min_iterations", &SubStepCallbackDecision::newtonMinIterations)
        .def_readwrite("growth_factor", &SubStepCallbackDecision::growthFactor)
        .def_readwrite("max_growth", &SubStepCallbackDecision::maxGrowth)
        .def_readwrite("restart_factor", &SubStepCallbackDecision::restartFactor)
        .def_readwrite("max_time_step", &SubStepCallbackDecision::maxTimeStep)
        .def("empty", &SubStepCallbackDecision::empty);
}

} // namespace Opm::Pybind

#endif // OPM_PY_SUB_STEP_CALLBACK_HEADER_INCLUDED
