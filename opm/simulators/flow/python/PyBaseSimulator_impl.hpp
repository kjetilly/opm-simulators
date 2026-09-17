/*
  Copyright 2025 Equinor ASA.

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
#ifndef OPM_PY_BASE_SIMULATOR_IMPL_HEADER_INCLUDED
#define OPM_PY_BASE_SIMULATOR_IMPL_HEADER_INCLUDED

// Improve IDE experience
#ifndef OPM_PY_BASE_SIMULATOR_HEADER_INCLUDED
#include <config.h>
#include <opm/simulators/flow/python/PyBaseSimulator.hpp>
#endif

#include <opm/simulators/flow/python/PySubStepCallback.hpp>

#include <map>
#include <stdexcept>
#include <string>
#include <utility>

namespace py = pybind11;

namespace Opm::Pybind {

template<class TypeTag>
PyBaseSimulator<TypeTag>::PyBaseSimulator(const std::string& deck_filename,
                                          const std::vector<std::string>& args)
    : deck_filename_{deck_filename}
    , args_{args}
{
}

template<class TypeTag>
PyBaseSimulator<TypeTag>::
PyBaseSimulator(std::shared_ptr<Deck> deck,
                std::shared_ptr<EclipseState> state,
                std::shared_ptr<Schedule> schedule,
                std::shared_ptr<SummaryConfig> summary_config,
                const std::vector<std::string>& args)
    : deck_{std::move(deck)}
    , eclipse_state_{std::move(state)}
    , schedule_{std::move(schedule)}
    , summary_config_{std::move(summary_config)}
    , args_{args}
{
}

// Public methods alphabetically sorted
// ------------------------------------
template<class TypeTag>
void PyBaseSimulator<TypeTag>::advance(int report_step)
{
    while (currentStep() < report_step) {
        step();
    }
}

template<class TypeTag>
bool PyBaseSimulator<TypeTag>::checkSimulationFinished()
{
    return getFlowMain().getSimTimer()->done();
}

// This returns the report step number that will be executed next time step()
//   is called.
template<class TypeTag>
int PyBaseSimulator<TypeTag>::currentStep()
{
    return getFlowMain().getSimTimer()->currentStepNum();
    // NOTE: this->simulator_->episodeIndex() would also return the current
    // report step number, but this number is always delayed by 1 step relative
    // to this->flow_main_->getSimTimer()->currentStepNum()
    // See details in runStep() in file SimulatorFullyImplicitBlackoilEbos.hpp
}

template<class TypeTag>
py::array_t<double>
PyBaseSimulator<TypeTag>::getCellVolumes()
{
    auto vector = getMaterialState().getCellVolumes();
    return py::array(vector.size(), vector.data());
}

template<class TypeTag>
double PyBaseSimulator<TypeTag>::getDT()
{
    return getFlowMain().getPreviousReportStepSize();
}

template<class TypeTag>
py::array_t<double>
PyBaseSimulator<TypeTag>::getPorosity()
{
    auto vector = getMaterialState().getPorosity();
    return py::array(vector.size(), vector.data());
}

template<class TypeTag>
py::array_t<double>
PyBaseSimulator<TypeTag>::
getFluidStateVariable(const std::string& name) const
{
    auto vector = getFluidState().getFluidStateVariable(name);
    return py::array(vector.size(), vector.data());
}

template<class TypeTag>
py::array_t<double>
PyBaseSimulator<TypeTag>::
getPrimaryVariable(const std::string& variable) const
{
    auto vector = getFluidState().getPrimaryVariable(variable);
    return py::array(vector.size(), vector.data());
}

template<class TypeTag>
py::array_t<int>
PyBaseSimulator<TypeTag>::
getPrimaryVarMeaning(const std::string& variable) const
{
    auto vector = getFluidState().getPrimaryVarMeaning(variable);
    return py::array(vector.size(), vector.data());
}

template<class TypeTag>
std::map<std::string, int>
PyBaseSimulator<TypeTag>::
getPrimaryVarMeaningMap(const std::string& variable) const
{

    return getFluidState().getPrimaryVarMeaningMap(variable);
}

template<class TypeTag>
void PyBaseSimulator<TypeTag>::setPorosity(PyCArray array)
{
    std::size_t size_ = array.size();
    const double *poro = array.data();
    getMaterialState().setPorosity(poro, size_);
}

template<class TypeTag>
void
PyBaseSimulator<TypeTag>::
setPrimaryVariable(const std::string& variable,
                   PyCArray array)
{
    std::size_t size_ = array.size();
    const double *data = array.data();
    getFluidState().setPrimaryVariable(variable, data, size_);
}

template<class TypeTag>
void PyBaseSimulator<TypeTag>::
setupMpi(bool mpi_init, bool mpi_finalize)
{
    if (this->has_run_init_) {
        throw std::logic_error("mpi_init() called after step_init()");
    }
    this->mpi_init_ = mpi_init;
    this->mpi_finalize_ = mpi_finalize;
}

template<class TypeTag>
int PyBaseSimulator<TypeTag>::step()
{
    if (!this->has_run_init_) {
        throw std::logic_error("step() called before step_init()");
    }
    if (this->has_run_cleanup_) {
        throw std::logic_error("step() called after step_cleanup().");
    }
    if(checkSimulationFinished()) {
        throw std::logic_error("step() called, but simulation is done");
    }
    auto result = getFlowMain().executeStep();
    return result;
}

template<class TypeTag>
int PyBaseSimulator<TypeTag>::stepCleanup()
{
    this->has_run_cleanup_ = true;
    return getFlowMain().executeStepsCleanup();
}

template<class TypeTag>
int PyBaseSimulator<TypeTag>::stepInit()
{
    if (this->has_run_init_) {
        // Running step_init() multiple times is not implemented yet,
        if (this->has_run_cleanup_) {
            throw std::logic_error("step_init() called again");
        }
        else {
            return EXIT_SUCCESS;
        }
    }
    if (this->deck_) {
        this->main_ = std::make_unique<PyMain<TypeTag>>(
            this->deck_->getDataFile(),
            this->eclipse_state_,
            this->schedule_,
            this->summary_config_,
            this->mpi_init_,
            this->mpi_finalize_
        );
    }
    else {
        this->main_ = std::make_unique<PyMain<TypeTag>>(
            this->deck_filename_,
            this->mpi_init_,
            this->mpi_finalize_
        );
    }
    this->main_->setArguments(args_);
    int exit_code = EXIT_SUCCESS;
    this->flow_main_ = this->main_->initFlowBlackoil(exit_code);
    if (this->flow_main_) {
        const int result = this->flow_main_->executeInitStep();
        this->has_run_init_ = true;
        this->simulator_ = this->flow_main_->getSimulatorPtr();
        this->fluid_state_ = std::make_unique<PyFluidState<TypeTag>>(this->simulator_);
        this->material_state_ = std::make_unique<PyMaterialState<TypeTag>>(this->simulator_);
        installSubStepCallback_();
        return result;
    }
    else {
        return exit_code;
    }
}

template<class TypeTag>
int PyBaseSimulator<TypeTag>::run()
{
    auto main_object = Main( this->deck_filename_ );
    return main_object.runStatic<TypeTag>();
}

// Private methods
// ---------------
template<class TypeTag>
FlowMain<TypeTag>&
PyBaseSimulator<TypeTag>::getFlowMain() const
{
    if (this->flow_main_) {
        return *this->flow_main_;
    }
    else {
        throw std::runtime_error(
           "BlackOilSimulator not initialized: "
           "Cannot get reference to FlowMain object"
        );
    }
}

template<class TypeTag>
PyFluidState<TypeTag>&
PyBaseSimulator<TypeTag>::getFluidState() const
{
    if (this->fluid_state_) {
        return *this->fluid_state_;
    }
    else {
        throw std::runtime_error(
            "BlackOilSimulator not initialized: "
            "Cannot get reference to fluid state object"
        );
    }
}

template<class TypeTag>
PyMaterialState<TypeTag>&
PyBaseSimulator<TypeTag>::getMaterialState() const
{
    if (this->material_state_) {
        return *this->material_state_;
    }
    else {
        throw std::runtime_error(
            "BlackOilSimulator not initialized: "
            "Cannot get reference to material state object"
        );
    }
}

template<class TypeTag>
void PyBaseSimulator<TypeTag>::setSubStepCallback(py::object callback)
{
    if (!callback.is_none() && !py::isinstance<py::function>(callback) && !py::hasattr(callback, "__call__")) {
        throw py::type_error("substep callback must be callable or None");
    }
    this->sub_step_callback_ = std::move(callback);
    if (this->has_run_init_) {
        installSubStepCallback_();
    }
}

template<class TypeTag>
void PyBaseSimulator<TypeTag>::clearSubStepCallback()
{
    setSubStepCallback(py::none());
}

template<class TypeTag>
std::map<std::string, double> PyBaseSimulator<TypeTag>::getSubStepTotals() const
{
    std::map<std::string, double> totals;
    if (!this->flow_main_ || !this->flow_main_->getStepDriverPtr()) {
        return totals;
    }
    const auto t = this->flow_main_->getStepDriverPtr()->subStepTotals();
    totals["newton_iterations"] = static_cast<double>(t.newtonIterations);
    totals["linear_iterations"] = static_cast<double>(t.linearIterations);
    totals["wasted_newton_iterations"] = static_cast<double>(t.wastedNewtonIterations);
    totals["wasted_linear_iterations"] = static_cast<double>(t.wastedLinearIterations);
    totals["sub_steps"] = static_cast<double>(t.subSteps);
    totals["failed_sub_steps"] = static_cast<double>(t.failedSubSteps);
    totals["solver_time"] = t.solverTime;
    return totals;
}

template<class TypeTag>
void PyBaseSimulator<TypeTag>::installSubStepCallback_()
{
    auto* driver = this->flow_main_ ? this->flow_main_->getStepDriverPtr() : nullptr;
    if (!driver) {
        return;
    }
    if (this->sub_step_callback_.is_none()) {
        driver->setSubStepCallback(SubStepCallback {});
        return;
    }
    // The GIL is held while step() executes, so calling back into Python here is safe.
    py::object callback = this->sub_step_callback_;
    driver->setSubStepCallback([callback](const SubStepCallbackInfo& info) {
        return subStepDecisionFromPython(callback(info));
    });
}

}  // namespace Opm::Pybind

#endif // OPM_PY_BASE_SIMULATOR_IMPL_HEADER_INCLUDED
