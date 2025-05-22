#ifndef OPM_LOAD_DATA_HPP
#define OPM_LOAD_DATA_HPP
#include <functional>
#include <opm/simulators/flow/FlowProblemBlackoilGpu.hpp>
template<class TypeTag>
using CallbackType = std::function<void(typename Opm::gpuistl::GpuProblem<TypeTag>::type&)>;

template<class TypeTag>
void loadData(int argc, char** argv, const CallbackType<TypeTag>& callback);
#endif