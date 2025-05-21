#ifndef OPM_LOAD_DATA_HPP
#define OPM_LOAD_DATA_HPP
#include <functional>
template<class TypeTag>
void loadData(int argc, char** argv, std::function<void(typename Opm::GetPropType<TypeTag, Opm::Properties::Problem>&)> callback);
#endif