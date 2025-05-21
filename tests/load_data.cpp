#include "config.h"
#include <tests/common_type_tag.hpp>

template <class TypeTag>
void loadData(int argc, char **argv) {
    auto mainObject = Opm::Main(argc, argv);
    //mainObject.runStatic<TypeTag>();
    auto mainFlow = mainObject.gimmeFlowMain<TypeTag>();
    mainFlow->execute();

    auto simulator = mainFlow->getSimulator();

    auto& problem = simulator->problem();
}
using namespace Opm::Properties::TTag;
template void loadData<FlowSimpleProblem>(int argc, char **argv);