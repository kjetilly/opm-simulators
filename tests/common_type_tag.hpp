#ifndef OPM_FLOW_SIMPLE_PROBLEM_HEADER_INCLUDED
#define OPM_FLOW_SIMPLE_PROBLEM_HEADER_INCLUDED
#include <opm/simulators/flow/Main.hpp>
#include <opm/material/fluidmatrixinteractions/EclMaterialLawManagerSimple.hpp>
#include <opm/models/discretization/common/tpfalinearizer.hh>
// do I need these?
#include <opm/simulators/flow/equil/EquilibrationHelpers.hpp>
#include <opm/simulators/flow/equil/InitStateEquil.hpp>
#include <tests/load_data.hpp>

namespace Opm {
    namespace Properties {
        namespace TTag {
            struct FlowSimpleProblem {
                using InheritsFrom = std::tuple<FlowProblem>;
            };
        }

        // Indices for two-phase gas-water.
        template<class TypeTag>
        struct Indices<TypeTag, TTag::FlowSimpleProblem>
        {
        private:
            // it is unfortunately not possible to simply use 'TypeTag' here because this leads
            // to cyclic definitions of some properties. if this happens the compiler error
            // messages unfortunately are *really* confusing and not really helpful.
            using BaseTypeTag = TTag::FlowProblem;
            using FluidSystem = GetPropType<BaseTypeTag, Properties::FluidSystem>;

        public:
            using type = BlackOilTwoPhaseIndices<getPropValue<TypeTag, Properties::EnableSolvent>(),
                                                getPropValue<TypeTag, Properties::EnableExtbo>(),
                                                getPropValue<TypeTag, Properties::EnablePolymer>(),
                                                getPropValue<TypeTag, Properties::EnableEnergy>(),
                                                getPropValue<TypeTag, Properties::EnableFoam>(),
                                                getPropValue<TypeTag, Properties::EnableBrine>(),
                                                /*PVOffset=*/0,
                                                /*disabledCompIdx=*/FluidSystem::oilCompIdx,
                                                getPropValue<TypeTag, Properties::EnableMICP>()>;
        };

        // SPE11C requires thermal/energy
        template<class TypeTag>
        struct EnableEnergy<TypeTag, TTag::FlowSimpleProblem> {
            static constexpr bool value = false;
        };

        // SPE11C requires dispersion
        template<class TypeTag>
        struct EnableDispersion<TypeTag, TTag::FlowSimpleProblem> {
            static constexpr bool value = true;
        };

        // Use the simple material law.
        template<class TypeTag>
        struct MaterialLaw<TypeTag, TTag::FlowSimpleProblem>
        {
        public:
            using Scalar = GetPropType<TypeTag, Properties::Scalar>;
            using FluidSystem = GetPropType<TypeTag, Properties::FluidSystem>;

            using Traits = ThreePhaseMaterialTraits<Scalar,
                                                    /*wettingPhaseIdx=*/FluidSystem::waterPhaseIdx,
                                                    /*nonWettingPhaseIdx=*/FluidSystem::oilPhaseIdx,
                                                    /*gasPhaseIdx=*/FluidSystem::gasPhaseIdx>;
            using EclMaterialLawManager = ::Opm::EclMaterialLawManagerSimple<Traits>;
            using type = typename EclMaterialLawManager::MaterialLaw;
        };

        // Use the TPFA linearizer.
        template<class TypeTag>
        struct Linearizer<TypeTag, TTag::FlowSimpleProblem> { using type = TpfaLinearizer<TypeTag>; };

        template<class TypeTag>
        struct LocalResidual<TypeTag, TTag::FlowSimpleProblem> { using type = BlackOilLocalResidualTPFA<TypeTag>; };

        // Diffusion.
        template<class TypeTag>
        struct EnableDiffusion<TypeTag, TTag::FlowSimpleProblem> { static constexpr bool value = true; };

        template<class TypeTag>
        struct EnableDisgasInWater<TypeTag, TTag::FlowSimpleProblem> { static constexpr bool value = true; };

        template<class TypeTag>
        struct EnableVapwat<TypeTag, TTag::FlowSimpleProblem> { static constexpr bool value = true; };

    };

}
#endif  // OPM_FLOW_SIMPLE_PROBLEM_HEADER_INCLUDED