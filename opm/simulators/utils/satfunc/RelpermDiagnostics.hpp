/*
  Copyright 2015 Statoil ASA.

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

#ifndef OPM_RELPERMDIAGNOSTICS_HEADER_INCLUDED
#define OPM_RELPERMDIAGNOSTICS_HEADER_INCLUDED

#include <vector>
#include <utility>

#include <opm/material/fluidmatrixinteractions/EclEpsScalingPoints.hpp>

namespace Opm {

    class EclipseState;
    template <typename Grid> class LevelCartesianIndexMapper;
    class MiscTable;
    class MsfnTable;
    class SgcwmisTable;
    class SgfnTable;
    class SgofTable;
    class SgwfnTable;
    class SlgofTable;
    class Sof2Table;
    class Sof3Table;
    class SorwmisTable;
    class SsfnTable;
    class SwfnTable;
    class SwofTable;
    class GsfTable;
    class WsfTable;

    ///This class is intend to be a relperm diagnostics, to detect
    ///wrong input of relperm table and endpoints.
    class RelpermDiagnostics
    {
    public:
        ///This function is used to diagnosis relperm in
        ///eclipse data file. Errors and warings will be
        ///output if they're found.
        ///\param[in] eclState  eclipse state.
        ///\param[in] grid      unstructured grid.
        template <class LevelCartesianIndexMapper>
        void diagnosis(const EclipseState& eclState,
                       const LevelCartesianIndexMapper& levelCartesianIndexMapper);

    private:
        enum FluidSystem {
            OilWater,
            OilGas,
            WaterGas,
            BlackOil,
            Solvent
        };

        FluidSystem fluidSystem_;

        enum SaturationFunctionFamily {
            FamilyI,
            FamilyII,
            FamilyIII,
            NoFamily
        };

        SaturationFunctionFamily satFamily_;

        std::vector<EclEpsScalingPointsInfo<double> > unscaledEpsInfo_;
        std::vector<EclEpsScalingPointsInfo<double> > scaledEpsInfo_;


        ///Check the phase that used.
        /// return false if one-phase system
        bool phaseCheck_(const EclipseState& es);

        ///Check saturation family I and II.
        void satFamilyCheck_(const EclipseState& eclState);

        ///Check saturation tables.
        void tableCheck_(const EclipseState& eclState);

        ///Check endpoints in the saturation tables.
        void unscaledEndPointsCheck_(const EclipseState& eclState);

        template <class LevelCartesianIndexMapper>
        void scaledEndPointsCheck_(const EclipseState& eclState,
                                   const LevelCartesianIndexMapper& levelCartesianIndexMapper);

        ///For every table, need to deal with case by case.
        void swofTableCheck_(const SwofTable& swofTables,
                             const long long satnumIdx);
        void sgofTableCheck_(const SgofTable& sgofTables,
                             const long long satnumIdx);
        void slgofTableCheck_(const SlgofTable& slgofTables,
                              const long long satnumIdx);
        void swfnTableCheck_(const SwfnTable& swfnTables,
                             const long long satnumIdx);
        void sgfnTableCheck_(const SgfnTable& sgfnTables,
                             const long long satnumIdx);
        void wsfTableCheck_(const WsfTable& wsfTables,
                            const long long satnumIdx);
        void gsfTableCheck_(const GsfTable& gsfTables,
                            const long long satnumIdx);
        void sof3TableCheck_(const Sof3Table& sof3Tables,
                             const long long satnumIdx);
        void sof2TableCheck_(const Sof2Table& sof2Tables,
                             const long long satnumIdx);
        void sgwfnTableCheck_(const SgwfnTable& sgwfnTables,
                              const long long satnumIdx);
        ///Tables for solvent model
        void sgcwmisTableCheck_(const SgcwmisTable& sgcwmisTables,
                                const long long satnumIdx);
        void sorwmisTableCheck_(const SorwmisTable& sorwmisTables,
                                const long long satnumIdx);
        void ssfnTableCheck_(const SsfnTable& ssfnTables,
                             const long long satnumIdx);
        void miscTableCheck_(const MiscTable& miscTables,
                             const long long miscnumIdx);
        void msfnTableCheck_(const MsfnTable& msfnTables,
                             const long long satnumIdx);
    };

} //namespace Opm

#endif // OPM_RELPERMDIAGNOSTICS_HEADER_INCLUDED
