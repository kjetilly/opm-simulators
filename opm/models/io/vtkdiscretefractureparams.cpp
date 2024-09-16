// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
// vi: set et ts=4 sw=4 sts=4:
/*
  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 2 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.

  Consult the COPYING file in the top-level source directory of this
  module for the precise wording of the license and the list of
  copyright holders.
*/

#include <config.h>
#include <opm/models/io/vtkdiscretefractureparams.hpp>

#include <opm/models/utils/parametersystem.hpp>

namespace Opm {

void VtkDiscreteFractureParams::registerParameters()
{
    Parameters::Register<Parameters::VtkWriteFractureSaturations>
        ("Include the phase saturations in the VTK output files");
    Parameters::Register<Parameters::VtkWriteFractureMobilities>
        ("Include the phase mobilities in the VTK output files");
    Parameters::Register<Parameters::VtkWriteFractureRelativePermeabilities>
        ("Include the phase relative permeabilities in the "
         "VTK output files");
    Parameters::Register<Parameters::VtkWriteFracturePorosity>
        ("Include the porosity in the VTK output files");
    Parameters::Register<Parameters::VtkWriteFractureIntrinsicPermeabilities>
        ("Include the intrinsic permeability in the VTK output files");
    Parameters::Register<Parameters::VtkWriteFractureFilterVelocities>
        ("Include in the filter velocities of the phases in the VTK output files");
    Parameters::Register<Parameters::VtkWriteFractureVolumeFraction>
        ("Add the fraction of the total volume which is "
         "occupied by fractures in the VTK output");
}

void VtkDiscreteFractureParams::read()
{
    saturationOutput_ = Parameters::Get<Parameters::VtkWriteFractureSaturations>();
    mobilityOutput_ = Parameters::Get<Parameters::VtkWriteFractureMobilities>();
    relativePermeabilityOutput_ = Parameters::Get<Parameters::VtkWriteFractureRelativePermeabilities>();
    porosityOutput_ = Parameters::Get<Parameters::VtkWriteFracturePorosity>();
    intrinsicPermeabilityOutput_ = Parameters::Get<Parameters::VtkWriteFractureIntrinsicPermeabilities>();
    volumeFractionOutput_ = Parameters::Get<Parameters::VtkWriteFractureVolumeFraction>();
    velocityOutput_ = Parameters::Get<Parameters::VtkWriteFractureFilterVelocities>();
}

} // namespace Opm