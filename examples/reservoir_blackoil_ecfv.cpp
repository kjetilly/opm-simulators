// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
// vi: set et ts=4 sw=4 sts=4:
/*
  Copyright (C) 2012-2013 by Andreas Lauser

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
*/
/*!
 * \file
 *
 * \brief Test for the reservoir problem using the black-oil model, the ECFV discretization
 *        and automatic differentiation.
 */
#include "config.h"

#include <ewoms/common/start.hh>
#include <ewoms/models/blackoil/blackoilmodel.hh>
#include <ewoms/disc/ecfv/ecfvdiscretization.hh>
#include "problems/reservoirproblem.hh"

namespace Ewoms {
namespace Properties {
NEW_TYPE_TAG(ReservoirBlackOilEcfvProblem, INHERITS_FROM(BlackOilModel, ReservoirBaseProblem));

// Select the element centered finite volume method as spatial discretization
SET_TAG_PROP(ReservoirBlackOilEcfvProblem, SpatialDiscretizationSplice, EcfvDiscretization);

// Use automatic differentiation to linearize the system of PDEs
SET_TAG_PROP(ReservoirBlackOilEcfvProblem, LocalLinearizerSplice, AutoDiffLocalLinearizer);
}}

int main(int argc, char **argv)
{
    typedef TTAG(ReservoirBlackOilEcfvProblem) ProblemTypeTag;
    return Ewoms::start<ProblemTypeTag>(argc, argv);
}
