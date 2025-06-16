/*
  Copyright 2025 Equinor ASA

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

/**
 * We currently get a -Wshadow warning from the third-party header cusparse.h, 
 * and it seems difficult to fix by using -isystem, therefore, we disable the warnings
 * in the inclusions.
 */
#if defined(__clang__)            // 1. Clang ***
#  pragma clang diagnostic push
#  pragma clang diagnostic ignored "-Wshadow"

#elif defined(__GNUC__)           // 2. GCC  ***
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wshadow"

#elif defined(_MSC_VER)           // 3. MSVC ***
  // - C4456: declaration hides previous local declaration
  // - C4457: declaration hides function parameter
  // - C4458: declaration hides class/struct member
  // - C4459: declaration hides global declaration
#  pragma warning(push)
#  pragma warning(disable: 4456 4457 4458 4459)
#endif

#include <cusparse.h>

#if defined(__clang__)
#  pragma clang diagnostic pop
#elif defined(__GNUC__)
#  pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#  pragma warning(pop)
#endif