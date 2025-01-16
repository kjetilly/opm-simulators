/*
  Copyright 2023 Equinor ASA

  This file is part of the Open Porous Media Project (OPM).

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

#ifndef OPM_UTIL_COMPRESS_PARTITION_HPP_INCLUDED
#define OPM_UTIL_COMPRESS_PARTITION_HPP_INCLUDED

#include <utility>
#include <vector>

namespace Opm { namespace util {
    std::pair<std::vector<long long>, long long>
    compressAndCountPartitionIDs(std::vector<long long>&& parts0);

    std::vector<long long> compressPartitionIDs(std::vector<long long>&& parts0);
    void compressPartitionIDs(std::vector<long long>& parts0);
}} // namespace Opm::util

#endif // OPM_UTIL_COMPRESS_PARTITION_HPP_INCLUDED
