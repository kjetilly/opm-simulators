/*
  Copyright TODO ADD YEAR AND NAME OF AUTHOR

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
#include <config.h>

#include <opm/simulators/linalg/LinearSolverRuntimeParameters.hpp>

#include <algorithm>
#include <cctype>
#include <ranges>
#include <string_view>

namespace Opm
{

std::vector<std::string>
splitLinearSolverNames(const std::string& value)
{
    auto trim = [](std::string_view part) {
        while (!part.empty() && std::isspace(static_cast<unsigned char>(part.front())) != 0) {
            part.remove_prefix(1);
        }
        while (!part.empty() && std::isspace(static_cast<unsigned char>(part.back())) != 0) {
            part.remove_suffix(1);
        }
        return part;
    };

    std::vector<std::string> names;
    for (const auto part : std::views::split(std::string_view {value}, ',')) {
        const auto name = trim(std::string_view {part.begin(), part.end()});
        if (!name.empty()) {
            names.emplace_back(name);
        }
    }
    return names;
}

} // namespace Opm
