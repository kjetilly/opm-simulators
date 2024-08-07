/*
  Copyright 2024 SINTEF Digital, Mathematics and Cybernetics.

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

#define BOOST_TEST_MODULE TestEclWriterLargeGrid
#include <boost/test/unit_test.hpp>

#include <fmt/format.h>
#include <opm/io/eclipse/EclFile.hpp>
#include <opm/io/eclipse/EclOutput.hpp>
#include <opm/simulators/flow/EclGenericWriter.hpp>
#include <opm/simulators/flow/Main.hpp>
#include <string>
#include <vector>

BOOST_AUTO_TEST_CASE(TestWrite)
{
    const auto makeFieldname = [](size_t i) { return fmt::format("S{:03d}", i); };

    const size_t size = 32; // 466 * 466 * 466;
    const size_t numberOfOutputs = 128;

    const auto makeFieldData = [](size_t i, size_t j) { return i * size + j; };
    const auto filename = std::string("testfile.EGRID");
    {
        Opm::EclIO::EclOutput output(filename, false);


        for (size_t i = 0; i < numberOfOutputs; ++i) {
            std::cout << fmt::format("{:04d}\r", i) << std::flush;
            std::flush(std::cout);
            std::vector<double> dummyOutput(size, 42.0);
            for (size_t j = 0; j < size; ++j) {
                dummyOutput[j] = makeFieldData(i, j);
            }
            output.write(makeFieldname(i), dummyOutput);
        }
        std::cout << std::endl;
    }

    Opm::EclIO::EclFile inputFile(filename);

    for (size_t i = 0; i < numberOfOutputs; ++i) {
        std::cout << "#";
        std::flush(std::cout);
        inputFile.loadData(makeFieldname(i));
        const auto& outputField = inputFile.get<double>(fmt::format("S{:03d}", i));

        for (size_t j = 0; j < size; ++j) {
            BOOST_CHECK_EQUAL(makeFieldData(i, j), outputField[j]);
        }
    }
    std::cout << std::endl;
}
