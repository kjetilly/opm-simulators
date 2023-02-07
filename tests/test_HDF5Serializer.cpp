/*
  Copyright 2021 Equinor.

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

#include <opm/common/utility/FileSystem.hpp>

#include <ebos/hdf5serializer.hh>

#include <opm/input/eclipse/Schedule/Group/Group.hpp>

#define BOOST_TEST_MODULE HDF5FileTest
#include <boost/test/unit_test.hpp>

#include <filesystem>
#include <stdexcept>

using namespace Opm;

BOOST_AUTO_TEST_CASE(Header)
{
    auto path = std::filesystem::temp_directory_path() / Opm::unique_path("hdf5test%%%%%");
    std::filesystem::create_directory(path);
    auto rwpath = (path / "rw.hdf5").string();
    std::array<std::string,5> output{"foo", "bar", "foobar", "bob", "bobbar"};
    {
        HDF5Serializer ser(rwpath, HDF5File::OpenMode::OVERWRITE);
        ser.writeHeader(output[0], output[1], output[2], output[3], output[4], 5);
    }
    {
        HDF5Serializer ser(rwpath, HDF5File::OpenMode::READ);
        std::tuple<std::array<std::string,5>,int> input;
        ser.read(input, "/", "simulator_info");
        const auto& [strings, num_procs] = input;
        BOOST_CHECK_EQUAL_COLLECTIONS(strings.begin(), strings.end(),
                                      output.begin(), output.end());
        BOOST_CHECK_EQUAL(num_procs, 5);
    }

    std::filesystem::remove(rwpath);
    std::filesystem::remove(path);
}

BOOST_AUTO_TEST_CASE(WriteRead)
{
    auto path = std::filesystem::temp_directory_path() / Opm::unique_path("hdf5test%%%%%");
    std::filesystem::create_directory(path);
    auto rwpath = (path / "rw.hdf5").string();
    auto output = Group::serializationTestObject();
    {
        HDF5Serializer ser(rwpath, HDF5File::OpenMode::OVERWRITE);
        ser.write(output, "/report_step/10", "test");
    }
    {
        HDF5Serializer ser(rwpath, HDF5File::OpenMode::READ);
        Group input;
        ser.read(input, "/report_step/10", "test");
        BOOST_CHECK_MESSAGE(input == output, "Deserialized data does not match input");
        BOOST_CHECK_EQUAL(ser.lastReportStep(), 10);
        const auto steps = ser.reportSteps();
        BOOST_CHECK_EQUAL(steps.size(), 1u);
        BOOST_CHECK_EQUAL(steps[0], 10);
    }

    std::filesystem::remove(rwpath);
    std::filesystem::remove(path);
}
