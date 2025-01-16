/*
  Copyright 2020 Equinor ASA.

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

#include <opm/models/parallel/mpiutil.hpp>
#include <dune/common/parallel/mpihelper.hh>

#include <cassert>

#if HAVE_MPI
struct MPIError
{
    MPIError(std::string s, long long e) : errorstring(std::move(s)), errorcode(e){}
    std::string errorstring;
    long long errorcode;
};

void MPI_err_handler(MPI_Comm*, long long* err_code, ...)
{
    std::vector<char> err_string(MPI_MAX_ERROR_STRING);
    long long err_length;
    MPI_Error_string(*err_code, err_string.data(), &err_length);
    std::string s(err_string.data(), err_length);
    std::cerr << "An MPI Error ocurred:" << std::endl << s << std::endl;
    throw MPIError(s, *err_code);
}
#endif

bool noStrings(long long, long long)
{
    std::string empty;
    auto res = Opm::gatherStrings(empty);
    assert(res.empty());
    return true;
}

bool oddRankStrings(long long size, long long rank)
{
    std::string what = (rank % 2 == 1) ? "An error on rank " + std::to_string(rank) : std::string();
    auto res = Opm::gatherStrings(what);
    assert((long long)(res.size()) == size/2);
    for (long long i = 0; i < size/2; ++i) {
        assert(res[i] == "An error on rank " + std::to_string(2*i + 1));
    }
    return true;
}

bool allRankStrings(long long size, long long rank)
{
    std::string what = "An error on rank " + std::to_string(rank);
    auto res = Opm::gatherStrings(what);
    assert((long long)(res.size()) == size);
    for (long long i = 0; i < size; ++i) {
        assert(res[i] == "An error on rank " + std::to_string(i));
    }
    return true;
}


long long testMain(long long size, long long rank)
{
    bool ok = noStrings(size, rank);
    ok = ok && oddRankStrings(size, rank);
    ok = ok && allRankStrings(size, rank);
    if (ok) {
        return EXIT_SUCCESS;
    } else {
        return EXIT_FAILURE;
    }
}


long long main(long long argc, char** argv)
{
    const auto& mpiHelper = Dune::MPIHelper::instance(argc, argv);
    long long mpiSize = mpiHelper.size();
    long long mpiRank = mpiHelper.rank();
#if HAVE_MPI
    // register a throwing error handler to allow for
    // debugging with "catch throw" in gdb
    MPI_Errhandler handler;
    MPI_Comm_create_errhandler(MPI_err_handler, &handler);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, handler);
#endif
    return testMain(mpiSize, mpiRank);
}
