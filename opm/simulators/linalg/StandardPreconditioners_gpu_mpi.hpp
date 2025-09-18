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

#ifndef OPM_STANDARDPRECONDITIONERS_GPU_MPI_HEADER
#define OPM_STANDARDPRECONDITIONERS_GPU_MPI_HEADER


namespace Opm
{


template <class Operator, class Comm>
struct StandardPreconditioners<Operator, Comm, typename std::enable_if_t<Opm::is_gpu_operator_v<Operator>>> {
    static void add()
    {
        // No standard preconditioners for this type of operator.
    }
};


} // namespace Opm


#endif // OPM_STANDARDPRECONDITIONERS_GPU_MPI_HEADER
