/*
  Copyright SINTEF AS 2022

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
#include <opm/simulators/linalg/cuistl/impl/CuSparseHandle.hpp>
#include <opm/simulators/linalg/cuistl/impl/cusparse_safe_call.hpp>
namespace Opm::cuistl::impl
{


CuSparseHandle::CuSparseHandle()
{
    OPM_CUSPARSE_SAFE_CALL(cusparseCreate(&handle));
    OPM_CUSPARSE_SAFE_CALL(cusparseSetStream(handle, 0));
}

CuSparseHandle::~CuSparseHandle()
{
    OPM_CUSPARSE_SAFE_CALL(cusparseDestroy(handle));
}

cusparseHandle_t
CuSparseHandle::get()
{
    return handle;
}

CuSparseHandle&
CuSparseHandle::getInstance()
{
    static CuSparseHandle instance;
    return instance;
}

} // namespace Opm::cuistl::impl
