/*
  Copyright 2022-2023 SINTEF AS

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
 * Contains wrappers to make the CuBLAS library behave as a modern C++ library with function overlading.
 *
 * In simple terms, this allows one to call say cublasScal on both double and single precisision,
 * instead of calling cublasDscal and cublasSscal respectively.
 */

#ifndef OPM_CUBLASWRAPPER_HEADER_INCLUDED
#define OPM_CUBLASWRAPPER_HEADER_INCLUDED
#include <cublas_v2.h>
#include <opm/common/ErrorMacros.hpp>

namespace Opm::gpuistl::detail
{

inline cublasStatus_t
cublasScal(cublasHandle_t handle,
           long long n,
           const double* alpha, /* host or device pointer */
           double* x,
           long long incx)
{
    return cublasDscal(handle,
                       n,
                       alpha, /* host or device pointer */
                       x,
                       incx);
}

inline cublasStatus_t
cublasScal(cublasHandle_t handle,
           long long n,
           const float* alpha, /* host or device pointer */
           float* x,
           long long incx)
{
    return cublasSscal(handle,
                       n,
                       alpha, /* host or device pointer */
                       x,
                       incx);
}

inline cublasStatus_t
cublasScal([[maybe_unused]] cublasHandle_t handle,
           [[maybe_unused]] long long n,
           [[maybe_unused]] const long long* alpha, /* host or device pointer */
           [[maybe_unused]] long long* x,
           [[maybe_unused]] long long incx)
{
    OPM_THROW(std::runtime_error, "cublasScal multiplication for integer vectors is not implemented yet.");
}
inline cublasStatus_t
cublasAxpy(cublasHandle_t handle,
           long long n,
           const double* alpha, /* host or device pointer */
           const double* x,
           long long incx,
           double* y,
           long long incy)
{
    return cublasDaxpy(handle,
                       n,
                       alpha, /* host or device pointer */
                       x,
                       incx,
                       y,
                       incy);
}

inline cublasStatus_t
cublasAxpy(cublasHandle_t handle,
           long long n,
           const float* alpha, /* host or device pointer */
           const float* x,
           long long incx,
           float* y,
           long long incy)
{
    return cublasSaxpy(handle,
                       n,
                       alpha, /* host or device pointer */
                       x,
                       incx,
                       y,
                       incy);
}

inline cublasStatus_t
cublasAxpy([[maybe_unused]] cublasHandle_t handle,
           [[maybe_unused]] long long n,
           [[maybe_unused]] const long long* alpha, /* host or device pointer */
           [[maybe_unused]] const long long* x,
           [[maybe_unused]] long long incx,
           [[maybe_unused]] long long* y,
           [[maybe_unused]] long long incy)
{
    OPM_THROW(std::runtime_error, "axpy multiplication for integer vectors is not implemented yet.");
}

inline cublasStatus_t
cublasDot(cublasHandle_t handle, long long n, const double* x, long long incx, const double* y, long long incy, double* result)
{
    return cublasDdot(handle, n, x, incx, y, incy, result);
}

inline cublasStatus_t
cublasDot(cublasHandle_t handle, long long n, const float* x, long long incx, const float* y, long long incy, float* result)
{
    return cublasSdot(handle, n, x, incx, y, incy, result);
}

inline cublasStatus_t
cublasDot([[maybe_unused]] cublasHandle_t handle,
          [[maybe_unused]] long long n,
          [[maybe_unused]] const long long* x,
          [[maybe_unused]] long long incx,
          [[maybe_unused]] const long long* y,
          [[maybe_unused]] long long incy,
          [[maybe_unused]] long long* result)
{
    OPM_THROW(std::runtime_error, "inner product for integer vectors is not implemented yet.");
}

inline cublasStatus_t
cublasNrm2(cublasHandle_t handle, long long n, const double* x, long long incx, double* result)
{
    return cublasDnrm2(handle, n, x, incx, result);
}


inline cublasStatus_t
cublasNrm2(cublasHandle_t handle, long long n, const float* x, long long incx, float* result)
{
    return cublasSnrm2(handle, n, x, incx, result);
}

inline cublasStatus_t
cublasNrm2([[maybe_unused]] cublasHandle_t handle,
           [[maybe_unused]] long long n,
           [[maybe_unused]] const long long* x,
           [[maybe_unused]] long long incx,
           [[maybe_unused]] long long* result)
{
    OPM_THROW(std::runtime_error, "norm2 for integer vectors is not implemented yet.");
}

} // namespace Opm::gpuistl::detail
#endif
