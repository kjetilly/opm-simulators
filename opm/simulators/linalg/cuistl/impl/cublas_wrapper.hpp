#ifndef OPM_CUBLASWRAPPER_HEADER_INCLUDED
#define OPM_CUBLASWRAPPER_HEADER_INCLUDED
#include <cublas_v2.h>
#include <opm/common/ErrorMacros.hpp>

namespace Opm::cuistl::impl
{
cublasStatus_t
cublasAxpy(cublasHandle_t handle,
           int n,
           const double* alpha, /* host or device pointer */
           const double* x,
           int incx,
           double* y,
           int incy)
{
    return cublasDaxpy(handle,
                       n,
                       alpha, /* host or device pointer */
                       x,
                       incx,
                       y,
                       incy);
}

cublasStatus_t
cublasAxpy(cublasHandle_t handle,
           int n,
           const float* alpha, /* host or device pointer */
           const float* x,
           int incx,
           float* y,
           int incy)
{
    return cublasSaxpy(handle,
                       n,
                       alpha, /* host or device pointer */
                       x,
                       incx,
                       y,
                       incy);
}
cublasStatus_t
cublasAxpy(cublasHandle_t handle,
           int n,
           const int* alpha, /* host or device pointer */
           const int* x,
           int incx,
           int* y,
           int incy)
{
    OPM_THROW(std::runtime_error, "axpy multiplication for integer vectors is not implemented yet.");
}

} // namespace Opm::cuistl::impl
#endif