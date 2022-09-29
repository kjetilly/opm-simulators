#include <cublas_v2.h>
#include <opm/simulators/linalg/cuistl/impl/CuBlasHandle.hpp>
#include <opm/simulators/linalg/cuistl/impl/cublas_safe_call.hpp>
namespace Opm::cuistl::impl
{


CuBlasHandle::CuBlasHandle()
{
    OPM_CUBLAS_SAFE_CALL(cublasCreate(&handle));
}

CuBlasHandle::~CuBlasHandle()
{
    OPM_CUBLAS_SAFE_CALL(cublasDestroy(handle));
}

cublasHandle_t
CuBlasHandle::get()
{
    return handle;
}

CuBlasHandle&
CuBlasHandle::getInstance()
{
    static CuBlasHandle instance;
    return instance;
}

} // namespace Opm::cuistl::impl
