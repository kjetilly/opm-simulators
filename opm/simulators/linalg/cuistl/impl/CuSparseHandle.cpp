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
