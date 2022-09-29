#ifndef CUBLAS_SAFE_CALL_HPP
#define CUBLAS_SAFE_CALL_HPP
#include <cublas_v2.h>
#include <exception>
#include <opm/common/ErrorMacros.hpp>
#define CHECK_CUBLAS_ERROR_TYPE(code, x)                                                                               \
    if (code == x) {                                                                                                   \
        return #x;                                                                                                     \
    }
namespace
{
inline std::string
getCublasErrorMessage(int code)
{
    CHECK_CUBLAS_ERROR_TYPE(code, CUBLAS_STATUS_SUCCESS);
    CHECK_CUBLAS_ERROR_TYPE(code, CUBLAS_STATUS_NOT_INITIALIZED);
    CHECK_CUBLAS_ERROR_TYPE(code, CUBLAS_STATUS_ALLOC_FAILED);
    CHECK_CUBLAS_ERROR_TYPE(code, CUBLAS_STATUS_INVALID_VALUE);
    CHECK_CUBLAS_ERROR_TYPE(code, CUBLAS_STATUS_ARCH_MISMATCH);
    CHECK_CUBLAS_ERROR_TYPE(code, CUBLAS_STATUS_MAPPING_ERROR);
    CHECK_CUBLAS_ERROR_TYPE(code, CUBLAS_STATUS_EXECUTION_FAILED);
    CHECK_CUBLAS_ERROR_TYPE(code, CUBLAS_STATUS_INTERNAL_ERROR);
    CHECK_CUBLAS_ERROR_TYPE(code, CUBLAS_STATUS_NOT_SUPPORTED);
    CHECK_CUBLAS_ERROR_TYPE(code, CUBLAS_STATUS_LICENSE_ERROR);

    return "UNKNOWN CUBLAS ERROR " + std::to_string(code);
}
} // namespace
#undef CHECK_CUSPRASE_ERROR_TYPE
// This macro does nothing as of yet, but will in the future
#define OPM_CUBLAS_SAFE_CALL(expression)                                                                               \
    do {                                                                                                               \
        cublasStatus_t error = expression;                                                                             \
        if (error != CUBLAS_STATUS_SUCCESS) {                                                                          \
            OPM_THROW(std::runtime_error,                                                                              \
                      "cuBLAS expression did not execute correctly. Expression was: \n\n"                              \
                          << "    " << #expression << "\n\n"                                                           \
                          << "in function " << __func__ << ", in " << __FILE__ << " at line " << __LINE__ << "\n"      \
                          << "CuBLAS error code was: " << getCublasErrorMessage(error) << "\n");                       \
        }                                                                                                              \
    } while (false)
#endif // CUBLAS_SAFE_CALL_HPP
