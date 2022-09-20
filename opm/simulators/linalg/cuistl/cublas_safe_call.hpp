#ifndef CUBLAS_SAFE_CALL_HPP
#define CUBLAS_SAFE_CALL_HPP
#include <cublas_v2.h>
#include <exception>
#include <opm/common/ErrorMacros.hpp>
#include <opm/common/OpmLog/Logger.hpp>
#include <sstream>
#define ADD_CUBLAS_ERROR_TYPE(x) errorMessages[int(x)] = #x
namespace
{
inline std::string
getCublasErrorMessage(int code)
{
    std::map<int, std::string> errorMessages;
    ADD_CUBLAS_ERROR_TYPE(CUBLAS_STATUS_SUCCESS);
    ADD_CUBLAS_ERROR_TYPE(CUBLAS_STATUS_NOT_INITIALIZED);
    ADD_CUBLAS_ERROR_TYPE(CUBLAS_STATUS_ALLOC_FAILED);
    ADD_CUBLAS_ERROR_TYPE(CUBLAS_STATUS_INVALID_VALUE);
    ADD_CUBLAS_ERROR_TYPE(CUBLAS_STATUS_ARCH_MISMATCH);
    ADD_CUBLAS_ERROR_TYPE(CUBLAS_STATUS_MAPPING_ERROR);
    ADD_CUBLAS_ERROR_TYPE(CUBLAS_STATUS_EXECUTION_FAILED);
    ADD_CUBLAS_ERROR_TYPE(CUBLAS_STATUS_INTERNAL_ERROR);
    ADD_CUBLAS_ERROR_TYPE(CUBLAS_STATUS_NOT_SUPPORTED);
    ADD_CUBLAS_ERROR_TYPE(CUBLAS_STATUS_LICENSE_ERROR);
    
    if (errorMessages.find(code) != errorMessages.end()) {
        return errorMessages.at(code);
    } else {
        return "UNKNOWN CUBLAS ERROR " + std::to_string(code);
    }
}
} // namespace
#undef ADD_CUSPRASE_ERROR_TYPE
// This macro does nothing as of yet, but will in the future
#define OPM_CUBLAS_SAFE_CALL(expression) \
{                                                                                                                  \
        cublasStatus_t error = expression;                                                                           \
        if (error != CUBLAS_STATUS_SUCCESS) {                                                                        \
            std::stringstream message;                                                                                 \
            message << "cuBLAS expression did not execute correctly. Expression was: \n\n";                            \
            message << "    " << #expression << "\n\n";                                                                  \
            message << "in function " << __func__ << ", in " << __FILE__ << " at line " << __LINE__ << "\n";           \
            message << "CuBLAS error code was: " << getCublasErrorMessage(error) << "\n";                          \
            OpmLog::error(message.str());                                                                              \
            OPM_THROW(std::runtime_error, message.str());                                                              \
        }                                                                                                              \
}
#endif // CUBLAS_SAFE_CALL_HPP
