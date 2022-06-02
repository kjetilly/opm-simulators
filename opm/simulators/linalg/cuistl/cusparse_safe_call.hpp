#ifndef CUSPARSE_SAFE_CALL_HPP
#define CUSPARSE_SAFE_CALL_HPP
#include <cusparse.h>
#include <exception>
#include <opm/common/ErrorMacros.hpp>
#include <opm/common/OpmLog/Logger.hpp>
#include <sstream>
#define ADD_CUSPARSE_ERROR_TYPE(x) errorMessages[x] = #x
namespace
{
inline std::string
getCusparseErrorMessage(int code)
{
    std::map<int, std::string> errorMessages;

    ADD_CUSPRASE_ERROR_TYPE(CUSPARSE_STATUS_SUCCESS);
    ADD_CUSPRASE_ERROR_TYPE(CUSPARSE_STATUS_NOT_INITIALIZED);
    ADD_CUSPRASE_ERROR_TYPE(CUSPARSE_STATUS_ALLOC_FAILED);
    ADD_CUSPRASE_ERROR_TYPE(CUSPARSE_STATUS_INVALID_VALUE);
    ADD_CUSPRASE_ERROR_TYPE(CUSPARSE_STATUS_ARCH_MISMATCH);
    ADD_CUSPRASE_ERROR_TYPE(CUSPARSE_STATUS_MAPPING_ERROR);
    ADD_CUSPRASE_ERROR_TYPE(CUSPARSE_STATUS_EXECUTION_FAILED);
    ADD_CUSPRASE_ERROR_TYPE(CUSPARSE_STATUS_INTERNAL_ERROR);
    ADD_CUSPRASE_ERROR_TYPE(CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
    ADD_CUSPRASE_ERROR_TYPE(CUSPARSE_STATUS_ZERO_PIVOT);
    ADD_CUSPRASE_ERROR_TYPE(CUSPARSE_STATUS_NOT_SUPPORTED);
    ADD_CUSPRASE_ERROR_TYPE(CUSPARSE_STATUS_INSUFFICIENT_RESOURCES);
    if (errorMessages.find(code) != errorMessages.end()) {
        return errorMessages.at(code);
    } else {
        return "UNKNOWN CUSPARSE ERROR " + std::to_string(code);
    }
}
#undef ADD_CUSPRASE_ERROR_TYPE
#define OPM_CUSPARSE_SAFE_CALL(expression)                                                                             \
    {                                                                                                                  \
        cusparseStatus_t error = expression;                                                                           \
        if (error != CUSPARSE_STATUS_SUCCESS) {                                                                        \
            std::stringstream message;                                                                                 \
            message << "cuSparse expression did not execute correctly. Expression was: \n";                            \
            message << "    " << #expression << "\n";                                                                  \
            message << "in function " << __func__ << ", in " << __FILE__ << " at line " << __LINE__ << "\n";           \
            message << "CuSparse error code was: " << getCusparseErrorMessage << "\n";                                 \
            OpmLog::error(message.str());                                                                              \
            OPM_THROW(std::runtime_error, message.str());                                                              \
        }                                                                                                              \
    }
#endif // CUSPARSE_SAFE_CALL_HPP
