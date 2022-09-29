#ifndef CUDA_SAFE_CALL_HPP
#define CUDA_SAFE_CALL_HPP

#include <opm/common/ErrorMacros.hpp>

#define OPM_CUDA_SAFE_CALL(expression)                                                                                 \
    do {                                                                                                               \
        cudaError_t error = expression;                                                                                \
        if (error != cudaSuccess) {                                                                                    \
            OPM_THROW(std::runtime_error,                                                                              \
                      "CUDA expression did not execute correctly. Expression was: \n"                                  \
                          << "    " << #expression << "\n"                                                             \
                          << "CUDA error was " << cudaGetErrorString(error) << "\n"                                    \
                          << "in function " << __func__ << ", in " << __FILE__ << " at line " << __LINE__ << "\n");    \
        }                                                                                                              \
    } while (false)
#endif
