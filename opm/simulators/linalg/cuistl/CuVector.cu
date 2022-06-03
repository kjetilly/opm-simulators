#include <cublas_v2.h>
#include <cuda.h>
#include <opm/simulators/linalg/cuistl/CuVector.hpp>
#include <opm/simulators/linalg/cuistl/cublas_safe_call.hpp>
#include <opm/simulators/linalg/cuistl/cuda_safe_call.hpp>


namespace Opm::cuistl
{

template <class T>
CuVector<T>::CuVector(const int numberOfElements)
    : numberOfElements(numberOfElements)
    , cuBlasHandle(CuBlasHandle::getInstance())
{
    OPM_CUDA_SAFE_CALL(cudaMalloc(&dataOnDevice, sizeof(T) * numberOfElements));
}

template <class T>
CuVector<T>::CuVector(const T* dataOnHost, const int numberOfElements)
    : CuVector(numberOfElements)
{
    OPM_CUDA_SAFE_CALL(cudaMemcpy(dataOnDevice, dataOnHost, numberOfElements * sizeof(T), cudaMemcpyHostToDevice));
}

template <class T>
CuVector<T>::~CuVector()
{
    OPM_CUDA_SAFE_CALL(cudaFree(dataOnDevice));
}

template <typename T>
const T*
CuVector<T>::data() const
{
    return dataOnDevice;
}

template <typename T>
T*
CuVector<T>::data()
{
    return dataOnDevice;
}

template <class T>
CuVector<T>&
CuVector<T>::operator*=(const T& scalar)
{
    // maybe this can be done more elegantly?
    if constexpr (std::is_same<T, double>::value) {
        OPM_CUBLAS_SAFE_CALL(cublasDscal(cuBlasHandle.get(), numberOfElements, &scalar, data(), 1));
    } else if constexpr (std::is_same<T, float>::value) {
        OPM_CUBLAS_SAFE_CALL(cublasSscal(cuBlasHandle.get(), numberOfElements, &scalar, data(), 1));
    } else if constexpr (std::is_same<T, int>::value) {
        OPM_THROW(std::runtime_error, "Scalar multiplication for integer vectors is not implemented yet.");
    } else {
        // TODO: Make this a static assert.
        OPM_THROW(std::runtime_error, "This should not land here...");
    }

    return *this;
}

template <class T>
void
CuVector<T>::copyFromHost(const T* dataPointer, int numberOfElements)
{
    OPM_CUDA_SAFE_CALL(cudaMemcpy(data(), dataPointer, numberOfElements * sizeof(T), cudaMemcpyHostToDevice));
}

template <class T>
void
CuVector<T>::copyToHost(T* dataPointer, int numberOfElements) const
{
    OPM_CUDA_SAFE_CALL(cudaMemcpy(dataPointer, data(), numberOfElements * sizeof(T), cudaMemcpyDeviceToHost));
}
template class CuVector<double>;
template class CuVector<float>;
template class CuVector<int>;

} // namespace Opm::cuistl
