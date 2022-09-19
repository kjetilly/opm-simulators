#include <cublas_v2.h>
#include <cuda.h>
#include <opm/simulators/linalg/cuistl/CuVector.hpp>
#include <opm/simulators/linalg/cuistl/cublas_safe_call.hpp>
#include <opm/simulators/linalg/cuistl/cuda_safe_call.hpp>
#include <opm/simulators/linalg/cuistl/impl/cublas_wrapper.hpp>

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
CuVector<T>& CuVector<T>::operator=(const CuVector<T>& other)
{
    if (other.numberOfElements != numberOfElements) {
        OPM_THROW(std::invalid_argument, "Can only copy from vector of same size.");
    }
    OPM_CUDA_SAFE_CALL(cudaMemcpy(dataOnDevice, other.dataOnDevice, numberOfElements * sizeof(T), cudaMemcpyDeviceToDevice));
    return *this;
}

template <class T>
CuVector<T>::CuVector(const CuVector<T>& other)
    : CuVector(other.numberOfElements)
{
    OPM_CUDA_SAFE_CALL(cudaMemcpy(dataOnDevice, other.dataOnDevice, numberOfElements * sizeof(T), cudaMemcpyDeviceToDevice));
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
typename CuVector<T>::size_type
CuVector<T>::dim() const
{
    return numberOfElements;
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
CuVector<T>&
CuVector<T>::axpy(T alpha, const CuVector<T>& y) {
    OPM_CUBLAS_SAFE_CALL(impl::cublasAxpy(
        cuBlasHandle.get(), 
        numberOfElements,
        &alpha,
        y.data(),
        1,
        data(),
        1
    ));
    return *this;
}

template<class T>
T CuVector<T>::dot(const CuVector<T>& other) const {
    T result = T(0);
    OPM_CUBLAS_SAFE_CALL(impl::cublasDot(
        cuBlasHandle.get(),
        numberOfElements,
        data(),
        1,
        other.data(),
        1,
        &result)
    );
    return result;
}
template<class T>
T CuVector<T>::two_norm() const {
    T result = T(0);
    OPM_CUBLAS_SAFE_CALL(impl::cublasNrm2(
        cuBlasHandle.get(),
        numberOfElements,
        data(),
        1,
        &result)
    );
    return result;
}

template <class T>
CuVector<T>&
CuVector<T>::operator+=(const CuVector<T>& other) {
    // TODO: [perf] Make a specialized version of this
    return axpy(1.0, other);
}


template <class T>
void
CuVector<T>::copyFromHost(const T* dataPointer, int numberOfElements)
{
    if (numberOfElements > dim()) {
        OPM_THROW(std::runtime_error, "Requesting to copy too many elements. Vector has "
            << dim() << " elements, while " << numberOfElements << " was requested.");
    }
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
