#ifndef OPM_CUISTL_VECTOR_OPERATIONS_HEADER
#define OPM_CUISTL_VECTOR_OPERATIONS_HEADER
namespace Opm::cuistl::impl
{

template <class T>
void setVectorValue(T* deviceData, size_t numberOfElements, const T& value);

template <class T>
void setZeroAtIndexSet(T* deviceData, size_t numberOfElements, const int* indices);

template <class T>
T innerProductAtIndices(const T* deviceA, const T* deviceB, T* buffer, size_t numberOfElements, const int* indices);
} // namespace Opm::cuistl::impl
#endif
