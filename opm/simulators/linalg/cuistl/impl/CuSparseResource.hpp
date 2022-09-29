#ifndef CUSPARSERESOURCE_HPP
#define CUSPARSERESOURCE_HPP
#include <cusparse.h>
#include <functional>
#include <memory>
#include <type_traits>

namespace Opm::cuistl::impl
{

//! @brief wraps a CuSparseResource in proper RAII.
template <class T>
class CuSparseResource
{
public:
    using CreatorType = typename std::function<cusparseStatus_t(T*)>;
    using DeleterType = typename std::function<cusparseStatus_t(T)>;
    CuSparseResource(CreatorType creator, DeleterType deleter);
    CuSparseResource();
    ~CuSparseResource();

    // This should not be copyable.
    CuSparseResource(const CuSparseResource&) = delete;
    CuSparseResource& operator=(const CuSparseResource&) = delete;

    T get()
    {
        return resource;
    }

private:
    T resource;

    DeleterType deleter;
};

} // namespace Opm::cuistl::impl
#include <opm/simulators/linalg/cuistl/impl/CuSparseResource_impl.hpp>
#endif // CUSPARSERESOURCE_HPP
