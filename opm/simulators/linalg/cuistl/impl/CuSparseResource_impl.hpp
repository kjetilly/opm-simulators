#include <exception>
#include <opm/common/ErrorMacros.hpp>
#include <opm/simulators/linalg/cuistl/impl/cusparse_safe_call.hpp>

namespace Opm::cuistl::impl
{

namespace
{
    template <class T>
    struct CuSparseDeleteAndCreate {
    };

    template <>
    struct CuSparseDeleteAndCreate<bsrilu02Info_t> {
        using DeleterType = typename CuSparseResource<bsrilu02Info_t>::DeleterType;
        using CreatorType = typename CuSparseResource<bsrilu02Info_t>::CreatorType;

        static DeleterType getDeleter()
        {
            return cusparseDestroyBsrilu02Info;
        }

        static CreatorType getCreator()
        {
            return cusparseCreateBsrilu02Info;
        }
    };

    template <>
    struct CuSparseDeleteAndCreate<bsrsv2Info_t> {
        using DeleterType = typename CuSparseResource<bsrsv2Info_t>::DeleterType;
        using CreatorType = typename CuSparseResource<bsrsv2Info_t>::CreatorType;

        static DeleterType getDeleter()
        {
            return cusparseDestroyBsrsv2Info;
        }

        static CreatorType getCreator()
        {
            return cusparseCreateBsrsv2Info;
        }
    };

    template <>
    struct CuSparseDeleteAndCreate<cusparseMatDescr_t> {
        using DeleterType = typename CuSparseResource<cusparseMatDescr_t>::DeleterType;
        using CreatorType = typename CuSparseResource<cusparseMatDescr_t>::CreatorType;

        static DeleterType getDeleter()
        {
            return cusparseDestroyMatDescr;
        }

        static CreatorType getCreator()
        {
            return cusparseCreateMatDescr;
        }
    };

} // namespace
template <class T>
CuSparseResource<T>::CuSparseResource(CreatorType creator, DeleterType deleter)
    : deleter(deleter)
{
    // TODO: This should probably not use this macro since it will disguise the
    // proper name of the function being called.
    OPM_CUSPARSE_SAFE_CALL(creator(&resource));
}

template <class T>
CuSparseResource<T>::CuSparseResource()
    : CuSparseResource<T>(CuSparseDeleteAndCreate<T>::getCreator(), CuSparseDeleteAndCreate<T>::getDeleter())
{
}

template <class T>
CuSparseResource<T>::~CuSparseResource()
{
    // TODO: This should probably not use this macro since it will disguise the
    // proper name of the function being called.
    OPM_CUSPARSE_SAFE_CALL(deleter(resource));
}
} // namespace Opm::cuistl
