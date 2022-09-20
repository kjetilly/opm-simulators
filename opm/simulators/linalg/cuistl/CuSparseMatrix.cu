#include <cuda.h>
#include <opm/simulators/linalg/cuistl/CuSparseMatrix.hpp>
#include <opm/simulators/linalg/cuistl/cusparse_safe_call.hpp>
#include <opm/simulators/linalg/cuistl/impl/cusparse_wrapper.hpp>
#include <opm/simulators/linalg/cuistl/cusparse_constants.hpp>
namespace Opm::cuistl
{

template <class T>
CuSparseMatrix<T>::CuSparseMatrix(const T* nonZeroElements,
                                  const int* rowIndices,
                                  const int* columnIndices,
                                  int numberOfNonzeroBlocks,
                                  int blockSize,
                                  int numberOfRows)
    : nonZeroElements(nonZeroElements, numberOfNonzeroBlocks * blockSize * blockSize)
    , rowIndices(rowIndices, numberOfRows + 1)
    , columnIndices(columnIndices, numberOfNonzeroBlocks)
    , numberOfNonzeroBlocks(numberOfNonzeroBlocks)
    , numberOfRows(numberOfRows)
    , matrixDescription(createMatrixDescription())
    , _blockSize(blockSize)
    , cusparseHandle(CuSparseHandle::getInstance())
{
}

template <class T>
CuSparseMatrix<T>::~CuSparseMatrix()
{
    // empty
}

template <typename T>
void
CuSparseMatrix<T>::setUpperTriangular()
{
    OPM_CUSPARSE_SAFE_CALL(cusparseSetMatFillMode(matrixDescription->get(), CUSPARSE_FILL_MODE_UPPER));
}

template <typename T>
void
CuSparseMatrix<T>::setLowerTriangular()
{
    OPM_CUSPARSE_SAFE_CALL(cusparseSetMatFillMode(matrixDescription->get(), CUSPARSE_FILL_MODE_LOWER));
}

template <typename T>
void
CuSparseMatrix<T>::setUnitDiagonal()
{
    OPM_CUSPARSE_SAFE_CALL(cusparseSetMatDiagType(matrixDescription->get(), CUSPARSE_DIAG_TYPE_UNIT));
}

template <typename T>
void
CuSparseMatrix<T>::setNonUnitDiagonal()
{
    OPM_CUSPARSE_SAFE_CALL(cusparseSetMatDiagType(matrixDescription->get(), CUSPARSE_DIAG_TYPE_NON_UNIT));
}

template<typename T>
void CuSparseMatrix<T>::mv(const CuVector<T>& x, CuVector<T>& y) const {
    usmv(1.0, x, y);
}

template<typename T>
void CuSparseMatrix<T>::usmv (T alpha, const CuVector<T>& x, CuVector<T>& y) const {
    if (blockSize() < 2) {
        OPM_THROW(std::invalid_argument, "CuSparseMatrix<T>::applyscaleadd and CuSparseMatrix<T>::apply are only implemented for block sizes greater than 1.");
    }
    const auto numberOfRows = N();
    const auto numberOfNonzeroBlocks = nonzeroes();
    const auto nonzeroValues = getNonZeroValues().data();

    auto rowIndices = getRowIndices().data();
    auto columnIndices = getColumnIndices().data();
            
    T beta = 1.0;
    OPM_CUSPARSE_SAFE_CALL(impl::cusparseBsrmv(cusparseHandle.get(),
               CUSPARSE_MATRIX_ORDER,
               CUSPARSE_OPERATION_NON_TRANSPOSE,
               numberOfRows,
               numberOfRows,
               numberOfNonzeroBlocks,
               &alpha,
               matrixDescription->get(),
               nonzeroValues,
               rowIndices,
               columnIndices,
               blockSize(),
               x.data(),
               &beta,
               y.data()));
}

template<typename T>
Dune::SolverCategory::Category CuSparseMatrix<T>::category() const {
    return Dune::SolverCategory::sequential;
}



template class CuSparseMatrix<float>;
template class CuSparseMatrix<double>;
} // namespace Opm::cuistl
