#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>
#include <opm/common/ErrorMacros.hpp>
#include <opm/simulators/linalg/cuistl/CuSeqILU0.hpp>
#include <opm/simulators/linalg/cuistl/impl/cusparse_constants.hpp>
#include <opm/simulators/linalg/cuistl/impl/cusparse_safe_call.hpp>
#include <opm/simulators/linalg/cuistl/impl/cusparse_wrapper.hpp>
#include <opm/simulators/linalg/cuistl/impl/fix_zero_diagonal.hpp>
#include <opm/simulators/linalg/cuistl/time_to_file.hpp>
#include <opm/simulators/linalg/matrixblock.hh>



namespace Opm::cuistl
{

template <class M, class X, class Y, int l>
CuSeqILU0<M, X, Y, l>::CuSeqILU0(const M& A, scalar_field_type w)
    : underlyingMatrix(A)
    , w(w)
    , LU(CuSparseMatrix<field_type>::fromMatrix(impl::makeMatrixWithNonzeroDiagonal(A)))
    , temporaryStorage(LU.N() * LU.blockSize())
    , descriptionL(impl::createLowerDiagonalDescription())
    , descriptionU(impl::createUpperDiagonalDescription())
    , cuSparseHandle(impl::CuSparseHandle::getInstance())
{
    // Some sanity check
    OPM_ERROR_IF(A.N() != LU.N(),
                 "CuSparse matrix not same size as DUNE matrix. " + std::to_string(LU.N()) + " vs "
                     + std::to_string(A.N()));
    OPM_ERROR_IF(A[0][0].N() != LU.blockSize(),
                 "CuSparse matrix not same blocksize as DUNE matrix. " + std::to_string(LU.blockSize()) + " vs "
                     + std::to_string(A[0][0].N()));
    OPM_ERROR_IF(A.N() * A[0][0].N() != LU.dim(),
                 "CuSparse matrix not same dimension as DUNE matrix. " + std::to_string(LU.dim()) + " vs "
                     + std::to_string(A.N() * A[0][0].N()));
    OPM_ERROR_IF(A.nonzeroes() != LU.nonzeroes(),
                 "CuSparse matrix not same number of non zeroes as DUNE matrix. " + std::to_string(LU.nonzeroes())
                     + " vs " + std::to_string(A.nonzeroes()));

    // https://docs.nvidia.com/cuda/cusparse/index.html#csrilu02_solve
    updateILUConfiguration();
}

template <class M, class X, class Y, int l>
void
CuSeqILU0<M, X, Y, l>::pre([[maybe_unused]] X& x, [[maybe_unused]] Y& b)
{
}

template <class M, class X, class Y, int l>
void
CuSeqILU0<M, X, Y, l>::apply(X& v, const Y& d)
{

    // We need to pass the solve routine a scalar to multiply.
    // In our case this scalar is 1.0
    const scalar_field_type one = 1.0;

    const auto numberOfRows = LU.N();
    const auto numberOfNonzeroBlocks = LU.nonzeroes();
    const auto blockSize = LU.blockSize();

    auto nonZeroValues = LU.getNonZeroValues().data();
    auto rowIndices = LU.getRowIndices().data();
    auto columnIndices = LU.getColumnIndices().data();

    // Solve L temporaryStorage = d
    {
        OPM_CU_TIME_TO_FILE(cuistl, LU.nonzeroes() * LU.blockSize() * LU.blockSize());
        OPM_CUSPARSE_SAFE_CALL(impl::cusparseBsrsv2_solve(cuSparseHandle.get(),
                                                          CUSPARSE_MATRIX_ORDER,
                                                          CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                          numberOfRows,
                                                          numberOfNonzeroBlocks,
                                                          &one,
                                                          descriptionL->get(),
                                                          nonZeroValues,
                                                          rowIndices,
                                                          columnIndices,
                                                          blockSize,
                                                          infoL.get(),
                                                          d.data(),
                                                          temporaryStorage.data(),
                                                          CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                                                          buffer->data()));

        // Solve U v = temporaryStorage
        OPM_CUSPARSE_SAFE_CALL(impl::cusparseBsrsv2_solve(cuSparseHandle.get(),
                                                          CUSPARSE_MATRIX_ORDER,
                                                          CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                          numberOfRows,
                                                          numberOfNonzeroBlocks,
                                                          &one,
                                                          descriptionU->get(),
                                                          nonZeroValues,
                                                          rowIndices,
                                                          columnIndices,
                                                          blockSize,
                                                          infoU.get(),
                                                          temporaryStorage.data(),
                                                          v.data(),
                                                          CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                                                          buffer->data()));
    }

    v *= w;
}

template <class M, class X, class Y, int l>
void
CuSeqILU0<M, X, Y, l>::post([[maybe_unused]] X& x)
{
}

template <class M, class X, class Y, int l>
Dune::SolverCategory::Category
CuSeqILU0<M, X, Y, l>::category() const
{
    return Dune::SolverCategory::sequential;
}

template <class M, class X, class Y, int l>
void
CuSeqILU0<M, X, Y, l>::update()
{
    LU.updateNonzeroValues(impl::makeMatrixWithNonzeroDiagonal(underlyingMatrix));
    // updateILUConfiguration();

    createILU();
}

template <class M, class X, class Y, int l>
void
CuSeqILU0<M, X, Y, l>::analyzeMatrix()
{

    if (!buffer) {
        OPM_THROW(std::runtime_error,
                  "Buffer not initialized. Call findBufferSize() then initialize with the appropiate size.");
    }
    const auto numberOfRows = LU.N();
    const auto numberOfNonzeroBlocks = LU.nonzeroes();
    const auto blockSize = LU.blockSize();

    auto nonZeroValues = LU.getNonZeroValues().data();
    auto rowIndices = LU.getRowIndices().data();
    auto columnIndices = LU.getColumnIndices().data();
    // analysis of ilu LU decomposition
    OPM_CUSPARSE_SAFE_CALL(impl::cusparseBsrilu02_analysis(cuSparseHandle.get(),
                                                           CUSPARSE_MATRIX_ORDER,
                                                           numberOfRows,
                                                           numberOfNonzeroBlocks,
                                                           LU.getDescription().get(),
                                                           nonZeroValues,
                                                           rowIndices,
                                                           columnIndices,
                                                           blockSize,
                                                           infoM.get(),
                                                           CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                                                           buffer->data()));

    // Make sure we can decompose the matrix.
    int structuralZero;
    auto statusPivot = cusparseXbsrilu02_zeroPivot(cuSparseHandle.get(), infoM.get(), &structuralZero);
    OPM_ERROR_IF(statusPivot != CUSPARSE_STATUS_SUCCESS,
                 "Found a structucal zero at A(" + std::to_string(structuralZero) + ", "
                     + std::to_string(structuralZero) + "). Could not decompose LU approx A.\n\n" + "A has dimension: "
                     + std::to_string(LU.N()) + ",\n" + "and has " + std::to_string(LU.nonzeroes()) + " nonzeroes.");

    // analysis of ilu apply
    OPM_CUSPARSE_SAFE_CALL(impl::cusparseBsrsv2_analysis(cuSparseHandle.get(),
                                                         CUSPARSE_MATRIX_ORDER,
                                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                         numberOfRows,
                                                         numberOfNonzeroBlocks,
                                                         descriptionL->get(),
                                                         nonZeroValues,
                                                         rowIndices,
                                                         columnIndices,
                                                         blockSize,
                                                         infoL.get(),
                                                         CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                                                         buffer->data()));

    OPM_CUSPARSE_SAFE_CALL(impl::cusparseBsrsv2_analysis(cuSparseHandle.get(),
                                                         CUSPARSE_MATRIX_ORDER,
                                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                         numberOfRows,
                                                         numberOfNonzeroBlocks,
                                                         descriptionU->get(),
                                                         nonZeroValues,
                                                         rowIndices,
                                                         columnIndices,
                                                         blockSize,
                                                         infoU.get(),
                                                         CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                                                         buffer->data()));
    analyzisDone = true;
}

template <class M, class X, class Y, int l>
size_t
CuSeqILU0<M, X, Y, l>::findBufferSize()
{
    // We have three calls that need buffers:
    //   1) LU decomposition
    //   2) solve Lv = y
    //   3) solve Ux = z
    // we combine these buffers into one since it is not used across calls,
    // however, it was used in the Across project.
    const auto numberOfRows = LU.N();
    const auto numberOfNonzeroBlocks = LU.nonzeroes();
    const auto blockSize = LU.blockSize();

    auto nonZeroValues = LU.getNonZeroValues().data();
    auto rowIndices = LU.getRowIndices().data();
    auto columnIndices = LU.getColumnIndices().data();

    int bufferSizeM = 0;
    OPM_CUSPARSE_SAFE_CALL(impl::cusparseBsrilu02_bufferSize(cuSparseHandle.get(),
                                                             CUSPARSE_MATRIX_ORDER,
                                                             numberOfRows,
                                                             numberOfNonzeroBlocks,
                                                             LU.getDescription().get(),
                                                             nonZeroValues,
                                                             rowIndices,
                                                             columnIndices,
                                                             blockSize,
                                                             infoM.get(),
                                                             &bufferSizeM));
    int bufferSizeL = 0;
    OPM_CUSPARSE_SAFE_CALL(impl::cusparseBsrsv2_bufferSize(cuSparseHandle.get(),
                                                           CUSPARSE_MATRIX_ORDER,
                                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                           numberOfRows,
                                                           numberOfNonzeroBlocks,
                                                           descriptionL->get(),
                                                           nonZeroValues,
                                                           rowIndices,
                                                           columnIndices,
                                                           blockSize,
                                                           infoL.get(),
                                                           &bufferSizeL));

    int bufferSizeU = 0;
    OPM_CUSPARSE_SAFE_CALL(impl::cusparseBsrsv2_bufferSize(cuSparseHandle.get(),
                                                           CUSPARSE_MATRIX_ORDER,
                                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                           numberOfRows,
                                                           numberOfNonzeroBlocks,
                                                           descriptionL->get(),
                                                           nonZeroValues,
                                                           rowIndices,
                                                           columnIndices,
                                                           blockSize,
                                                           infoU.get(),
                                                           &bufferSizeU));

    OPM_ERROR_IF(bufferSizeL <= 0, "bufferSizeL is non-positive. Given value is " << bufferSizeL);
    OPM_ERROR_IF(bufferSizeU <= 0, "bufferSizeU is non-positive. Given value is " << bufferSizeU);
    OPM_ERROR_IF(bufferSizeM <= 0, "bufferSizeM is non-positive. Given value is " << bufferSizeM);

    return size_t(std::max(bufferSizeL, std::max(bufferSizeU, bufferSizeM)));
}

template <class M, class X, class Y, int l>
void
CuSeqILU0<M, X, Y, l>::createILU()
{
    OPM_ERROR_IF(!buffer, "Buffer not initialized. Call findBufferSize() then initialize with the appropiate size.");
    OPM_ERROR_IF(!analyzisDone, "Analyzis of matrix not done. Call analyzeMatrix() first.");

    const auto numberOfRows = LU.N();
    const auto numberOfNonzeroBlocks = LU.nonzeroes();
    const auto blockSize = LU.blockSize();

    auto nonZeroValues = LU.getNonZeroValues().data();
    auto rowIndices = LU.getRowIndices().data();
    auto columnIndices = LU.getColumnIndices().data();
    OPM_CUSPARSE_SAFE_CALL(impl::cusparseBsrilu02(cuSparseHandle.get(),
                                                  CUSPARSE_MATRIX_ORDER,
                                                  numberOfRows,
                                                  numberOfNonzeroBlocks,
                                                  LU.getDescription().get(),
                                                  nonZeroValues,
                                                  rowIndices,
                                                  columnIndices,
                                                  blockSize,
                                                  infoM.get(),
                                                  CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                                                  buffer->data()));

    // We need to do this here as well. The first call was to check that we could decompose the system A=LU
    // the second call here is to make sure we can solve LUx=y
    int structuralZero;
    // cusparseXbsrilu02_zeroPivot() calls cudaDeviceSynchronize()
    auto statusPivot = cusparseXbsrilu02_zeroPivot(cuSparseHandle.get(), infoM.get(), &structuralZero);

    OPM_ERROR_IF(statusPivot != CUSPARSE_STATUS_SUCCESS,
                 "Found a structucal zero at LU(" + std::to_string(structuralZero) + ", "
                     + std::to_string(structuralZero) + "). Could not solve LUx = y.");
}

template <class M, class X, class Y, int l>
void
CuSeqILU0<M, X, Y, l>::updateILUConfiguration()
{
    auto bufferSize = findBufferSize();
    if (!buffer || buffer->dim() < bufferSize) {
        buffer.reset(new CuVector<field_type>((bufferSize + sizeof(field_type) - 1) / sizeof(field_type)));
    }
    analyzeMatrix();
    createILU();
}
} // namespace Opm::cuistl
#define INSTANTIATE_CUSEQILU0_DUNE(realtype, blockdim)                                                                 \
    template class ::Opm::cuistl::CuSeqILU0<Dune::BCRSMatrix<Dune::FieldMatrix<realtype, blockdim, blockdim>>,         \
                                            ::Opm::cuistl::CuVector<realtype>,                                         \
                                            ::Opm::cuistl::CuVector<realtype>>;                                        \
    template class ::Opm::cuistl::CuSeqILU0<Dune::BCRSMatrix<Opm::MatrixBlock<realtype, blockdim, blockdim>>,          \
                                            ::Opm::cuistl::CuVector<realtype>,                                         \
                                            ::Opm::cuistl::CuVector<realtype>>


INSTANTIATE_CUSEQILU0_DUNE(double, 1);
INSTANTIATE_CUSEQILU0_DUNE(double, 2);
INSTANTIATE_CUSEQILU0_DUNE(double, 3);
INSTANTIATE_CUSEQILU0_DUNE(double, 4);
INSTANTIATE_CUSEQILU0_DUNE(double, 5);
INSTANTIATE_CUSEQILU0_DUNE(double, 6);

INSTANTIATE_CUSEQILU0_DUNE(float, 1);
INSTANTIATE_CUSEQILU0_DUNE(float, 2);
INSTANTIATE_CUSEQILU0_DUNE(float, 3);
INSTANTIATE_CUSEQILU0_DUNE(float, 4);
INSTANTIATE_CUSEQILU0_DUNE(float, 5);
INSTANTIATE_CUSEQILU0_DUNE(float, 6);
