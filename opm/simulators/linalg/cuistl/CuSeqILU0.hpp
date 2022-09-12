/*
  Copyright SINTEF AS

  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef OPM_CUSEQILU0_HEADER_INCLUDED
#define OPM_CUSEQILU0_HEADER_INCLUDED
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <dune/common/simd.hh>
#include <dune/common/unused.hh>
#include <dune/istl/preconditioner.hh>
#include <opm/simulators/linalg/cuistl/CuMatrixDescription.hpp>
#include <opm/simulators/linalg/cuistl/CuSparseHandle.hpp>
#include <opm/simulators/linalg/cuistl/CuSparseMatrix.hpp>
#include <opm/simulators/linalg/cuistl/CuSparseResource.hpp>
#include <opm/simulators/linalg/cuistl/cusparse_constants.hpp>
#include <opm/simulators/linalg/cuistl/cusparse_safe_call.hpp>
#include <opm/simulators/linalg/PreconditionerWithUpdate.hpp>
#include <opm/simulators/linalg/cuistl/time_to_file.hpp>


namespace Opm::cuistl
{
/*!
   \brief Sequential ILU0 preconditioner.

Wraps the naked ISTL generic ILU0 preconditioner into the solver framework.

     \tparam M The matrix type to operate on
     \tparam X Type of the update
     \tparam Y Type of the defect
     \tparam l Ignored. Just there to have the same number of template arguments
        as other preconditioners.
            */
template <class M, class X, class Y, int l = 1>
class CuSeqILU0 : public Dune::PreconditionerWithUpdate<X, Y>
{
public:
    //! \brief The matrix type the preconditioner is for.
    typedef typename std::remove_const<M>::type matrix_type;
    //! \brief The domain type of the preconditioner.
    typedef X domain_type;
    //! \brief The range type of the preconditioner.
    typedef Y range_type;
    //! \brief The field type of the preconditioner.
    typedef typename X::field_type field_type;
    //! \brief scalar type underlying the field_type
    typedef Dune::SimdScalar<field_type> scalar_field_type;

    /*! \brief Constructor.

    Constructor gets all parameters to operate the prec.
       \param A The matrix to operate on.
       \param w The relaxation factor.
            */
    CuSeqILU0(const M& A, scalar_field_type w)
        : underlyingMatrix(A)
        , w(w)
        , LU(CuSparseMatrix<field_type>::fromMatrix(A))
        , temporaryStorage(LU.N() * LU.blockSize())
        , descriptionL(createLowerDiagonalDescription())
        , descriptionU(createUpperDiagonalDescription())
        , cuSparseHandle(CuSparseHandle::getInstance())
    {
        // Some sanity check
        if (A.N() != LU.N()) {
            OPM_THROW(std::runtime_error,
                      "CuSparse matrix not same size as DUNE matrix. " + std::to_string(LU.N()) + " vs "
                          + std::to_string(A.N()));
        }
        if (A[0][0].N() != LU.blockSize()) {
            OPM_THROW(std::runtime_error,
                      "CuSparse matrix not same blocksize as DUNE matrix. " + std::to_string(LU.blockSize()) + " vs "
                          + std::to_string(A[0][0].N()));
        }
        if (A.N() * A[0][0].N() != LU.dim()) {
            OPM_THROW(std::runtime_error,
                      "CuSparse matrix not same dimension as DUNE matrix. " + std::to_string(LU.dim()) + " vs "
                          + std::to_string(A.N() * A[0][0].N()));
        }

        if (A.nonzeroes() != LU.nonzeroes()) {
            OPM_THROW(std::runtime_error,
                      "CuSparse matrix not same number of non zeroes as DUNE matrix. " + std::to_string(LU.nonzeroes())
                          + " vs " + std::to_string(A.nonzeroes()));
        }


        // https://docs.nvidia.com/cuda/cusparse/index.html#csrilu02_solve
        updateILUConfiguration();
    }

    /*!
       \brief Prepare the preconditioner.

    \copydoc Preconditioner::pre(X&,Y&)
        */
    virtual void pre(X& x, Y& b) override
    {
        DUNE_UNUSED_PARAMETER(x);
        DUNE_UNUSED_PARAMETER(b);
    }

    /*!
       \brief Apply the preconditoner.

    \copydoc Preconditioner::apply(X&,const Y&)
        */
    virtual void apply(X& v, const Y& d) override
    {
        
        // We need to pass the solve routine a scalar to multiply.
        // In our case this scalar is 1.0
        const double one = 1.0;

        const auto numberOfRows = LU.N();
        const auto numberOfNonzeroBlocks = LU.nonzeroes();
        const auto blockSize = LU.blockSize();

        auto nonZeroValues = LU.getNonZeroValues().data();
        auto rowIndices = LU.getRowIndices().data();
        auto columnIndices = LU.getColumnIndices().data();

        // Solve L temporaryStorage = d
        {
        TimeToFile timer("cuistl", LU.nonzeroes() * LU.blockSize() * LU.blockSize());
        OPM_CUSPARSE_SAFE_CALL(cusparseDbsrsv2_solve(cuSparseHandle.get(),
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
                                                     CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                                                     buffer->data()));

        // Solve U v = temporaryStorage
        OPM_CUSPARSE_SAFE_CALL(cusparseDbsrsv2_solve(cuSparseHandle.get(),
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
                                                     cudaDeviceSynchronize();
        }

        v *= w;
    }

    /*!
       \brief Clean up.

    \copydoc Preconditioner::post(X&)
        */
    virtual void post(X& x) override
    {
        DUNE_UNUSED_PARAMETER(x);
    }

    //! Category of the preconditioner (see SolverCategory::Category)
    virtual Dune::SolverCategory::Category category() const override
    {
        return Dune::SolverCategory::sequential;
    }

    virtual void update() override {
        LU.updateNonzeroValues(underlyingMatrix);

        // We can probably get away with updating less than this,
        // but for now we're sticking to the safe route.
        updateILUConfiguration();
    }

private:
    //! \brief Reference to the underlying matrix
    const M& underlyingMatrix;
    //! \brief The relaxation factor to use.
    scalar_field_type w;

    //! This is the storage for the LU composition.
    //! Initially this will have the values of A, but will be
    //! modified in the constructor to be te proper LU decomposition.
    CuSparseMatrix<field_type> LU;

    CuVector<field_type> temporaryStorage;


    CuSparseMatrixDescriptionPtr descriptionL;
    CuSparseMatrixDescriptionPtr descriptionU;
    CuSparseResource<bsrsv2Info_t> infoL;
    CuSparseResource<bsrsv2Info_t> infoU;
    CuSparseResource<bsrilu02Info_t> infoM;

    std::unique_ptr<CuVector<field_type>> buffer;
    CuSparseHandle& cuSparseHandle;

    bool analyzisDone = false;

    void analyzeMatrix()
    {
        // TODO: Move this method to the .cu file
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
        OPM_CUSPARSE_SAFE_CALL(cusparseDbsrilu02_analysis(cuSparseHandle.get(),
                                                          CUSPARSE_MATRIX_ORDER,
                                                          numberOfRows,
                                                          numberOfNonzeroBlocks,
                                                          LU.getDescription().get(),
                                                          nonZeroValues,
                                                          rowIndices,
                                                          columnIndices,
                                                          blockSize,
                                                          infoM.get(),
                                                          CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                                                          buffer->data()));

        // Make sure we can decompose the matrix.
        int structuralZero;
        auto statusPivot = cusparseXbsrilu02_zeroPivot(cuSparseHandle.get(), infoM.get(), &structuralZero);
        if (statusPivot != CUSPARSE_STATUS_SUCCESS) {
            OPM_THROW(std::runtime_error,
                      "Found a structucal zero at A(" + std::to_string(structuralZero) + ", "
                          + std::to_string(structuralZero) + "). Could not decompose LU approx A.\n\n"
                          + "A has dimension: " + std::to_string(LU.N()) + ",\n" + "and has "
                          + std::to_string(LU.nonzeroes()) + " nonzeroes.");
        }
        // analysis of ilu apply
        OPM_CUSPARSE_SAFE_CALL(cusparseDbsrsv2_analysis(cuSparseHandle.get(),
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
                                                        CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                                                        buffer->data()));

        OPM_CUSPARSE_SAFE_CALL(cusparseDbsrsv2_analysis(cuSparseHandle.get(),
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
    int findBufferSize()
    {
        // TODO: Move this method to the .cu file
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
        OPM_CUSPARSE_SAFE_CALL(cusparseDbsrilu02_bufferSize(cuSparseHandle.get(),
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
        OPM_CUSPARSE_SAFE_CALL(cusparseDbsrsv2_bufferSize(cuSparseHandle.get(),
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
        OPM_CUSPARSE_SAFE_CALL(cusparseDbsrsv2_bufferSize(cuSparseHandle.get(),
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

        return std::max(bufferSizeL, std::max(bufferSizeU, bufferSizeM));
    }

    void createILU()
    {
        // TODO: Move this method to the .cu file
        if (!buffer) {
            OPM_THROW(std::runtime_error,
                      "Buffer not initialized. Call findBufferSize() then initialize with the appropiate size.");
        }
        if (!analyzisDone) {
            OPM_THROW(std::runtime_error, "Analyzis of matrix not done. Call analyzeMatrix() first.");
        }
        const auto numberOfRows = LU.N();
        const auto numberOfNonzeroBlocks = LU.nonzeroes();
        const auto blockSize = LU.blockSize();

        auto nonZeroValues = LU.getNonZeroValues().data();
        auto rowIndices = LU.getRowIndices().data();
        auto columnIndices = LU.getColumnIndices().data();
        OPM_CUSPARSE_SAFE_CALL(cusparseDbsrilu02(cuSparseHandle.get(),
                                                 CUSPARSE_MATRIX_ORDER,
                                                 numberOfRows,
                                                 numberOfNonzeroBlocks,
                                                 LU.getDescription().get(),
                                                 nonZeroValues,
                                                 rowIndices,
                                                 columnIndices,
                                                 blockSize,
                                                 infoM.get(),
                                                 CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                                                 buffer->data()));

        // We need to do this here as well. The first call was to check that we could decompose the system A=LU
        // the second call here is to make sure we can solve LUx=y
        int structuralZero;
        // cusparseXbsrilu02_zeroPivot() calls cudaDeviceSynchronize()
        auto statusPivot = cusparseXbsrilu02_zeroPivot(cuSparseHandle.get(), infoM.get(), &structuralZero);

        if (statusPivot != CUSPARSE_STATUS_SUCCESS) {
            OPM_THROW(std::runtime_error,
                      "Found a structucal zero at LU(" + std::to_string(structuralZero) + ", "
                          + std::to_string(structuralZero) + "). Could not solve LUx = y.");
        }
    }

    void updateILUConfiguration() {
        auto bufferSize = findBufferSize();
        if (!buffer || buffer->dim() < bufferSize) {
            buffer.reset(new CuVector<field_type>((bufferSize + sizeof(field_type) - 1) / sizeof(field_type)));
        }
        analyzeMatrix();
        createILU();
    }
};
} // end namespace Opm::cuistl

#endif
