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
#include <cusparse.h>
#include <dune/common/simd.hh>
#include <dune/istl/preconditioner.hh>
#include <opm/simulators/linalg/cuistl/CuMatrixDescription.hpp>
#include <opm/simulators/linalg/cuistl/CuSparseHandle.hpp>
#include <opm/simulators/linalg/cuistl/CuSparseMatrix.hpp>
#include <opm/simulators/linalg/cuistl/CuSparseResource.hpp>
#include <opm/simulators/linalg/cuistl/cusparse_constants.hpp>
#include <opm/simulators/linalg/cuistl/cusparse_safe_call.hpp>


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
class CuSeqILU0 : public Dune::Preconditioner<X, Y>
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
        : w(w)
        , LU(CuSparseMatrix<field_type>::fromMatrix(A))
        , temporaryStorage(LU.numberOfRows() * LU.blockSize())
        , descriptionL(createLowerDiagonalDescription())
        , descriptionU(createUpperDiagonalDescription())
        , cuSparseHandle(CuSparseHandle::getInstance())
    {
        auto bufferSize = findBufferSize();
        buffer.reset(new CuVector<field_type>(bufferSize));
    }

    /*!
       \brief Prepare the preconditioner.

    \copydoc Preconditioner::pre(X&,Y&)
        */
    virtual void pre(X& x, Y& b)
    {
        DUNE_UNUSED_PARAMETER(x);
        DUNE_UNUSED_PARAMETER(b);
    }

    /*!
       \brief Apply the preconditoner.

    \copydoc Preconditioner::apply(X&,const Y&)
        */
    virtual void apply(X& v, const Y& d)
    {
        // We need to pass the solve routine a scalar to multiply.
        // In our case this scalar is 1.0
        const double one = 1.0;

        const auto numberOfRows = LU.N();
        const auto numberOfNonzeroElements = LU.nonzeroes();
        const auto blockSize = LU.blockSize();

        auto nonZeroValues = LU.getNonZeroValues().data();
        auto rowIndices = LU.getRowIndices().data();
        auto columnIndices = LU.getColumnIndices().data();

        // Solve L temporaryStorage = d
        OPM_CUSPARSE_SAFE_CALL(cusparseDbsrsv2_solve(cuSparseHandle.get(),
                                                     CUSPARSE_MATRIX_ORDER,
                                                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                     numberOfRows,
                                                     numberOfNonzeroElements,
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
                                                     buffer.data()));

        // Solve U v = temporaryStorage
        OPM_CUSPARSE_SAFE_CALL(cusparseDbsrsv2_solve(cuSparseHandle.get(),
                                                     CUSPARSE_MATRIX_ORDER,
                                                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                     numberOfRows,
                                                     numberOfNonzeroElements,
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
                                                     buffer.data()));

        v *= w;
    }

    /*!
       \brief Clean up.

    \copydoc Preconditioner::post(X&)
        */
    virtual void post(X& x)
    {
        DUNE_UNUSED_PARAMETER(x);
    }

    //! Category of the preconditioner (see SolverCategory::Category)
    virtual Dune::SolverCategory::Category category() const
    {
        return Dune::SolverCategory::sequential;
    }

private:
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

    std::unique_ptr<CuVector<field_type>> buffer;
    CuSparseHandle& cuSparseHandle;

    size_t findBufferSize()
    {

        // We have three calls that need buffers:
        //   1) LU decomposition
        //   2) solve Lv = y
        //   3) solve Ux = z
        // we combine these buffers into one since it is not used across calls,
        // however, it was used in the Across project.
        return 1;
        // cusparseDbsrilu02_bufferSize(cusparseHandle.get(),
        //                              CUSPARSE_MATRIX_ORDER,
        //                              Nb,
        //                              nnzb,
        //                              descr_M,
        //                              d_bVals,
        //                              d_bRows,
        //                              d_bCols,
        //                              block_size,
        //                              info_M,
        //                              &d_bufferSize_M);
        // cusparseDbsrsv2_bufferSize(cusparseHandle,
        //                            CUSPARSE_MATRIX_ORDER,
        //                            operation,
        //                            Nb,
        //                            nnzb,
        //                            descr_L,
        //                            d_bVals,
        //                            d_bRows,
        //                            d_bCols,
        //                            block_size,
        //                            info_L,
        //                            &d_bufferSize_L);
        // cusparseDbsrsv2_bufferSize(cusparseHandle,
        //                            order,
        //                            operation,
        //                            Nb,
        //                            nnzb,
        //                            descr_U,
        //                            d_bVals,
        //                            d_bRows,
        //                            d_bCols,
        //                            block_size,
        //                            info_U,
        //                            &d_bufferSize_U);
    }
};
} // end namespace Opm::cuistl

#endif
