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

#include <dune/common/simd.hh>
#include <dune/istl/preconditioner.hh>
#include <opm/simulators/linalg/PreconditionerWithUpdate.hpp>
#include <opm/simulators/linalg/cuistl/CuSparseMatrix.hpp>
#include <opm/simulators/linalg/cuistl/impl/CuMatrixDescription.hpp>
#include <opm/simulators/linalg/cuistl/impl/CuSparseHandle.hpp>
#include <opm/simulators/linalg/cuistl/impl/CuSparseResource.hpp>



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
    CuSeqILU0(const M& A, scalar_field_type w);

    /*!
       \brief Prepare the preconditioner.

    \copydoc Preconditioner::pre(X&,Y&)
        */
    virtual void pre(X& x, Y& b) override;

    /*!
       \brief Apply the preconditoner.

    \copydoc Preconditioner::apply(X&,const Y&)
        */
    virtual void apply(X& v, const Y& d) override;

    /*!
       \brief Clean up.

    \copydoc Preconditioner::post(X&)
        */
    virtual void post(X& x) override;

    //! Category of the preconditioner (see SolverCategory::Category)
    virtual Dune::SolverCategory::Category category() const override;

    virtual void update() override;


    static constexpr bool shouldCallPre()
    {
        return false;
    }
    static constexpr bool shouldCallPost()
    {
        return false;
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


    impl::CuSparseMatrixDescriptionPtr descriptionL;
    impl::CuSparseMatrixDescriptionPtr descriptionU;
    impl::CuSparseResource<bsrsv2Info_t> infoL;
    impl::CuSparseResource<bsrsv2Info_t> infoU;
    impl::CuSparseResource<bsrilu02Info_t> infoM;

    std::unique_ptr<CuVector<field_type>> buffer;
    impl::CuSparseHandle& cuSparseHandle;

    bool analyzisDone = false;

    void analyzeMatrix();
    size_t findBufferSize();

    void createILU();

    void updateILUConfiguration();
};
} // end namespace Opm::cuistl

#endif
