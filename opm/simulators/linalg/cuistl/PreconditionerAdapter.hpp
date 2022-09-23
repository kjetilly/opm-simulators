/*
  Copyright SINTEF AS 2022

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
#ifndef OPM_PRECONDITIONERADAPTER_HEADER_INCLUDED
#define OPM_PRECONDITIONERADAPTER_HEADER_INCLUDED
#include <cusparse.h>
#include <dune/common/simd.hh>
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
//!\brief Makes a CUDA preconditioner available to a CPU simulator.
template <class X, class Y>
class PreconditionerAdapter : public Dune::PreconditionerWithUpdate<X, Y>
{
public:
    //! \brief The domain type of the preconditioner.
    typedef X domain_type;
    //! \brief The range type of the preconditioner.
    typedef Y range_type;
    //! \brief The field type of the preconditioner.
    typedef typename X::field_type field_type;
    //! \brief scalar type underlying the field_type
    typedef Dune::SimdScalar<field_type> scalar_field_type;
    using CudaPreconditionerType = Dune::PreconditionerWithUpdate<CuVector<scalar_field_type>, CuVector<scalar_field_type>>;

    /*! \brief Constructor.

    Constructor gets all parameters to operate the prec.
       \param A The matrix to operate on.
       \param w The relaxation factor.
            */
    PreconditionerAdapter(std::shared_ptr<CudaPreconditionerType> preconditioner_)
        : underlyingPreconditioner(preconditioner_)
    {

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
        OPM_TIME_TO_FILE_ALWAYS(preconditioneradapter, v.dim());
        if (!inputBuffer) {
            inputBuffer.reset(new CuVector<field_type>(v.dim()));
            outputBuffer.reset(new CuVector<field_type>(v.dim()));
        }
        inputBuffer->copyFrom(d);
        underlyingPreconditioner->apply(*outputBuffer, *inputBuffer);
        outputBuffer->copyTo(v);
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
    virtual Dune::SolverCategory::Category category() const
    {
        return underlyingPreconditioner->category();
    }

    virtual void update() override {
        underlyingPreconditioner->update();
    }

    std::shared_ptr<CudaPreconditionerType> getUnderlyingPreconditioner() {
        return underlyingPreconditioner;
    }

private:
    //! \brief the underlying preconditioner to use
    std::shared_ptr<CudaPreconditionerType> underlyingPreconditioner;

    std::unique_ptr<CuVector<field_type>> inputBuffer;
    std::unique_ptr<CuVector<field_type>> outputBuffer;
};
} // end namespace Opm::cuistl

#endif
