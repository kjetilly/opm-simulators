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
#ifndef OPM_PRECONDITIONERCONVERTOFLOATADAPTER_HEADER_INCLUDED
#define OPM_PRECONDITIONERCONVERTOFLOATADAPTER_HEADER_INCLUDED
#include <cusparse.h>
#include <dune/common/simd.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/preconditioner.hh>
#include <opm/simulators/linalg/PreconditionerWithUpdate.hpp>
#include <opm/simulators/linalg/cuistl/impl/CuMatrixDescription.hpp>
#include <opm/simulators/linalg/cuistl/impl/CuSparseHandle.hpp>
#include <opm/simulators/linalg/cuistl/CuSparseMatrix.hpp>
#include <opm/simulators/linalg/cuistl/impl/CuSparseResource.hpp>
#include <opm/simulators/linalg/cuistl/impl/cusparse_constants.hpp>
#include <opm/simulators/linalg/cuistl/impl/cusparse_safe_call.hpp>


namespace Opm::cuistl
{
//!\brief Makes a CUDA preconditioner available to a CPU simulator.
template <class CudaPreconditionerType, class M, class X, class Y, int l = 1>
class PreconditionerConvertToFloatAdapter : public Dune::PreconditionerWithUpdate<X, Y>
{
public:
    //! \brief The matrix type the preconditioner is for.
    typedef typename std::remove_const<M>::type matrix_type;

    //! \brief The domain type of the preconditioner.
    typedef X domain_type;
    //! \brief The range type of the preconditioner.
    typedef Y range_type;
    //! \brief The field type of the preconditioner.
    using field_type = typename X::field_type;
    typedef typename X::field_type block_typefield_type;
    //! \brief scalar type underlying the field_type
    typedef Dune::SimdScalar<field_type> scalar_field_type;

    typedef typename CudaPreconditionerType::domain_type domain_type_to;
    //! \brief The range type of the preconditioner.
    typedef typename CudaPreconditionerType::range_type range_type_to;
    //! \brief The field type of the preconditioner.
    typedef typename domain_type_to::field_type field_type_to;
    using block_type = typename domain_type::block_type;
    //! \brief scalar type underlying the field_type
    typedef Dune::SimdScalar<field_type_to> scalar_field_type_to;
    using XTo = Dune::BlockVector<Dune::FieldVector<scalar_field_type_to, block_type::dimension>>;
    using YTo = Dune::BlockVector<Dune::FieldVector<scalar_field_type_to, block_type::dimension>>;
    using matrix_type_to = typename Dune::BCRSMatrix<
        Dune::FieldMatrix<scalar_field_type_to, block_type::dimension, block_type::dimension>>;

    /*! \brief Constructor.

    Constructor gets all parameters to operate the prec.
       \param A The matrix to operate on.
       \param w The relaxation factor.
            */
    PreconditionerConvertToFloatAdapter(const M& matrix)
        : matrix(matrix)
        , convertedMatrix(createConvertedMatrix())
    {
    }

    /*!
       \brief Prepare the preconditioner.

    \copydoc Preconditioner::pre(X&,Y&)
        */
    virtual void pre([[maybe_unused]] X& x, [[maybe_unused]] Y& b) override
    {
    }

    /*!
       \brief Apply the preconditoner.

    \copydoc Preconditioner::apply(X&,const Y&)
        */
    virtual void apply(X& v, const Y& d) override
    {
        XTo convertedV(v.N());
        for (size_t i = 0; i < v.N(); ++i) {
            for (size_t j = 0; j < block_type::dimension; ++j) {
                // This is probably unnecessary, but doing it anyway:
                convertedV[i][j] = scalar_field_type_to(v[i][j]);
            }
        }
        YTo convertedD(d.N());
        for (size_t i = 0; i < d.N(); ++i) {
            for (size_t j = 0; j < block_type::dimension; ++j) {
                convertedD[i][j] = scalar_field_type_to(d[i][j]);
            }
        }

        underlyingPreconditioner->apply(convertedV, convertedD);

        for (size_t i = 0; i < v.N(); ++i) {
            for (size_t j = 0; j < block_type::dimension; ++j) {
                v[i][j] = scalar_field_type(convertedV[i][j]);
            }
        }
    }

    /*!
       \brief Clean up.

    \copydoc Preconditioner::post(X&)
        */
    virtual void post([[maybe_unused]] X& x) override
    {
    }

    //! Category of the preconditioner (see SolverCategory::Category)
    virtual Dune::SolverCategory::Category category() const
    {
        return underlyingPreconditioner->category();
    }

    virtual void update() override
    {
        updateMatrix();
        underlyingPreconditioner->update();
    }

    const matrix_type_to& getConvertedMatrix() const
    {
        return convertedMatrix;
    }

    void setUnderlyingPreconditioner(const std::shared_ptr<CudaPreconditionerType>& conditioner)
    {
        underlyingPreconditioner = conditioner;
    }


private:
    void updateMatrix()
    {
        const auto nnz = matrix.nonzeroes() * matrix[0][0].N() * matrix[0][0].N();
        const auto dataPointerIn = static_cast<const scalar_field_type*>(&((matrix[0][0][0][0])));
        auto dataPointerOut = static_cast<scalar_field_type_to*>(&((convertedMatrix[0][0][0][0])));

        std::vector<scalar_field_type_to> buffer(nnz, 0);
        for (size_t i = 0; i < nnz; ++i) {
            dataPointerOut[i] = scalar_field_type_to(dataPointerIn[i]);
        }
    }
    matrix_type_to createConvertedMatrix()
    {
        // TODO: Check if this whole conversion can be done more efficiently.
        const auto N = matrix.N();
        matrix_type_to matrixBuilder(N, N, matrix.nonzeroes(), matrix_type_to::row_wise);
        {
            auto rowIn = matrix.begin();
            for (auto rowOut = matrixBuilder.createbegin(); rowOut != matrixBuilder.createend(); ++rowOut) {
                for (auto column = rowIn->begin(); column != rowIn->end(); ++column) {
                    rowOut.insert(column.index());
                }
                ++rowIn;
            }
        }

        for (auto row = matrix.begin(); row != matrix.end(); ++row) {
            for (auto column = row->begin(); column != row->end(); ++column) {
                for (size_t i = 0; i < block_type::dimension; ++i) {
                    for (size_t j = 0; j < block_type::dimension; ++j) {
                        matrixBuilder[row.index()][column.index()][i][j]
                            = scalar_field_type_to(matrix[row.index()][column.index()][i][j]);
                    }
                }
            }
        }

        return matrixBuilder;
    }
    const M& matrix;
    matrix_type_to convertedMatrix;
    //! \brief the underlying preconditioner to use
    std::shared_ptr<CudaPreconditionerType> underlyingPreconditioner;
};
} // end namespace Opm::cuistl

#endif
