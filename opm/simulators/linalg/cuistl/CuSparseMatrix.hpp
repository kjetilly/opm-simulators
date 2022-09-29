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
#ifndef OPM_CUSPARSEMATRIX_HEADER_INCLUDED
#define OPM_CUSPARSEMATRIX_HEADER_INCLUDED
#include <cusparse.h>
#include <dune/istl/preconditioner.hh>
#include <iostream>
#include <memory>
#include <opm/common/ErrorMacros.hpp>
#include <opm/simulators/linalg/cuistl/impl/CuMatrixDescription.hpp>
#include <opm/simulators/linalg/cuistl/impl/CuSparseHandle.hpp>
#include <opm/simulators/linalg/cuistl/CuVector.hpp>
#include <vector>

namespace Opm::cuistl
{

/*! \brief Wrapper class for the CuSparse compatible matrix storage.
 *
 */
template <typename T>
class CuSparseMatrix
{
public:
    /// Create the sparse matrix specified by the raw data.
    ///
    /// \note Prefer to use the constructor taking a const reference to a matrix instead.
    ///
    /// \param[in] nonZeroElements the non-zero values of the matrix
    /// \param[in] rowIndices      the row indices of the non-zero elements
    /// \param[in] columnIndices   the column indices of the non-zero elements
    /// \param[in] numberOfNonzeroElements number of nonzero elements
    /// \param[in] blockSize size of each block matrix (typically 3)
    /// \param[in] numberOfRows the number of rows
    CuSparseMatrix(const T* nonZeroElements,
                   const int* rowIndices,
                   const int* columnIndices,
                   int numberOfNonzeroBlocks,
                   int blockSize,
                   int numberOfRows);

//    CuSparseMatrix(const CuSparseMatrix<T>& other) = delete;
//    CuSparseMatrix& operator=(const CuSparseMatrix<T>& other) = delete;

    virtual ~CuSparseMatrix();

    template <class MatrixType>
    static CuSparseMatrix<T> fromMatrix(const MatrixType& matrix);

    void setUpperTriangular();
    void setLowerTriangular();
    void setUnitDiagonal();
    void setNonUnitDiagonal();

    size_t N() const
    {
        return numberOfRows;
    }

    size_t nonzeroes() const
    {
        return numberOfNonzeroBlocks;
    }

    CuVector<T>& getNonZeroValues()
    {
        return nonZeroElements;
    }
    const CuVector<T>& getNonZeroValues() const
    {
        return nonZeroElements;
    }

    CuVector<int>& getRowIndices()
    {
        return rowIndices;
    }
    const CuVector<int>& getRowIndices() const
    {
        return rowIndices;
    }

    CuVector<int>& getColumnIndices()
    {
        return columnIndices;
    }
    const CuVector<int>& getColumnIndices() const
    {
        return columnIndices;
    }

    int dim() const
    {
        return _blockSize * numberOfRows;
    }
    int blockSize() const
    {
        return _blockSize;
    }

    impl::CuSparseMatrixDescription& getDescription()
    {
        return *matrixDescription;
    }

    virtual void mv(const CuVector<T>& x, CuVector<T>& y) const;
    virtual void umv(const CuVector<T>& x, CuVector<T>& y) const;
    virtual void usmv(T alpha, const CuVector<T>& x, CuVector<T>& y) const;
    virtual Dune::SolverCategory::Category category() const;

    template <class MatrixType>
    void updateNonzeroValues(const MatrixType& matrix);

private:
    CuVector<T> nonZeroElements;
    CuVector<int> columnIndices;
    CuVector<int> rowIndices;
    const int numberOfNonzeroBlocks;
    const int numberOfRows;
    const int _blockSize;

    impl::CuSparseMatrixDescriptionPtr matrixDescription;
    impl::CuSparseHandle& cusparseHandle;
};
} // namespace Opm::cuistl
#endif
