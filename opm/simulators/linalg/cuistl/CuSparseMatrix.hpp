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
#include <dune/istl/operators.hh>
#include <memory>
#include <opm/common/ErrorMacros.hpp>
#include <opm/simulators/linalg/cuistl/CuMatrixDescription.hpp>
#include <opm/simulators/linalg/cuistl/CuSparseHandle.hpp>
#include <opm/simulators/linalg/cuistl/CuVector.hpp>
#include <vector>
#include <iostream>

namespace Opm::cuistl
{

/*! \brief Wrapper class for the CuSparse compatible matrix storage.
 *
 */
template <typename T>
class CuSparseMatrix : public Dune::LinearOperator<CuVector<T>, CuVector<T>>
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

    virtual ~CuSparseMatrix();

    template <class MatrixType>
    static CuSparseMatrix<T> fromMatrix(const MatrixType& matrix)
    {
        // TODO: Do we need the static_cast? Do we need more paranthesis than a normal Lisp codebase?
        const T* nonZeroElements = static_cast<const T*>(&((matrix[0][0][0][0])));

        // TODO: Do we need this intermediate storage? Or this shuffling of data?
        std::vector<int> columnIndices;
        std::vector<int> rowIndices;

        rowIndices.push_back(0);

        const int blockSize = matrix[0][0].N();
        const int numberOfRows = matrix.N();
        const int numberOfNonzeroBlocks = matrix.nonzeroes();
        const int numberOfNonzeroElements = blockSize * blockSize * numberOfNonzeroBlocks;

        columnIndices.reserve(numberOfNonzeroBlocks);
        rowIndices.reserve(numberOfRows + 1);
        for (auto& row : matrix) {
            for (auto columnIterator = row.begin(); columnIterator != row.end(); ++columnIterator) {
                columnIndices.push_back(columnIterator.index());
            }
            rowIndices.push_back(columnIndices.size());
        }

        // Sanity check
        // h_rows and h_cols could be changed to 'unsigned int', but cusparse expects 'int'
        if (static_cast<unsigned int>(rowIndices[matrix.N()]) != matrix.nonzeroes()) {
            OPM_THROW(std::logic_error, "Error size of rows do not sum to number of nonzeroes in CuSparseMatrix.");
        }

        if (rowIndices.size() != numberOfRows + 1) {
            OPM_THROW(std::logic_error, "Row indices do not match for CuSparseMatrix.");
        }

        if (columnIndices.size() != numberOfNonzeroBlocks) {
            OPM_THROW(std::logic_error, "Column indices do not match for CuSparseMatrix.");
        }

        return CuSparseMatrix<T>(
            nonZeroElements, rowIndices.data(), columnIndices.data(), numberOfNonzeroBlocks, blockSize, numberOfRows);
    }

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

    CuSparseMatrixDescription& getDescription()
    {
        return *matrixDescription;
    }

    virtual void apply (const CuVector<T>& x, CuVector<T>& y) const override;
    virtual void applyscaleadd (T alpha, const CuVector<T>& x, CuVector<T>& y) const override;
    virtual Dune::SolverCategory::Category category() const override;

    template <class MatrixType>
    void updateNonzeroValues(const MatrixType& matrix) {
        if (nonzeroes() != matrix.nonzeroes()) {
            OPM_THROW(std::logic_error, "Matrix does not have the same number of non-zero elements.");
        }

        if (matrix[0][0].N() != blockSize()) {
            OPM_THROW(std::logic_error, "Matrix does not have the same blocksize.");
        }


        if (matrix.N() != N()) {
            OPM_THROW(std::logic_error, "Matrix does not have the same number of rows.");
        }

        const T* newNonZeroElements = static_cast<const T*>(&((matrix[0][0][0][0])));
        nonZeroElements.copyFromHost(newNonZeroElements, nonzeroes() * blockSize() * blockSize());
    }
private:
    CuVector<T> nonZeroElements;
    CuVector<int> columnIndices;
    CuVector<int> rowIndices;
    const int numberOfNonzeroBlocks;
    const int numberOfRows;
    const int _blockSize;

    CuSparseMatrixDescriptionPtr matrixDescription;
    CuSparseHandle& cusparseHandle;
};
} // namespace Opm::cuistl
#endif
