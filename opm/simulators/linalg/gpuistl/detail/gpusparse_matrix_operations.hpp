/*
  Copyright 2022-2023 SINTEF AS

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
#ifndef OPM_GPUISTL_GPUSPARSE_MATRIX_OPERATIONS_HPP
#define OPM_GPUISTL_GPUSPARSE_MATRIX_OPERATIONS_HPP
#include <cstddef>
#include <vector>
namespace Opm::gpuistl::detail
{
/**
 * @brief Reorders the elements of a matrix by copying them from one matrix to another using a permutation list
 * @param srcMatrix The source matrix we will copy data from
 * @param srcRowIndices Pointer to vector on GPU containing row indices for the source matrix compliant wiht bsr format
 * @param [out] dstMatrix The destination matrix that we copy data to
 * @param dstRowIndices Pointer to vector on GPU containing riw indices for the destination matrix compliant wiht bsr
 * format
 * @param naturalToReordered Permuation list that converts indices in the src matrix to the indices in the dst matrix
 * @param numberOfRows The number of rows in the matrices
 */
template <class T, long long blocksize>
void copyMatDataToReordered(T* srcMatrix,
                            long long* srcRowIndices,
                            T* dstMatrix,
                            long long* dstRowIndices,
                            long long* naturalToReordered,
                            size_t numberOfRows,
                            long long threadBlockSize);

/**
 * @brief Reorders the elements of a matrix by copying them from one matrix to a split matrix using a permutation list
 * @param srcMatrix The source matrix we will copy data from
 * @param srcRowIndices Pointer to vector on GPU containing row indices for the source matrix compliant wiht bsr format
 * @param [out] dstLowerMatrix The destination of entries that originates from the strictly lower triangular matrix
 * @param dstRowIndices Pointer to vector on GPU containing rww indices for the destination lower matrix compliant wiht
 * bsr format
 * @param [out] dstUpperMatrix The destination of entries that originates from the strictly upper triangular matrix
 * @param dstRowIndices Pointer to vector on GPU containing riw indices for the destination upper matrix compliant wiht
 * bsr format
 * @param [out] dstDiag The destination buffer for the diagonal part of the matrix
 * @param naturalToReordered Permuation list that converts indices in the src matrix to the indices in the dst matrix
 * @param numberOfRows The number of rows in the matrices
 */
template <class T, long long blocksize>
void copyMatDataToReorderedSplit(T* srcMatrix,
                                 long long* srcRowIndices,
                                 long long* srcColumnIndices,
                                 T* dstLowerMatrix,
                                 long long* dstLowerRowIndices,
                                 T* dstUpperMatrix,
                                 long long* dstUpperRowIndices,
                                 T* dstDiag,
                                 long long* naturalToReordered,
                                 size_t numberOfRows,
                                 long long threadBlockSize);

} // namespace Opm::gpuistl::detail
#endif
