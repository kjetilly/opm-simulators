/*
  Copyright 2019 Equinor ASA

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

#if HAVE_CONFIG_H
#include "config.h"
#endif // HAVE_CONFIG_H

#include <opm/simulators/linalg/gpubridge/Reorder.hpp>

#include <vector>
#include <cassert>


namespace Opm
{
namespace Accelerator
{


/* Check is operations on a node in the matrix can be started
 * A node can only be started if all nodes that it depends on during sequential execution have already completed.*/

bool canBeStarted(const long long rowIndex, const long long *rowPointers, const long long *colIndices, const std::vector<bool>& doneRows) {
    bool canStart = !doneRows[rowIndex];
    long long i, thisDependency;
    if (canStart) {
        for (i = rowPointers[rowIndex]; i < rowPointers[rowIndex + 1]; i++) {
            thisDependency = colIndices[i];
            // Only dependencies on rows that should execute before the current one are relevant
            if (thisDependency >= rowIndex)
                break;
            // Check if dependency has been resolved
            if (!doneRows[thisDependency]) {
                return false;
            }
        }
    }
    return canStart;
}

/*
 * The level scheduling of a non-symmetric, blocked matrix requires access to a CSC encoding and a CSR encoding of the sparsity pattern of the input matrix.
 * This function is based on a standard level scheduling algorithm, like the one described in:
 * "Iterative methods for Sparse Linear Systems" by Yousef Saad in section 11.6.3
 */

void findLevelScheduling(long long *CSRColIndices, long long *CSRRowPointers, long long *CSCRowIndices, long long *CSCColPointers, long long Nb, long long *numColors, long long *toOrder, long long* fromOrder, std::vector<long long>& rowsPerColor) {
    long long activeRowIndex = 0, colorEnd, nextActiveRowIndex = 0;
    long long thisRow;
    std::vector<bool> doneRows(Nb, false);
    std::vector <long long> rowsToStart;

    // since emplace_back() is used to fill, the vector must be empty
    assert(rowsPerColor.empty());

    // find starting rows: rows that are independent from all rows that come before them.
    for (thisRow = 0; thisRow < Nb; thisRow++) {
        if (canBeStarted(thisRow, CSCColPointers, CSCRowIndices, doneRows)) {
            fromOrder[nextActiveRowIndex] = thisRow;
            toOrder[thisRow] = nextActiveRowIndex;
            nextActiveRowIndex++;
        }
    }
    // 'do' compute on all active rows
    for (colorEnd = 0; colorEnd < nextActiveRowIndex; colorEnd++) {
        doneRows[fromOrder[colorEnd]] = true;
    }

    rowsPerColor.emplace_back(nextActiveRowIndex - activeRowIndex);

    while (colorEnd < Nb) {
        // Go over all rows active from the last color, and check which of their neighbours can be activated this color
        for (; activeRowIndex < colorEnd; activeRowIndex++) {
            thisRow = fromOrder[activeRowIndex];

            for (long long i = CSCColPointers[thisRow]; i < CSCColPointers[thisRow + 1]; i++) {
                long long thatRow = CSCRowIndices[i];

                if (canBeStarted(thatRow, CSRRowPointers, CSRColIndices, doneRows)) {
                    rowsToStart.emplace_back(thatRow);
                }
            }
        }
        // 'do' compute on all active rows
        for (size_t i = 0; i < rowsToStart.size(); i++) {
            thisRow = rowsToStart[i];
            if (!doneRows[thisRow]) {
                doneRows[thisRow] = true;
                fromOrder[nextActiveRowIndex] = thisRow;
                toOrder[thisRow] = nextActiveRowIndex;
                nextActiveRowIndex++;
            }
        }
        rowsToStart.clear();
        colorEnd = nextActiveRowIndex;
        rowsPerColor.emplace_back(nextActiveRowIndex - activeRowIndex);
    }

    *numColors = rowsPerColor.size();
}


// based on the scipy package from python, scipy/sparse/sparsetools/csr.h on github
void csrPatternToCsc(long long *CSRColIndices, long long *CSRRowPointers, long long *CSCRowIndices, long long *CSCColPointers, long long Nb) {

    long long nnz = CSRRowPointers[Nb];

    // compute number of nnzs per column
    std::fill(CSCColPointers, CSCColPointers + Nb, 0);

    for (long long n = 0; n < nnz; ++n) {
        CSCColPointers[CSRColIndices[n]]++;
    }

    // cumsum the nnz per col to get CSCColPointers
    for (long long col = 0, cumsum = 0; col < Nb; ++col) {
        long long temp = CSCColPointers[col];
        CSCColPointers[col] = cumsum;
        cumsum += temp;
    }
    CSCColPointers[Nb] = nnz;

    for (long long row = 0; row < Nb; ++row) {
        for (long long j = CSRRowPointers[row]; j < CSRRowPointers[row + 1]; ++j) {
            long long col = CSRColIndices[j];
            long long dest = CSCColPointers[col];
            CSCRowIndices[dest] = row;
            CSCColPointers[col]++;
        }
    }

    for (long long col = 0, last = 0; col <= Nb; ++col) {
        long long temp = CSCColPointers[col];
        CSCColPointers[col] = last;
        last = temp;
    }
}


} // namespace Accelerator
} // namespace Opm
