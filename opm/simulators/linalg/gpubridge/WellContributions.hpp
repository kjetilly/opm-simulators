/*
  Copyright 2020 Equinor ASA

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

#ifndef WELLCONTRIBUTIONS_HEADER_INCLUDED
#define WELLCONTRIBUTIONS_HEADER_INCLUDED

#include <memory>
#include <vector>

#include <umfpack.h>
#include <dune/common/version.hh>

namespace Opm {

template<class Scalar> class MultisegmentWellContribution;

/// This class serves to eliminate the need to include the WellContributions into the matrix (with --matrix-add-well-contributions=true) for the cusparseSolver or openclSolver.
/// If the --matrix-add-well-contributions commandline parameter is true, this class should still be used, but be empty.
/// StandardWell and MultisegmentWell are supported for both cusparseSolver and openclSolver.
/// A single instance (or pointer) of this class is passed to the GpuSolver.
/// For StandardWell, this class contains all the data and handles the computation. For MultisegmentWell, the vector 'multisegments' contains all the data. For more information, check the MultisegmentWellContribution class.

/// A StandardWell uses C, D and B and performs y -= (C^T * (D^-1 * (B*x)))
/// B and C are vectors, disguised as matrices and contain blocks of StandardWell::numEq by StandardWell::numStaticWellEq
/// D is a block, disguised as matrix, the square block has size StandardWell::numStaticWellEq. D is actually stored as D^-1
/// B*x and D*B*x are a vector with numStaticWellEq doubles
/// C*D*B*x is a blocked matrix with a symmetric sparsity pattern, contains square blocks with size numEq. For every columnindex i, j in StandardWell::duneB_, there is a block on (i, j) in C*D*B*x.
///
/// This class is used in 3 phases:
/// - get total size of all wellcontributions that must be stored here
/// - allocate memory
/// - copy data of wellcontributions
template<class Scalar>
class WellContributions
{
public:
    static std::unique_ptr<WellContributions> create(const std::string& accelerator_mode, bool useWellConn);

    using UMFPackIndex = SuiteSparse_long;
    /// StandardWell has C, D and B matrices that need to be copied
    enum class MatrixType {
        C,
        D,
        B
    };

protected:
    bool allocated = false;

    size_t N;                          // number of rows (not blockrows) in vectors x and y
    size_t dim;                        // number of columns in blocks in B and C, equal to StandardWell::numEq
    size_t dim_wells;                  // number of rows in blocks in B and C, equal to StandardWell::numStaticWellEq
    size_t num_blocks = 0;             // total number of blocks in all wells
    size_t num_std_wells = 0;          // number of StandardWells in this object
    size_t num_ms_wells = 0;           // number of MultisegmentWells in this object, must equal multisegments.size()
    size_t num_blocks_so_far = 0;      // keep track of where next data is written
    size_t num_std_wells_so_far = 0;   // keep track of where next data is written
    std::vector<size_t> val_pointers;    // val_pointers[wellID] == index of first block for this well in Ccols and Bcols

    std::vector<std::unique_ptr<MultisegmentWellContribution<Scalar>>> multisegments;

public:
    size_t getNumWells(){
        return num_std_wells + num_ms_wells;
    }

    /// Indicate how large the next StandardWell is, this function cannot be called after alloc() is called
    /// \param[in] numBlocks   number of blocks in C and B of next StandardWell
    void addNumBlocks(size_t numBlocks);

    /// Allocate memory for the StandardWells
    void alloc();

    /// Empty destructor.
    virtual ~WellContributions();

    /// Indicate how large the blocks of the StandardWell (C and B) are
    /// \param[in] dim         number of columns
    /// \param[in] dim_wells   number of rows
    void setBlockSize(size_t dim, size_t dim_wells);

    /// Set size of vector that the wells are applied to
    /// \param[in] N          size of vector
    void setVectorSize(unsigned N);

    /// Store a matrix in this object, in blocked csr format, can only be called after alloc() is called
    /// \param[in] type        indicate if C, D or B is sent
    /// \param[in] colIndices  columnindices of blocks in C or B, ignored for D
    /// \param[in] values      array of nonzeroes
    /// \param[in] val_size    number of blocks in C or B, ignored for D
    void addMatrix(MatrixType type, long long* colIndices, Scalar* values, size_t val_size);

    /// Add a MultisegmentWellContribution, actually creates an object on heap that is destroyed in the destructor
    /// Matrices C and B are passed in Blocked CSR, matrix D in CSC
    /// \param[in] dim              size of blocks in vectors x and y, equal to MultisegmentWell::numEq
    /// \param[in] dim_wells        size of blocks of C, B and D, equal to MultisegmentWell::numWellEq
    /// \param[in] Mb               number of blockrows in C, B and D
    /// \param[in] Bvalues          nonzero values of matrix B
    /// \param[in] BcolIndices      columnindices of blocks of matrix B
    /// \param[in] BrowPointers     rowpointers of matrix B
    /// \param[in] DnumBlocks       number of blocks in D
    /// \param[in] Dvalues          nonzero values of matrix D
    /// \param[in] DcolPointers     columnpointers of matrix D
    /// \param[in] DrowIndices      rowindices of matrix D
    /// \param[in] Cvalues          nonzero values of matrix C
    void addMultisegmentWellContribution(size_t dim,
                                         size_t dim_wells,
                                         size_t Mb,
                                         std::vector<Scalar>& Bvalues,
                                         std::vector<size_t>& BcolIndices,
                                         std::vector<size_t>& BrowPointers,
                                         size_t DnumBlocks,
                                         Scalar* Dvalues,
                                         UMFPackIndex* DcolPointers,
                                         UMFPackIndex* DrowIndices,
                                         std::vector<Scalar>& Cvalues);
protected:
    //! \brief API specific allocation.
    virtual void APIalloc() {}

    /// Api specific upload of matrix.
    virtual void APIaddMatrix(MatrixType, long long*, Scalar*, size_t) {}
};

} //namespace Opm

#endif
