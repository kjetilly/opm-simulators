#include <config.h>

#define BOOST_TEST_MODULE TestCuSparseMatrix

#include <boost/test/unit_test.hpp>
#include <opm/simulators/linalg/cuistl/CuVector.hpp>
#include <opm/simulators/linalg/cuistl/cuda_safe_call.hpp>
#include <opm/simulators/linalg/cuistl/PreconditionerAdapter.hpp>
#include <dune/istl/bcrsmatrix.hh>
#include <memory>

BOOST_AUTO_TEST_CASE(TestConstruction1D) 
{
     // Here we will test a simple 1D finite difference scheme for 
    // the Laplace equation:
    //
    //    -\Delta u = f on [0,1]
    //
    // Using a central difference approximation of \Delta u, this can 
    // be approximated by
    //
    //    -(u_{i+1}-2u_i+u_{i-1})/Dx^2 = f(x_i)
    //
    // giving rise to the matrix
    //
    //     -2  1  0  0 ... 0  0
    //      1 -2  1  0  0 ... 0
    //      ....
    //      0  0  0  ...1 -2  1
    //      0  0  0  ...   1 -2

    const int N = 5;
    const int nonZeroes = N*3-2;
    using M = Dune::FieldMatrix<double,1,1>;
    using SpMatrix = Dune::BCRSMatrix<M>;
    using Vector = Dune::BlockVector<Dune::FieldVector<double, 1>>;
    
    SpMatrix B(N, N, nonZeroes, SpMatrix::row_wise);
    for(auto row=B.createbegin(); row!=B.createend(); ++row)
    {
        // Add nonzeros for left neighbour, diagonal and right neighbour
        if(row.index()>0) {
            row.insert(row.index()-1);
        }
        row.insert(row.index());
        if(row.index()<B.N()-1) {
            row.insert(row.index()+1);
        }
    }
    // This might not be the most elegant way of filling in a Dune sparse matrix, but it works.
    for (int i = 0; i < N; ++i) {
        B[i][i] = -2;
        if (i < N - 1) {
            B[i][i+1] = 1;
        }

        if (i > 0) {
            B[i][i-1] = 1;
        }
    }

    auto cuSparseMatrix = Opm::cuistl::CuSparseMatrix<double>::fromMatrix(B);

    const auto& nonZeroValuesCuda = cuSparseMatrix.getNonZeroValues();
    std::vector<double> buffer(cuSparseMatrix.nonzeroes(), 0.0);
    nonZeroValuesCuda.copyToHost(buffer.data(), buffer.size());
    const double* nonZeroElements = static_cast<const double*>(&((B[0][0][0][0])));
    BOOST_CHECK_EQUAL_COLLECTIONS(buffer.begin(), buffer.end(), nonZeroElements, nonZeroElements + B.nonzeroes());
    for (int i = 0; i < buffer.size(); ++i) {
        std::cout << buffer[i] << ", " << nonZeroElements[i] << std::endl;
    }
}

