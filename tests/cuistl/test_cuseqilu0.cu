#include <config.h>

#define BOOST_TEST_MODULE TestCuSeqILU0

#include <boost/test/unit_test.hpp>
#include <opm/simulators/linalg/cuistl/CuVector.hpp>
#include <opm/simulators/linalg/cuistl/cuda_safe_call.hpp>
#include <opm/simulators/linalg/cuistl/CuSeqILU0.hpp>
#include <opm/simulators/linalg/cuistl/PreconditionerAdapter.hpp>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/preconditioners.hh>
#include <memory>

BOOST_AUTO_TEST_CASE(TestFiniteDifference1D) 
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
    using CuILU0 = Opm::cuistl::CuSeqILU0<SpMatrix, Opm::cuistl::CuVector<double>, Opm::cuistl::CuVector<double>>;
    
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

    
    auto duneILU = Dune::SeqILU<SpMatrix, Vector, Vector>(B, 1.0);
 
    auto cuILU = Opm::cuistl::PreconditionerAdapter<CuILU0, SpMatrix, Vector, Vector>(std::make_shared<CuILU0>(B, 1.0));
    
    // check for the standard basis {e_i} 
    // (e_i=(0,...,0, 1 (i-th place), 0, ..., 0))
    for (int i = 0; i < N; ++i) {
        Vector inputVector(N);
        inputVector[i][0] = 1.0;
        Vector outputVectorDune(N);
        Vector outputVectorCuistl(N);
        duneILU.apply(outputVectorDune, inputVector);
        cuILU.apply(outputVectorCuistl, inputVector);
        BOOST_CHECK_EQUAL_COLLECTIONS(&outputVectorDune[0][0], &outputVectorDune[0][0] + N,
            &outputVectorCuistl[0][0], &outputVectorCuistl[0][0]+N);
    }
}


