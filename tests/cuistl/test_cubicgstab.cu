#include <config.h>

#define BOOST_TEST_MODULE TestCuBICGSTAB
#define BOOST_TEST_NO_MAIN


#include <boost/mpl/list.hpp>
#include <boost/test/unit_test.hpp>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/preconditioners.hh>
#include <opm/simulators/linalg/cuistl/CuSeqILU0.hpp>
#include <opm/simulators/linalg/cuistl/CuVector.hpp>
#include <opm/simulators/linalg/cuistl/PreconditionerAdapter.hpp>
#include <opm/simulators/linalg/cuistl/cuda_safe_call.hpp>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/istl/solvers.hh>

#include <limits>
#include <memory>


using NumericTypes = boost::mpl::list<double>;

BOOST_AUTO_TEST_CASE_TEMPLATE(TestFiniteDifference1D, T, NumericTypes)
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
    const int nonZeroes = N * 3 - 2;
    using M = Dune::FieldMatrix<T, 2, 2>;
    using SpMatrix = Dune::BCRSMatrix<M>;
    using Vector = Dune::BlockVector<Dune::FieldVector<T, 1>>;
    using CuILU0 = Opm::cuistl::CuSeqILU0<SpMatrix, Opm::cuistl::CuVector<T>, Opm::cuistl::CuVector<T>>;

    SpMatrix B(N, N, nonZeroes, SpMatrix::row_wise);
    for (auto row = B.createbegin(); row != B.createend(); ++row) {
        row.insert(row.index());
        if (row.index() < N - 1) {
            row.insert(row.index() + 1);
        } 
        if (row.index() > 0) {
            row.insert(row.index() - 1);
        }
    }
    // This might not be the most elegant way of filling in a Dune sparse matrix, but it works.
    for (int i = 0; i < N; ++i) {
        B[i][i][0][0] = -2;
        B[i][i][1][1] = -2;
        B[i][i][0][1] = 1;
        B[i][i][1][0] = 1;
    }

    auto BonGPU = std::make_shared<Opm::cuistl::CuSparseMatrix<T>>(Opm::cuistl::CuSparseMatrix<T>::fromMatrix(B));
    auto BonGPUAsLinearOperator = std::static_pointer_cast<Dune::LinearOperator<Opm::cuistl::CuVector<T>, Opm::cuistl::CuVector<T>>>(BonGPU);
    auto cuILU = std::make_shared<CuILU0>(B, 1.0);
    auto cuILUAsDunePreconditioner = std::static_pointer_cast<Dune::Preconditioner<Opm::cuistl::CuVector<T>, Opm::cuistl::CuVector<T>>>(cuILU);
    auto scalarProduct = std::make_shared<Dune::ScalarProduct<Opm::cuistl::CuVector<T>>>();

    Dune::ParameterTree config;
    auto solver = Dune::BiCGSTABSolver<Opm::cuistl::CuVector<T>>(BonGPUAsLinearOperator, scalarProduct, cuILUAsDunePreconditioner, 1.0, 10, 0);
    // auto solver = Dune::CGSolver<Opm::cuistl::CuVector<T>>(BonGPUAsLinearOperator, scalarProduct, cuILUAsDunePreconditioner, 1.0, 10, 0, true);

    Opm::cuistl::CuVector<double> x(N*2);
    Opm::cuistl::CuVector<double> y(N*2);
    Dune::InverseOperatorResult result;
    solver.apply(x, y, result);
    // auto solver = Dune::BiCGSTABSolver<Opm::cuistl::CuVector<T>>(BonGPUAsLinearOperator, scalarProduct, cuILUAsDunePreconditioner, config);
}   
    

bool init_unit_test_func()
{
    return true;
}

int main(int argc, char** argv)
{
    [[maybe_unused]] const auto& helper = Dune::MPIHelper::instance(argc, argv);
    boost::unit_test::unit_test_main(&init_unit_test_func,
                                     argc, argv);
}