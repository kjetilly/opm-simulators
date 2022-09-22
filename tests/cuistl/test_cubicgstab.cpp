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
#include <dune/istl/matrixmarket.hh>
// #include <opm/simulators/linalg/WriteSystemMatrixHelper.hpp>
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
    auto BOperator = std::make_shared<Dune::MatrixAdapter< Opm::cuistl::CuSparseMatrix<T>, Opm::cuistl::CuVector<T>, Opm::cuistl::CuVector<T>>>(BonGPU);
    auto cuILU = std::make_shared<CuILU0>(B, 1.0);
    auto scalarProduct = std::make_shared<Dune::SeqScalarProduct<Opm::cuistl::CuVector<T>>>();

    auto solver = Dune::BiCGSTABSolver<Opm::cuistl::CuVector<T>>(BOperator, scalarProduct, cuILU, 1.0, 100, 0);
    std::vector<T> correct(N*2, 2.0);
    std::vector<T> initialGuess(N*2, 0.0);
    Opm::cuistl::CuVector<T> x(N*2);
    Opm::cuistl::CuVector<T> y(N*2);
    x.copyFromHost(correct.data(), correct.size());
    BonGPU->mv(x, y);
    x.copyFromHost(initialGuess.data(), initialGuess.size());

    Dune::InverseOperatorResult result;
    Opm::cuistl::CuVector<T> tmp(N*2);
    tmp.copyFromHost(correct.data(), correct.size());
    tmp -= x;
    auto normBefore = tmp.two_norm();
    BOOST_CHECK_GT(normBefore, 0.5);
    solver.apply(x, y, result);
    tmp.copyFromHost(correct.data(), correct.size());
    tmp -= x;
    auto normAfter = tmp.two_norm();
    BOOST_CHECK_CLOSE(normAfter, 0.0, 1e-7);
}

BOOST_AUTO_TEST_CASE(TestLoadFromFile, * boost::unit_test::tolerance(1e-7))
{
    using T = double;   
    static constexpr size_t dim = 3;
    using M = Dune::FieldMatrix<T, dim, dim>;
    using SpMatrix = Dune::BCRSMatrix<M>;
    using Vector = Dune::BlockVector<Dune::FieldVector<T, dim>>;
    using CuILU0 = Opm::cuistl::CuSeqILU0<SpMatrix, Opm::cuistl::CuVector<T>, Opm::cuistl::CuVector<T>>;

    SpMatrix B(300, 300, 16020/9, SpMatrix::row_wise);

    Dune::loadMatrixMarket(B, "../matrix.mm");
    //B.compress();
    const size_t N = B.N();
    auto BonGPU = std::make_shared<Opm::cuistl::CuSparseMatrix<T>>(Opm::cuistl::CuSparseMatrix<T>::fromMatrix(B));
    auto BOperator = std::make_shared<Dune::MatrixAdapter< Opm::cuistl::CuSparseMatrix<T>, Opm::cuistl::CuVector<T>, Opm::cuistl::CuVector<T>>>(BonGPU);
    auto cuILU = std::make_shared<CuILU0>(B, 1.0);
    auto scalarProduct = std::make_shared<Dune::SeqScalarProduct<Opm::cuistl::CuVector<T>>>();

    auto solver = Dune::BiCGSTABSolver<Opm::cuistl::CuVector<T>>(BOperator, scalarProduct, cuILU, 1.0, 100, 0);
    std::vector<T> correct(N*dim, 1.0);
    //correct[N/2] =  1.0;
    std::vector<T> initialGuess(N*dim, 0.0);
    //initialGuess[N/2]  = 0.5;
    Opm::cuistl::CuVector<T> x(correct);
    Opm::cuistl::CuVector<T> y(initialGuess);

    Vector xHost(N), yHost(N);
    x.copyTo(xHost);
    xHost = 1.0;
    x.copyFrom(xHost);

    BonGPU->mv(x, y);

    
    B.mv(xHost, yHost);
    std::vector<double> dataOnHost(y.dim());
    y.copyToHost(dataOnHost);
    //BOOST_CHECK_EQUAL_COLLECTIONS(dataOnHost.begin(),
    //dataOnHost.end(), &yHost[0][0], &yHost[0][0] + dim*N);
    int totalfound = 0;
    for (size_t i = 0; i < yHost.N(); ++i) {
        for (size_t c = 0; c < dim; ++c) {
            if (std::abs(yHost[i][c] - dataOnHost[i * dim + c])/std::abs(yHost[i][c]) > 1e-4) {
                totalfound++;
            }
            // bool failed = false;
            // if (std::abs(yHost[i][c]) > 1e-8) { 
            //     std::cout << "Dune is " << yHost[i][c] << " ";
            //     failed = true;
            // }
            // if (std::abs(dataOnHost[i * dim + c]) > 1e-8) { 
            //     std::cout << "cuistl is " << dataOnHost[i * dim + c] << " ";
            //     failed = true;
            // }
            // if (failed) {
            //     std::cout << std::endl;
            // }
            // std::cout << yHost[i][c] << " " << dataOnHost[i * dim + c] << std::endl;
            // BOOST_TEST_EQUAL(yHost[i][c], dataOnHost[i * dim + c], 1e-6);
            BOOST_CHECK_CLOSE_FRACTION(yHost[i][c], dataOnHost[i * dim + c], 1e-4);
        }
    }
    std::cout << "totalfound = " << totalfound << std::endl;
    std::cout << "percent found: " << 100.0 * totalfound / double(N*dim) << " %" << std::endl;

    x.copyFromHost(initialGuess.data(), initialGuess.size());

    Dune::InverseOperatorResult result;
    Opm::cuistl::CuVector<T> tmp(N*dim);
    tmp.copyFromHost(correct.data(), correct.size());
    tmp -= x;
    auto normBefore = tmp.two_norm();
    std::cout << normBefore << std::endl;
    BOOST_CHECK_GT(normBefore, 0.2);

    auto normy =  y.two_norm();

    y.copyTo(yHost);
    auto normyhost = yHost.two_norm();
    std::cout << "normy = "  << normy << std::endl;
    std::cout << "normyhost = "  << normyhost << std::endl;

    solver.apply(x, y, result);
    tmp.copyFromHost(correct.data(), correct.size());
    tmp -= x;
    auto normAfter = tmp.two_norm();
    BOOST_CHECK_LT(normAfter, 1e-6);
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