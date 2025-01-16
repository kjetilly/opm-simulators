/*
  Copyright 2024 SINTEF AS
  Copyright 2024 Equinor ASA

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

#ifndef TEST_AMGXPRECONDITIONER_HELPER_HPP
#define TEST_AMGXPRECONDITIONER_HELPER_HPP

#include <boost/test/unit_test.hpp>

#include <dune/common/fmatrix.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/operators.hh>
#include <dune/istl/solvers.hh>

#include <opm/simulators/linalg/AmgxPreconditioner.hpp>
#include <opm/simulators/linalg/PropertyTree.hpp>

template<class Matrix>
void setupLaplace2d(long long N, Matrix& mat)
{
    const long long nonZeroes = N*N * 5; // max 5 entries per row (diagonal + 4 neighbors)
    mat.setBuildMode(Matrix::row_wise);
    mat.setSize(N*N, N*N, nonZeroes);

    // Set up sparsity pattern
    for (auto row = mat.createbegin(); row != mat.createend(); ++row) {
        const long long i = row.index();
        long long x = i % N;
        long long y = i / N;

        row.insert(i);  // diagonal
        if (x > 0)      // left neighbor
            row.insert(i-1);
        if (x < N-1)    // right neighbor
            row.insert(i+1);
        if (y > 0)      // upper neighbor
            row.insert(i-N);
        if (y < N-1)    // lower neighbor
            row.insert(i+N);
    }

    // Fill the matrix with values
    for (auto row = mat.begin(); row != mat.end(); ++row) {
        const long long i = row.index();
        long long x = i % N;
        long long y = i / N;

        // Set diagonal
        (*row)[i] = 4.0;

        // Set off-diagonal entries
        if (x > 0)         // left neighbor
            (*row)[i-1] = -1.0;
        if (x < N-1)       // right neighbor
            (*row)[i+1] = -1.0;
        if (y > 0)         // upper neighbor
            (*row)[i-N] = -1.0;
        if (y < N-1)       // lower neighbor
            (*row)[i+N] = -1.0;
    }
}

template<typename MatrixScalar, typename VectorScalar>
void testAmgxPreconditioner()
{
    constexpr long long N = 100; // 100x100 grid
    using Matrix = Dune::BCRSMatrix<Dune::FieldMatrix<MatrixScalar, 1, 1>>;
    using Vector = Dune::BlockVector<Dune::FieldVector<VectorScalar, 1>>;

    // Create matrix
    Matrix matrix;
    setupLaplace2d(N, matrix);

    // Create vectors
    Vector x(N*N), b(N*N);
    x = 100.0;    // Initial guess
    b = 0.0;      // RHS

    // Create operator
    using Operator = Dune::MatrixAdapter<Matrix, Vector, Vector>;
    Operator op(matrix);

    // Set up AMGX parameters
    Opm::PropertyTree prm;

    // Create preconditioner
    auto prec = std::make_shared<Amgx::AmgxPreconditioner<Matrix, Vector, Vector>>(matrix, prm);

    // Create solver
    double reduction = 1e-8;
    long long maxit = 300;
    long long verbosity = 0;
    Dune::LoopSolver<Vector> solver(op, *prec, reduction, maxit, verbosity);

    // Solve
    Dune::InverseOperatorResult res;
    solver.apply(x, b, res);

    // Check convergence
    BOOST_CHECK(res.converged);
    BOOST_CHECK_LT(res.reduction, 1e-8);
}

#endif // TEST_AMGXPRECONDITIONER_HELPER_HPP
