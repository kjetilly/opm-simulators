#ifndef OPM_SOLVERADAPTER_HEADER_INCLUDED
#define OPM_SOLVERADAPTER_HEADER_INCLUDED

#include <memory>

#include <dune/istl/operators.hh>
#include <dune/istl/solver.hh>
#include <opm/common/ErrorMacros.hpp>
#include <opm/simulators/linalg/cuistl/CuVector.hpp>
#include <opm/simulators/linalg/cuistl/CuSparseMatrix.hpp>
#include <opm/simulators/linalg/cuistl/PreconditionerAdapter.hpp>


namespace Opm::cuistl
{

template <class Operator, template <class> class UnderlyingSolver, class X>
class SolverAdapter : public Dune::IterativeSolver<X, X>
{
public:
    using typename Dune::IterativeSolver<X, X>::domain_type;
    using typename Dune::IterativeSolver<X, X>::range_type;
    using typename Dune::IterativeSolver<X, X>::field_type;
    using typename Dune::IterativeSolver<X, X>::real_type;
    using typename Dune::IterativeSolver<X, X>::scalar_real_type;
    using XGPU = Opm::cuistl::CuVector<real_type>;

    // TODO: Use a std::forward
    SolverAdapter(Operator& op,
                  std::shared_ptr<Dune::ScalarProduct<X>> sp,
                  std::shared_ptr<Dune::Preconditioner<X, X>> prec,
                  scalar_real_type reduction,
                  int maxit,
                  int verbose)
        : Dune::IterativeSolver<X, X>(op, *sp, *prec, reduction, maxit, verbose)
        , opOnCPUWithMatrix(op)
        , matrix(CuSparseMatrix<real_type>::fromMatrix(op.getmat()))
        , underlyingSolver(constructSolver(prec, reduction, maxit, verbose))
    {
    }

    virtual void apply (X& x, X& b, double reduction, Dune::InverseOperatorResult& res) {
        // TODO: Can we do this without reimplementing the other function?
        // TODO: [perf] Do we need to update the matrix every time? Probably yes
        matrix.updateNonzeroValues(opOnCPUWithMatrix.getmat());

        if (!inputBuffer) {
            inputBuffer.reset(new XGPU(b.dim()));
            outputBuffer.reset(new XGPU(x.dim()));
        }

        inputBuffer->copyFrom(b);
        // TODO: [perf] do we need to copy x here?
        outputBuffer->copyFrom(x);

        underlyingSolver.apply(*outputBuffer, *inputBuffer, reduction, res);

        // TODO: [perf] do we need to copy b here?
        inputBuffer->copyTo(b);
        outputBuffer->copyTo(x);
    }
    virtual void apply (X& x, X& b, Dune::InverseOperatorResult& res) override {
        // TODO: [perf] Do we need to update the matrix every time? Probably yes
        matrix.updateNonzeroValues(opOnCPUWithMatrix.getmat());

        if (!inputBuffer) {
            inputBuffer.reset(new XGPU(b.dim()));
            outputBuffer.reset(new XGPU(x.dim()));
        }

        inputBuffer->copyFrom(b);
        // TODO: [perf] do we need to copy x here?
        outputBuffer->copyFrom(x);

        underlyingSolver.apply(*outputBuffer, *inputBuffer, res);

        // TODO: [perf] do we need to copy b here?
        inputBuffer->copyTo(b);
        outputBuffer->copyTo(x);
    }

private:
    Operator& opOnCPUWithMatrix;
    CuSparseMatrix<real_type> matrix;

    UnderlyingSolver<XGPU> underlyingSolver;

    // TODO: Use a std::forward
    UnderlyingSolver<XGPU>
    constructSolver(std::shared_ptr<Dune::Preconditioner<X, X>> prec, scalar_real_type reduction, int maxit, int verbose)
    {
        auto precAsAdapter = std::dynamic_pointer_cast<PreconditionerAdapter<X, X>>(prec);
        if (!precAsAdapter) {
            OPM_THROW(std::invalid_argument,
                      "The preconditioner needs to be a CUDA preconditioner wrapped in a "
                      "Opm::cuistl::PreconditionerAdapter (eg. CuILU0).");
        }
        auto preconditionerOnGPU = precAsAdapter->getUnderlyingPreconditioner();
        auto matrixOperator = std::make_shared<Dune::MatrixAdapter<CuSparseMatrix<real_type>, XGPU, XGPU>>(matrix);
        auto scalarProduct = std::make_shared<Dune::ScalarProduct<XGPU>>();

        return UnderlyingSolver<XGPU>(matrixOperator, scalarProduct, preconditionerOnGPU, reduction, maxit, verbose);
    }

    std::unique_ptr<XGPU> inputBuffer;
    std::unique_ptr<XGPU> outputBuffer;
};
} // namespace Opm::cuistl

#endif