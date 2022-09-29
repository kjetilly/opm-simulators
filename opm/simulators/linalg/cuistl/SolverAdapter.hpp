#ifndef OPM_SOLVERADAPTER_HEADER_INCLUDED
#define OPM_SOLVERADAPTER_HEADER_INCLUDED

#include "opm/simulators/linalg/matrixblock.hh"
#include <memory>

#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/operators.hh>
#include <dune/istl/schwarz.hh>
#include <dune/istl/solver.hh>
#include <opm/common/ErrorMacros.hpp>
#include <opm/simulators/linalg/cuistl/CuOwnerOverlapCopy.hpp>
#include <opm/simulators/linalg/cuistl/CuSparseMatrix.hpp>
#include <opm/simulators/linalg/cuistl/CuVector.hpp>
#include <opm/simulators/linalg/cuistl/PreconditionerAdapter.hpp>




namespace Opm::cuistl
{

namespace impl
{
    template <typename T>
    class has_communication
    {
        using yes_type = char;
        using no_type = long;
        template <typename U>
        static yes_type test(decltype(&U::getCommunication));
        template <typename U>
        static no_type test(...);

    public:
        static constexpr bool value = sizeof(test<T>(0)) == sizeof(yes_type);
    };

    template <typename T>
    class is_a_well_operator
    {
        using yes_type = char;
        using no_type = long;
        template <typename U>
        static yes_type test(decltype(&U::addWellPressureEquations));
        template <typename U>
        static no_type test(...);

    public:
        static constexpr bool value = sizeof(test<T>(0)) == sizeof(yes_type);
    };

} // namespace impl

template <class Operator, template <class> class UnderlyingSolver, class X>
class SolverAdapter : public Dune::IterativeSolver<X, X>
{
public:
    using typename Dune::IterativeSolver<X, X>::domain_type;
    using typename Dune::IterativeSolver<X, X>::range_type;
    using typename Dune::IterativeSolver<X, X>::field_type;
    using typename Dune::IterativeSolver<X, X>::real_type;
    using typename Dune::IterativeSolver<X, X>::scalar_real_type;
    static constexpr auto block_size = domain_type::block_type::dimension;
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

    virtual void apply(X& x, X& b, double reduction, Dune::InverseOperatorResult& res) override
    {
        // TODO: Can we do this without reimplementing the other function?
        // TODO: [perf] Do we need to update the matrix every time? Probably yes
        matrix.updateNonzeroValues(opOnCPUWithMatrix.getmat());

        if (!inputBuffer) {
            inputBuffer.reset(new XGPU(b.dim()));
            outputBuffer.reset(new XGPU(x.dim()));
        }

        inputBuffer->copyFromHost(b);
        // TODO: [perf] do we need to copy x here?
        outputBuffer->copyFromHost(x);

        underlyingSolver.apply(*outputBuffer, *inputBuffer, reduction, res);

        // TODO: [perf] do we need to copy b here?
        inputBuffer->copyToHost(b);
        outputBuffer->copyToHost(x);
    }
    virtual void apply(X& x, X& b, Dune::InverseOperatorResult& res) override
    {
        // TODO: [perf] Do we need to update the matrix every time? Probably yes
        matrix.updateNonzeroValues(opOnCPUWithMatrix.getmat());

        if (!inputBuffer) {
            inputBuffer.reset(new XGPU(b.dim()));
            outputBuffer.reset(new XGPU(x.dim()));
        }

        inputBuffer->copyFromHost(b);
        // TODO: [perf] do we need to copy x here?
        outputBuffer->copyFromHost(x);

        underlyingSolver.apply(*outputBuffer, *inputBuffer, res);

        // TODO: [perf] do we need to copy b here?
        inputBuffer->copyToHost(b);
        outputBuffer->copyToHost(x);
    }

private:
    Operator& opOnCPUWithMatrix;
    CuSparseMatrix<real_type> matrix;

    UnderlyingSolver<XGPU> underlyingSolver;


    // TODO: Use a std::forward
    UnderlyingSolver<XGPU> constructSolver(std::shared_ptr<Dune::Preconditioner<X, X>> prec,
                                           scalar_real_type reduction,
                                           int maxit,
                                           int verbose)
    {

        OPM_ERROR_IF(impl::is_a_well_operator<Operator>::value,
                     "Currently we only support operators of type MatrixAdapter in the CUDA solver. "
                     "Use --matrix-add-well-contributions=true. "
                     "Using WellModelMatrixAdapter with SolverAdapter won't work... well.");


        auto precAsAdapter = std::dynamic_pointer_cast<PreconditionerAdapter<X, X>>(prec);
        if (!precAsAdapter) {
            OPM_THROW(std::invalid_argument,
                      "The preconditioner needs to be a CUDA preconditioner wrapped in a "
                      "Opm::cuistl::PreconditionerAdapter (eg. CuILU0).");
        }


        auto preconditionerOnGPU = precAsAdapter->getUnderlyingPreconditioner();

        if constexpr (impl::has_communication<Operator>::value) {
            const auto& communication = opOnCPUWithMatrix.getCommunication();
            using CudaCommunication = CuOwnerOverlapCopy<real_type, block_size, decltype(communication)>;
            using SchwarzOperator
                = Dune::OverlappingSchwarzOperator<CuSparseMatrix<real_type>, XGPU, XGPU, CudaCommunication>;
            const auto& cudaCommunication = CudaCommunication::getInstance(communication);
            auto scalarProduct = std::make_shared<Dune::ParallelScalarProduct<XGPU, CudaCommunication>>(
                cudaCommunication, opOnCPUWithMatrix.category());
            auto overlappingCudaOperator = std::make_shared<SchwarzOperator>(matrix, cudaCommunication);

            return UnderlyingSolver<XGPU>(
                overlappingCudaOperator, scalarProduct, preconditionerOnGPU, reduction, maxit, verbose);
        } else {
            auto matrixOperator = std::make_shared<Dune::MatrixAdapter<CuSparseMatrix<real_type>, XGPU, XGPU>>(matrix);
            auto scalarProduct = std::make_shared<Dune::SeqScalarProduct<XGPU>>();
            return UnderlyingSolver<XGPU>(
                matrixOperator, scalarProduct, preconditionerOnGPU, reduction, maxit, verbose);
        }
    }

    std::unique_ptr<XGPU> inputBuffer;
    std::unique_ptr<XGPU> outputBuffer;
};
} // namespace Opm::cuistl

#endif
