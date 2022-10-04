#ifndef OPM_CUISTL_AMGX_SOLVER_HEADER
#define OPM_CUISTL_AMGX_SOLVER_HEADER
#include "amgx_c.h"
#include <dune/istl/solver.hh>
#include <opm/common/OpmLog/OpmLog.hpp>
#include <string>
namespace
{
void
print_callback(const char* msg, int length)
{
    Opm::OpmLog::info(std::string("AMGX message:\n") + std::string(msg, length) + std::string("\n------------------------------------"));
}
} // namespace

#define OPM_AMGX_SAFE_CALL(x)                                                                                              \
    do {                                                                                                               \
        auto amgxSafeCallError = x;                                                                                    \
        if (amgxSafeCallError != AMGX_RC_OK) {                                                                         \
            std::vector<char> buffer(4096, '\0');                                                                      \
            AMGX_get_error_string(amgxSafeCallError, buffer.data(), buffer.size());                                    \
            OPM_THROW(std::runtime_error,                                                                              \
                      "AMGX error in expression " << #x << ". Error message was " << std::string(buffer.data()));      \
        }                                                                                                              \
    } while (false)

namespace Opm::cuistl
{
template <class Operator, class X, class Y>
class AMGXSolver : public Dune::IterativeSolver<X, Y>
{
public:
    using typename Dune::IterativeSolver<X, X>::domain_type;
    using typename Dune::IterativeSolver<X, X>::range_type;
    using typename Dune::IterativeSolver<X, X>::field_type;
    using typename Dune::IterativeSolver<X, X>::real_type;
    using typename Dune::IterativeSolver<X, X>::scalar_real_type;
    static constexpr auto blocksize = domain_type::block_type::dimension;

    AMGXSolver(Operator& op,
               std::shared_ptr<Dune::ScalarProduct<X>> sp,
               std::shared_ptr<Dune::Preconditioner<X, X>> prec,
               const std::string& amgxJsonFilename,
               scalar_real_type reduction,
               int maxit,
               int verbose)
        : Dune::IterativeSolver<X, Y>(op, *sp, *prec, reduction, maxit, verbose)
        , opOnCPU(op)
    {

        OPM_AMGX_SAFE_CALL(AMGX_initialize());
        OPM_AMGX_SAFE_CALL(AMGX_initialize_plugins());

        OPM_AMGX_SAFE_CALL(AMGX_register_print_callback(&print_callback));
        OPM_AMGX_SAFE_CALL(AMGX_install_signal_handler());
        OPM_AMGX_SAFE_CALL(AMGX_config_create_from_file(&cfg, amgxJsonFilename.c_str()));
        OPM_AMGX_SAFE_CALL(AMGX_resources_create_simple(&rsrc, cfg));

        OPM_AMGX_SAFE_CALL(AMGX_matrix_create(&A, rsrc, mode));
        OPM_AMGX_SAFE_CALL(AMGX_vector_create(&x, rsrc, mode));
        OPM_AMGX_SAFE_CALL(AMGX_vector_create(&b, rsrc, mode));
        OPM_AMGX_SAFE_CALL(AMGX_solver_create(&solver, rsrc, mode, cfg));
        updateMatrix();
        OPM_AMGX_SAFE_CALL(AMGX_solver_setup(solver, A));
    }

    // don't shadow four-argument version of apply defined in the base class
    using Dune::IterativeSolver<X, X>::apply;

    /*!
       \brief Apply inverse operator.

       \copydoc InverseOperator::apply(X&,Y&,InverseOperatorResult&)

       \note Currently, the BiCGSTABSolver aborts when it detects a breakdown.
     */
    virtual void apply(X& xCPU, X& bCPU, Dune::InverseOperatorResult& res) override
    {
        updateMatrix();
        OPM_AMGX_SAFE_CALL(AMGX_solver_setup(solver, A));
        OPM_AMGX_SAFE_CALL(AMGX_vector_upload(b, bCPU.N(), blocksize, static_cast<void*>(&bCPU[0][0])));
        OPM_AMGX_SAFE_CALL(AMGX_vector_upload(x, xCPU.N(), blocksize, static_cast<void*>(&xCPU[0][0])));
        OPM_AMGX_SAFE_CALL(AMGX_solver_solve(solver, b, x));
        OPM_AMGX_SAFE_CALL(AMGX_solver_get_status(solver, &status));
        // TODO: [perf] probably don't need both of them copied back
        OPM_AMGX_SAFE_CALL(AMGX_vector_download(x, static_cast<void*>(&xCPU[0][0])));
        OPM_AMGX_SAFE_CALL(AMGX_vector_download(b, static_cast<void*>(&bCPU[0][0])));

        res.converged = status == AMGX_SOLVE_SUCCESS;
        OPM_AMGX_SAFE_CALL(AMGX_solver_get_iterations_number(solver, &res.iterations));
    }


private:
    void updateMatrix()
    {
        std::vector<int> columnIndices;
        std::vector<int> rowIndices;
        rowIndices.push_back(0);
        const auto matrix = opOnCPU.getmat();
        const auto numberOfRows = matrix.N();
        const auto numberOfNonzeroBlocks = matrix.nonzeroes();
        const auto numberOfNonzeroElements = blocksize * blocksize * numberOfNonzeroBlocks;

        std::vector<scalar_real_type> nonZeroElementsData;
        // TODO: [perf] Can we avoid building nonZeroElementsData?
        nonZeroElementsData.reserve(numberOfNonzeroElements);

        columnIndices.reserve(numberOfNonzeroBlocks);
        rowIndices.reserve(numberOfRows + 1);
        for (auto& row : matrix) {
            for (auto columnIterator = row.begin(); columnIterator != row.end(); ++columnIterator) {
                columnIndices.push_back(columnIterator.index());
                for (int c = 0; c < blocksize; ++c) {
                    for (int d = 0; d < blocksize; ++d) {
                        nonZeroElementsData.push_back((*columnIterator)[c][d]);
                    }
                }
            }
            rowIndices.push_back(columnIndices.size());
        }
        auto nonZeroElements = nonZeroElementsData.data();
        // Sanity check
        // h_rows and h_cols could be changed to 'unsigned int', but amgx expects 'int'
        OPM_ERROR_IF(size_t(rowIndices[matrix.N()]) != matrix.nonzeroes(),
                     "Error size of rows do not sum to number of nonzeroes in CuSparseMatrix.");
        OPM_ERROR_IF(rowIndices.size() != numberOfRows + 1, "Row indices do not match for CuSparseMatrix.");
        OPM_ERROR_IF(columnIndices.size() != numberOfNonzeroBlocks, "Column indices do not match for CuSparseMatrix.");

        OPM_AMGX_SAFE_CALL(AMGX_matrix_upload_all(A,
                                              numberOfRows,
                                              numberOfNonzeroBlocks,
                                              blocksize,
                                              blocksize,
                                              rowIndices.data(),
                                              columnIndices.data(),
                                              nonZeroElements,
                                              nullptr)); // TODO: Extract diagonal maybe?
    }

    void updateMatrixLinear()
    {
        std::vector<int> columnIndices;
        std::vector<int> rowIndices;
        rowIndices.push_back(0);
        const auto matrix = opOnCPU.getmat();
        const auto numberOfRows = matrix.N();
        const auto numberOfNonzeroBlocks = matrix.nonzeroes();
        const auto numberOfNonzeroElements = blocksize * blocksize * numberOfNonzeroBlocks;

        std::vector<scalar_real_type> nonZeroElementsData;
        // TODO: [perf] Can we avoid building nonZeroElementsData?
        nonZeroElementsData.reserve(numberOfNonzeroElements);

        columnIndices.reserve(numberOfNonzeroBlocks);
        rowIndices.reserve(numberOfRows + 1);
        for (auto& row : matrix) {
            for (auto columnIterator = row.begin(); columnIterator != row.end(); ++columnIterator) {
                columnIndices.push_back(columnIterator.index());
                for (int c = 0; c < blocksize; ++c) {
                    for (int d = 0; d < blocksize; ++d) {
                        nonZeroElementsData.push_back((*columnIterator)[c][d]);
                    }
                }
            }
            rowIndices.push_back(columnIndices.size());
        }
        auto nonZeroElements = nonZeroElementsData.data();
        // Sanity check
        // h_rows and h_cols could be changed to 'unsigned int', but amgx expects 'int'
        OPM_ERROR_IF(size_t(rowIndices[matrix.N()]) != matrix.nonzeroes(),
                     "Error size of rows do not sum to number of nonzeroes in CuSparseMatrix.");
        OPM_ERROR_IF(rowIndices.size() != numberOfRows + 1, "Row indices do not match for CuSparseMatrix.");
        OPM_ERROR_IF(columnIndices.size() != numberOfNonzeroBlocks, "Column indices do not match for CuSparseMatrix.");

        OPM_AMGX_SAFE_CALL(AMGX_matrix_upload_all(A,
                                                  numberOfRows,
                                                  numberOfNonzeroBlocks,
                                                  blocksize,
                                                  blocksize,
                                                  rowIndices.data(),
                                                  columnIndices.data(),
                                                  nonZeroElements,
                                                  nullptr)); // TODO: Extract diagonal maybe?
    }
    Operator& opOnCPU;
    AMGX_config_handle cfg;
    AMGX_resources_handle rsrc;
    AMGX_matrix_handle A;
    AMGX_vector_handle b, x;
    AMGX_solver_handle solver;
    // status handling
    AMGX_SOLVE_STATUS status;
    const AMGX_Mode mode = AMGX_mode_dDDI; // first d is device, second D is double for matrices, third D is double for vectors, I is (am?) the integer type (i32)
};
} // namespace Opm::cuistl
#endif
