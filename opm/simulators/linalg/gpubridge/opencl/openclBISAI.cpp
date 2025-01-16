/*
  Copyright 2022 Equinor ASA

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

#include <config.h>
#include <algorithm>

#include <opm/common/OpmLog/OpmLog.hpp>
#include <opm/common/ErrorMacros.hpp>
#include <dune/common/timer.hh>

#include <opm/simulators/linalg/gpubridge/GpuSolver.hpp>
#include <opm/simulators/linalg/gpubridge/opencl/opencl.hpp>
#include <opm/simulators/linalg/gpubridge/opencl/openclBILU0.hpp>
#include <opm/simulators/linalg/gpubridge/opencl/openclBISAI.hpp>
#include <opm/simulators/linalg/gpubridge/opencl/openclKernels.hpp>
#include <opm/simulators/linalg/gpubridge/Reorder.hpp>
#include <opm/simulators/linalg/gpubridge/opencl/ChowPatelIlu.hpp> // disable BISAI if ChowPatel is selected

#include <sstream>

namespace Opm::Accelerator {

using Opm::OpmLog;
using Dune::Timer;

template<class Scalar, size_t block_size>
openclBISAI<Scalar,block_size>::openclBISAI(bool opencl_ilu_parallel_, long long verbosity_)
    : Base(verbosity_)
{
#if CHOW_PATEL
    OPM_THROW(std::logic_error, "Error --linear-solver=isai cannot be used if ChowPatelIlu is used, probably defined by CMake\n");
#endif
    bilu0 = std::make_unique<openclBILU0<Scalar,block_size>>(opencl_ilu_parallel_, verbosity_);
}

template<class Scalar, size_t block_size>
void openclBISAI<Scalar,block_size>::
setOpencl(std::shared_ptr<cl::Context>& context_,
          std::shared_ptr<cl::CommandQueue>& queue_)
{
    context = context_;
    queue = queue_;

    bilu0->setOpencl(context, queue);
}

std::vector<long long>
buildCsrToCscOffsetMap(std::vector<long long> colPointers, std::vector<long long> rowIndices)
{
    std::vector<long long> aux(colPointers); // colPointers must be copied to this vector
    std::vector<long long> csrToCscOffsetMap(rowIndices.size()); // map must have the same size as the indices vector

    for(size_t row = 0; row < colPointers.size() - 1; row++){
        for(long long jj = colPointers[row]; jj < colPointers[row+1]; jj++){
            long long col = rowIndices[jj];
            long long dest = aux[col];
            csrToCscOffsetMap[dest] = jj;
            aux[col]++;
        }
    }

    return csrToCscOffsetMap;
}

template<class Scalar, size_t block_size>
bool openclBISAI<Scalar,block_size>::analyze_matrix(BlockedMatrix<Scalar>* mat)
{
    return analyze_matrix(mat, nullptr);
}

template<class Scalar, size_t block_size>
bool openclBISAI<Scalar,block_size>::
analyze_matrix(BlockedMatrix<Scalar>* mat, BlockedMatrix<Scalar>* jacMat)
{
    const size_t bs = block_size;
    auto *m = mat;

    if (jacMat) {
        m = jacMat;
    }

    this->N = m->Nb * bs;
    this->Nb = m->Nb;
    this->nnz = m->nnzbs * bs * bs;
    this->nnzb = m->nnzbs;

    if (jacMat) {
        return bilu0->analyze_matrix(mat, jacMat);
    } else {
        return bilu0->analyze_matrix(mat);
    }
}

template<class Scalar, size_t block_size>
void openclBISAI<Scalar,block_size>::buildLowerSubsystemsStructures()
{
    lower.subsystemPointers.assign(Nb + 1, 0);

    Dune::Timer t_buildLowerSubsystemsStructures;

    for (long long tcol = 0; tcol < Nb; tcol++) {
        long long frow = diagIndex[tcol] + 1;
        long long lrow = colPointers[tcol + 1];
        long long nx = lrow - frow;
        long long nv = 0;

        for (long long sweep = 0; sweep < nx - 1; sweep++) {
            for (long long xid = sweep + 1; xid < nx; xid++) {
                for( long long ptr = diagIndex[rowIndices[frow + sweep]] + 1; ptr < colPointers[rowIndices[frow + sweep + 1]]; ptr++) {
                    if(rowIndices[ptr] == rowIndices[frow + xid]){
                        lower.nzIndices.push_back(csrToCscOffsetMap[ptr]);
                        lower.knownRhsIndices.push_back(csrToCscOffsetMap[frow + sweep]);
                        lower.unknownRhsIndices.push_back(csrToCscOffsetMap[frow + xid]);
                        nv++;
                    }
                }
            }
        }

        lower.subsystemPointers[tcol + 1] = lower.subsystemPointers[tcol] + nv;
    }

    if (verbosity >= 4) {
        std::ostringstream out;
        out << "openclBISAI buildLowerSubsystemsStructures time: "
            << t_buildLowerSubsystemsStructures.stop() << " s";
        OpmLog::info(out.str());
    }
}

template<class Scalar, size_t block_size>
void openclBISAI<Scalar,block_size>::buildUpperSubsystemsStructures()
{
    upper.subsystemPointers.assign(Nb + 1, 0);

    Dune::Timer t_buildUpperSubsystemsStructures;

    for (long long tcol = 0; tcol < Nb; tcol++) {
        long long frow = colPointers[tcol];
        long long lrow = diagIndex[tcol];
        long long nx = lrow - frow + 1;
        long long nv = 0;

        for (long long sweep = 0; sweep < nx - 1; sweep++) {
            for (long long xid = 0; xid < nx; xid++) {
                for (long long ptr = colPointers[rowIndices[lrow - sweep]]; ptr < diagIndex[rowIndices[lrow - sweep]]; ptr++) {
                    if (rowIndices[ptr] == rowIndices[lrow - xid]) {
                        upper.nzIndices.push_back(csrToCscOffsetMap[ptr]);
                        upper.knownRhsIndices.push_back(csrToCscOffsetMap[lrow - sweep]);
                        upper.unknownRhsIndices.push_back(csrToCscOffsetMap[lrow - xid]);
                        nv++;
                    }
                }
            }
        }

        upper.subsystemPointers[tcol + 1] = upper.subsystemPointers[tcol] + nv;
    }

    if (verbosity >= 4) {
        std::ostringstream out;
        out << "openclBISAI buildUpperSubsystemsStructures time: "
            << t_buildUpperSubsystemsStructures.stop() << " s";
        OpmLog::info(out.str());
    }
}

template<class Scalar, size_t block_size>
bool openclBISAI<Scalar,block_size>::
create_preconditioner(BlockedMatrix<Scalar>* mat, BlockedMatrix<Scalar>* jacMat)
{
    const size_t bs = block_size;

    if (bs != 3) {
        OPM_THROW(std::logic_error, "Creation of ISAI preconditioner on GPU only supports block_size = 3");
    }

    Dune::Timer t_preconditioner;

    if (jacMat) {
        bilu0->create_preconditioner(mat, jacMat);
    } else {
        bilu0->create_preconditioner(mat);
    }

    std::call_once(initialize, [&]() {
        std::tie(colPointers, rowIndices, diagIndex) = bilu0->get_preconditioner_structure();

        csrToCscOffsetMap = buildCsrToCscOffsetMap(colPointers, rowIndices);
        buildLowerSubsystemsStructures();
        buildUpperSubsystemsStructures();

        d_colPointers = cl::Buffer(*context, CL_MEM_READ_WRITE,
                                   sizeof(long long) * colPointers.size());
        d_rowIndices = cl::Buffer(*context, CL_MEM_READ_WRITE,
                                  sizeof(long long) * rowIndices.size());
        d_csrToCscOffsetMap = cl::Buffer(*context, CL_MEM_READ_WRITE,
                                         sizeof(long long) * csrToCscOffsetMap.size());
        d_diagIndex = cl::Buffer(*context, CL_MEM_READ_WRITE,
                                 sizeof(long long) * diagIndex.size());
        d_invLvals = cl::Buffer(*context, CL_MEM_READ_WRITE,
                                sizeof(Scalar) * nnzb * bs * bs);
        d_invUvals = cl::Buffer(*context, CL_MEM_READ_WRITE,
                                sizeof(Scalar) * nnzb * bs * bs);
        d_invL_x = cl::Buffer(*context, CL_MEM_READ_WRITE,
                              sizeof(Scalar) * Nb * bs);
        d_lower.subsystemPointers = cl::Buffer(*context, CL_MEM_READ_WRITE,
                                               sizeof(long long) * lower.subsystemPointers.size());
        d_upper.subsystemPointers = cl::Buffer(*context, CL_MEM_READ_WRITE,
                                               sizeof(long long) * upper.subsystemPointers.size());

        if (!lower.nzIndices.empty()) { // knownRhsIndices and unknownRhsIndices will also be empty if nzIndices is empty
            d_lower.nzIndices = cl::Buffer(*context, CL_MEM_READ_WRITE,
                                           sizeof(long long) * lower.nzIndices.size());
            d_lower.knownRhsIndices = cl::Buffer(*context, CL_MEM_READ_WRITE,
                                                 sizeof(long long) * lower.knownRhsIndices.size());
            d_lower.unknownRhsIndices = cl::Buffer(*context, CL_MEM_READ_WRITE,
                                                   sizeof(long long) * lower.unknownRhsIndices.size());
        }

        if (!upper.nzIndices.empty()) { // knownRhsIndices and unknownRhsIndices will also be empty if nzIndices is empty
            d_upper.nzIndices = cl::Buffer(*context, CL_MEM_READ_WRITE,
                                           sizeof(long long) * upper.nzIndices.size());
            d_upper.knownRhsIndices = cl::Buffer(*context, CL_MEM_READ_WRITE,
                                                 sizeof(long long) * upper.knownRhsIndices.size());
            d_upper.unknownRhsIndices = cl::Buffer(*context, CL_MEM_READ_WRITE,
                                                   sizeof(long long) * upper.unknownRhsIndices.size());
        }

        events.resize(6);
        err = queue->enqueueWriteBuffer(d_colPointers, CL_FALSE, 0,
                                        colPointers.size() * sizeof(long long),
                                        colPointers.data(), nullptr, &events[0]);
        err |= queue->enqueueWriteBuffer(d_rowIndices, CL_FALSE, 0,
                                         rowIndices.size() * sizeof(long long),
                                         rowIndices.data(), nullptr, &events[1]);
        err |= queue->enqueueWriteBuffer(d_csrToCscOffsetMap, CL_FALSE, 0,
                                         csrToCscOffsetMap.size() * sizeof(long long),
                                         csrToCscOffsetMap.data(), nullptr, &events[2]);
        err |= queue->enqueueWriteBuffer(d_diagIndex, CL_FALSE, 0,
                                         diagIndex.size() * sizeof(long long),
                                         diagIndex.data(), nullptr, &events[3]);
        err |= queue->enqueueWriteBuffer(d_lower.subsystemPointers, CL_FALSE, 0,
                                         sizeof(long long) * lower.subsystemPointers.size(),
                                         lower.subsystemPointers.data(), nullptr, &events[4]);
        err |= queue->enqueueWriteBuffer(d_upper.subsystemPointers, CL_FALSE, 0,
                                         sizeof(long long) * upper.subsystemPointers.size(),
                                         upper.subsystemPointers.data(), nullptr, &events[5]);

        if (!lower.nzIndices.empty()) {
            events.resize(events.size() + 3);
            err |= queue->enqueueWriteBuffer(d_lower.nzIndices, CL_FALSE, 0,
                                             sizeof(long long) * lower.nzIndices.size(),
                                             lower.nzIndices.data(), nullptr,
                                             &events[events.size() - 3]);
            err |= queue->enqueueWriteBuffer(d_lower.knownRhsIndices, CL_FALSE, 0,
                                             sizeof(long long) * lower.knownRhsIndices.size(),
                                             lower.knownRhsIndices.data(), nullptr,
                                             &events[events.size() - 2]);
            err |= queue->enqueueWriteBuffer(d_lower.unknownRhsIndices, CL_FALSE, 0,
                                             sizeof(long long) * lower.unknownRhsIndices.size(),
                                             lower.unknownRhsIndices.data(), nullptr,
                                             &events[events.size() - 1]);
        }

        if (!upper.nzIndices.empty()) {
            events.resize(events.size() + 3);
            err |= queue->enqueueWriteBuffer(d_upper.nzIndices, CL_FALSE,
                                             0, sizeof(long long) * upper.nzIndices.size(),
                                             upper.nzIndices.data(), nullptr,
                                             &events[events.size() - 3]);
            err |= queue->enqueueWriteBuffer(d_upper.knownRhsIndices, CL_FALSE, 0,
                                             sizeof(long long) * upper.knownRhsIndices.size(),
                                             upper.knownRhsIndices.data(), nullptr,
                                             &events[events.size() - 2]);
            err |= queue->enqueueWriteBuffer(d_upper.unknownRhsIndices, CL_FALSE, 0,
                                             sizeof(long long) * upper.unknownRhsIndices.size(),
                                             upper.unknownRhsIndices.data(), nullptr,
                                             &events[events.size() - 1]);
        }

        cl::WaitForEvents(events);
        events.clear();

        if (err != CL_SUCCESS) {
            // enqueueWriteBuffer is C and does not throw exceptions like C++ OpenCL
            OPM_THROW(std::logic_error, "openclBISAI OpenCL enqueueWriteBuffer error");
        }
    });

    std::tie(d_LUvals, d_invDiagVals) = bilu0->get_preconditioner_data();

    events.resize(2);
    err = queue->enqueueFillBuffer(d_invLvals, 0, 0,
                                   sizeof(Scalar) * nnzb * bs * bs, nullptr, &events[0]);
    err |= queue->enqueueFillBuffer(d_invUvals, 0, 0,
                                    sizeof(Scalar) * nnzb * bs * bs, nullptr, &events[1]);
    cl::WaitForEvents(events);
    events.clear();

    OpenclKernels<Scalar>::isaiL(d_diagIndex, d_colPointers, d_csrToCscOffsetMap,
                                 d_lower.subsystemPointers, d_lower.nzIndices,
                                 d_lower.unknownRhsIndices, d_lower.knownRhsIndices,
                                 d_LUvals, d_invLvals, Nb);
    OpenclKernels<double>::isaiU(d_diagIndex, d_colPointers, d_rowIndices,
                                 d_csrToCscOffsetMap, d_upper.subsystemPointers,
                                 d_upper.nzIndices, d_upper.unknownRhsIndices,
                                 d_upper.knownRhsIndices, d_LUvals,
            d_invDiagVals, d_invUvals, Nb);

    if (verbosity >= 4) {
        std::ostringstream out;
        out << "openclBISAI createPreconditioner time: " << t_preconditioner.stop() << " s";
        OpmLog::info(out.str());
    }

    return true;
}

template<class Scalar, size_t block_size>
bool openclBISAI<Scalar,block_size>::
create_preconditioner(BlockedMatrix<Scalar>* mat)
{
    return create_preconditioner(mat, nullptr);
}

template<class Scalar, size_t block_size>
void openclBISAI<Scalar,block_size>::apply(const cl::Buffer& x, cl::Buffer& y)
{
    const size_t bs = block_size;

    OpenclKernels<Scalar>::spmv(d_invLvals, d_rowIndices, d_colPointers,
                                x, d_invL_x, Nb, bs, true, true); // application of isaiL is a simple spmv with addition
                                                                  // (to compensate for the unitary diagonal that is not
                                                                  // included in isaiL, for simplicity)
    OpenclKernels<Scalar>::spmv(d_invUvals, d_rowIndices, d_colPointers,
                                d_invL_x, y, Nb, bs); // application of isaiU is a simple spmv
}

#define INSTANTIATE_TYPE(T)          \
    template class openclBISAI<T,1>; \
    template class openclBISAI<T,2>; \
    template class openclBISAI<T,3>; \
    template class openclBISAI<T,4>; \
    template class openclBISAI<T,5>; \
    template class openclBISAI<T,6>;

INSTANTIATE_TYPE(double)

#if FLOW_INSTANTIATE_FLOAT
INSTANTIATE_TYPE(float)
#endif

} // namespace Opm::Accelerator
