#include <chrono>
#include <fstream>
#include <mpi.h>
#include <string>
#include <cuda_runtime.h>

#define OPM_TIME_TO_FILE(basename, numberOfNonzeroes) ::Opm::cuistl::TimeToFile timer##basename (#basename, numberOfNonzeroes)
#define OPM_CU_TIME_TO_FILE(basename, numberOfNonzeroes) ::Opm::cuistl::TimeToFile timer##basename (#basename, numberOfNonzeroes, true)

namespace
{
std::string
makeFilename(const std::string& basename)
{
    int worldSize;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    if (worldSize == 1) {
        return basename + "_runtimes.txt";
    }

    // Number of current process
    int processId;
    MPI_Comm_rank(MPI_COMM_WORLD, &processId);
    return basename + "_runtimes_" + std::to_string(processId) + ".txt";
}
} // namespace

namespace Opm::cuistl
{
struct TimeToFile {
    TimeToFile(const std::string& filename, int numberOfNonzeroes, bool doDeviceSynchronize=false)
        : filename(makeFilename(filename))
        , numberOfNonzeroes(numberOfNonzeroes)
        , start(std::chrono::high_resolution_clock::now())
        , doDeviceSynchronize(doDeviceSynchronize)
    {
    }

    ~TimeToFile()
    {
        if (doDeviceSynchronize) {
            cudaDeviceSynchronize();
        }
        const auto stop = std::chrono::high_resolution_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

        std::ofstream outfile(filename, std::ios::app);

        outfile << numberOfNonzeroes << " " << duration.count() << std::endl;
    }

private:
    const std::string filename;
    const int numberOfNonzeroes;
    const std::chrono::time_point<std::chrono::high_resolution_clock> start;
    const bool doDeviceSynchronize;
};
} // namespace Opm::cuistl