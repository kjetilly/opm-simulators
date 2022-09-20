#include <config.h>

#define BOOST_TEST_MODULE TestCuVector

#include <boost/test/unit_test.hpp>
#include <opm/simulators/linalg/cuistl/CuVector.hpp>
#include <opm/simulators/linalg/cuistl/cuda_safe_call.hpp>
#include <dune/istl/bvector.hh>
#include <dune/common/fvector.hh>


BOOST_AUTO_TEST_CASE(TestConstructionSize)
{
    const int numberOfElements = 1234;
    auto vectorOnGPU = Opm::cuistl::CuVector<double>(numberOfElements);
    BOOST_CHECK_EQUAL(numberOfElements, vectorOnGPU.dim());
}

BOOST_AUTO_TEST_CASE(TestCopyFromHostConstructor)
{
    std::vector<double> data{{1,2,3,4,5,6,7}};
    auto vectorOnGPU = Opm::cuistl::CuVector<double>(data.data(), data.size());
    BOOST_CHECK_EQUAL(data.size(), vectorOnGPU.dim());
    std::vector<double> buffer(data.size(), 0.0);
    vectorOnGPU.copyToHost(buffer.data(), buffer.size());
    BOOST_CHECK_EQUAL_COLLECTIONS(buffer.begin(), buffer.end(), data.begin(), data.end());
}


BOOST_AUTO_TEST_CASE(TestCopyFromHostFunction)
{
    std::vector<double> data{{1,2,3,4,5,6,7}};
    auto vectorOnGPU = Opm::cuistl::CuVector<double>(data.size());
    BOOST_CHECK_EQUAL(data.size(), vectorOnGPU.dim());
    vectorOnGPU.copyFromHost(data.data(), data.size());
    std::vector<double> buffer(data.size(), 0.0);
    vectorOnGPU.copyToHost(buffer.data(), buffer.size());
    BOOST_CHECK_EQUAL_COLLECTIONS(buffer.begin(), buffer.end(), data.begin(), data.end());
}


BOOST_AUTO_TEST_CASE(TestCopyFromBvector)
{
    auto blockVector = Dune::BlockVector<Dune::FieldVector<double, 2>>{{{42,43},{44,45},{46, 47}}};
    auto vectorOnGPU = Opm::cuistl::CuVector<double>(blockVector.dim());
    vectorOnGPU.copyFrom(blockVector);
    std::vector<double> buffer(vectorOnGPU.dim());
    vectorOnGPU.copyToHost(buffer.data(), buffer.size());
    
    BOOST_CHECK_EQUAL_COLLECTIONS(buffer.begin(), buffer.end(), &blockVector[0][0], &blockVector[0][0]+blockVector.dim());
}

BOOST_AUTO_TEST_CASE(TestCopyToBvector)
{
    std::vector<double> data{{1,2,3,4,5,6,7,8,9}};
    auto blockVector = Dune::BlockVector<Dune::FieldVector<double, 3>>(3);
    auto vectorOnGPU = Opm::cuistl::CuVector<double>(data.data(), data.size());
    vectorOnGPU.copyTo(blockVector);

    
    BOOST_CHECK_EQUAL_COLLECTIONS(data.begin(), data.end(), &blockVector[0][0], &blockVector[0][0]+blockVector.dim());
}

BOOST_AUTO_TEST_CASE(TestDataPointer)
{
    std::vector<double> data{{1,2,3,4,5,6,7,8,9}};
    auto vectorOnGPU = Opm::cuistl::CuVector<double>(data.data(), data.size());

    std::vector<double> buffer(data.size(), 0.0);
    cudaMemcpy(buffer.data(), vectorOnGPU.data(), sizeof(double)*data.size(), cudaMemcpyDeviceToHost);
    BOOST_CHECK_EQUAL_COLLECTIONS(data.begin(), data.end(), buffer.begin(), buffer.end());
}

BOOST_AUTO_TEST_CASE(TestCopyScalarMultiply)
{
    std::vector<double> data{{1,2,3,4,5,6,7}};
    auto vectorOnGPU = Opm::cuistl::CuVector<double>(data.data(), data.size());
    BOOST_CHECK_EQUAL(data.size(), vectorOnGPU.dim());
    const double scalar = 42.25;
    vectorOnGPU *= scalar;
    std::vector<double> buffer(data.size(), 0.0);
    vectorOnGPU.copyToHost(buffer.data(), buffer.size());
    
    for (size_t i = 0; i < buffer.size(); ++i) {
        BOOST_CHECK_EQUAL(buffer[i], scalar * data[i]);
    }
}

BOOST_AUTO_TEST_CASE(TestTwoNorm)
{
    std::vector<double> data{{1,2,3,4,5,6,7}};
    auto vectorOnGPU = Opm::cuistl::CuVector<double>(data.data(), data.size());
    auto twoNorm = vectorOnGPU.two_norm();

    double correctAnswer = 0.0;
    for (double d : data) {
        correctAnswer += d*d;
    }
    correctAnswer = std::sqrt(correctAnswer);
    BOOST_CHECK_EQUAL(correctAnswer, twoNorm);
}

BOOST_AUTO_TEST_CASE(TestDot)
{
    std::vector<double> dataA{{1,2,3,4,5,6,7}};
    std::vector<double> dataB{{8,9,10,11,12,13,14}};
    auto vectorOnGPUA = Opm::cuistl::CuVector<double>(dataA.data(), dataA.size());
    auto vectorOnGPUB = Opm::cuistl::CuVector<double>(dataB.data(), dataB.size());
    auto dot = vectorOnGPUA.dot(vectorOnGPUB);

    double correctAnswer = 0.0;
    for (size_t i = 0; i < dataA.size(); ++i) {
        correctAnswer += dataA[i] * dataB[i];
    }
    correctAnswer = correctAnswer;
    BOOST_CHECK_EQUAL(correctAnswer, dot);
}