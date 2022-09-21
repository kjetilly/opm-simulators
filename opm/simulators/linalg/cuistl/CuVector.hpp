/*
  Copyright SINTEF AS

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
#ifndef OPM_CUVECTOR_HEADER_INCLUDED
#define OPM_CUVECTOR_HEADER_INCLUDED
#include <exception>
#include <opm/common/ErrorMacros.hpp>
#include <opm/simulators/linalg/cuistl/CuBlasHandle.hpp>
#include <vector>

namespace Opm::cuistl
{

/*! \brief Simple vector class on the GPU.
 *
 */
template <typename T>
class CuVector
{
public:
    using field_type = T;
    using size_type = size_t;
    
    CuVector(const CuVector<T>& other);
    explicit CuVector(const std::vector<T>& data);
    CuVector& operator=(const CuVector<T>&);
    CuVector& operator=(T scalar);
    
    explicit CuVector(const int numberOfElements);
    CuVector(const T* dataOnHost, const int numberOfElements);
    virtual ~CuVector();

    T* data();
    const T* data() const;

    template <class VectorType>
    void copyFrom(const VectorType& vector)
    {
        if (numberOfElements != vector.dim()) {
            OPM_THROW(std::runtime_error,
                      "Given incompatible vector size. CuVector has size " + std::to_string(numberOfElements)
                          + ",\nhowever, " + typeid(VectorType).name() + " has N() = " + std::to_string(vector.N())
                          + ", and dim() = " + std::to_string(vector.dim()));
        }
        const auto dataPointer = static_cast<const T*>(&(vector[0][0]));
        copyFromHost(dataPointer, numberOfElements);
    }

    template <class VectorType>
    void copyTo(VectorType& vector) const
    {
        if (numberOfElements != vector.dim()) {
            OPM_THROW(std::runtime_error,
                      "Given incompatible vector size. CuVector has size " + std::to_string(numberOfElements)
                          + ",\nhowever, " + typeid(VectorType).name() + " has N() = " + std::to_string(vector.N())
                          + ", and dim() = " + std::to_string(vector.dim()));
        }
        const auto dataPointer = static_cast<T*>(&(vector[0][0]));
        copyToHost(dataPointer, numberOfElements);
    }
    void copyFromHost(const T* dataPointer, int numberOfElements);
    void copyToHost(T* dataPointer, int numberOfElements) const;
    void copyFromHost(const std::vector<T>& data);
    void copyToHost(std::vector<T>& data) const;

    CuVector<T>& operator*=(const T& scalar);

    CuVector<T>& axpy(T alpha, const CuVector<T>& y);
    CuVector<T>& operator+=(const CuVector<T>& other);
    CuVector<T>& operator-=(const CuVector<T>& other);
    T dot(const CuVector<T>& other) const;
    T two_norm() const;

    size_type dim() const;


private:
    T* dataOnDevice = nullptr;
    const int numberOfElements;
    CuBlasHandle& cuBlasHandle;
};

} // namespace Opm::cuistl
#endif
