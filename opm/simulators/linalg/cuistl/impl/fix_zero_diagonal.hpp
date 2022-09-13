#ifndef OPM_FIXZERODIAGONAL_HEADER_INCLUDED
#define OPM_FIXZERODIAGONAL_HEADER_INCLUDED

#include <limits>
#include <vector>


namespace Opm::cuistl::impl
{
template <class Matrix>
std::vector<typename Matrix::field_type>
fixZeroDiagonal(const Matrix& matrix,
                const typename Matrix::field_type replacementValue
                = std::numeric_limits<typename Matrix::field_type>::epsilon())
{
    using field_type = typename Matrix::field_type;
    std::vector<field_type> nonZeroes(matrix.nonzeroes() * Matrix::block_type::cols * Matrix::block_type::cols, 0.0);

    const auto dataPointer = static_cast<const field_type*>(&(matrix[0][0][0][0]));
    std::copy(dataPointer, dataPointer + nonZeroes.size(), nonZeroes.begin());

    // TODO: Is there a neater way of accessing the underlying CRS structure?
    size_t currentNonzeroPointer = 0u;
    for (auto row = matrix.begin(); row != matrix.end(); ++row) {
        for (auto column = row->begin(); column != row->end(); ++column) {
            if (column.index() == row.index()) {
                for (int component = 0; component < Matrix::block_type::cols; ++component) {
                    const auto index = currentNonzeroPointer + Matrix::block_type::cols * component + component;
                    if (nonZeroes[index] == 0) {
                        nonZeroes[index] = replacementValue;
                    }
                }
            }
            currentNonzeroPointer += 1;
        }
    }

    return nonZeroes;
}

template <class Matrix>
Matrix
makeMatrixWithNonzeroDiagonal(const Matrix& matrix,
                              const typename Matrix::field_type replacementValue
                              = std::numeric_limits<typename Matrix::field_type>::epsilon())
{
    auto newMatrix = matrix;
    // TODO: Is this fast enough?
    for (int row = 0; row < newMatrix.N(); ++row) {
        for (int component = 0; component < Matrix::block_type::cols; ++component) {
            if (newMatrix[row][row][component][component] == 0) {
                newMatrix[row][row][component][component] = replacementValue;
            }
        }
    }

    return newMatrix;
}
} // namespace Opm::cuistl::impl

#endif