#include "matrix.hpp"

double &Matrix::operator()(std::size_t row, std::size_t col) &
{
    if (m_layout == Layout::ColumnMajor)
        return m_data[col * m_rows + row];

    return m_data[row * m_cols + col];
}

const double &Matrix::operator()(std::size_t row, std::size_t col) const &
{
    if (m_layout == Layout::ColumnMajor)
        return m_data[col * m_rows + row];

    return m_data[row * m_cols + col];
}

MatrixSlice<double> Matrix::operator()(std::size_t row, std::nullopt_t) &
{
    if (m_cols == 0)
        return MatrixSlice<double>(nullptr, 0, 1, SliceAxis::Row);

    if (m_layout == Layout::RowMajor)
        return MatrixSlice<double>(&m_data[row * m_cols], m_cols, 1, SliceAxis::Row);

    return MatrixSlice<double>(&m_data[row], m_cols, m_rows, SliceAxis::Row);
}

MatrixSlice<const double> Matrix::operator()(std::size_t row, std::nullopt_t) const &
{
    if (m_cols == 0)
        return MatrixSlice<const double>(nullptr, 0, 1, SliceAxis::Row);

    if (m_layout == Layout::RowMajor)
        return MatrixSlice<const double>(&m_data[row * m_cols], m_cols, 1, SliceAxis::Row);

    return MatrixSlice<const double>(&m_data[row], m_cols, m_rows, SliceAxis::Row);
}

MatrixSlice<double> Matrix::operator()(std::nullopt_t, std::size_t col) &
{
    if (m_rows == 0)
        return MatrixSlice<double>(nullptr, 0, 1, SliceAxis::Column);

    if (m_layout == Layout::RowMajor)
        return MatrixSlice<double>(&m_data[col], m_rows, m_cols, SliceAxis::Column);

    return MatrixSlice<double>(&m_data[col * m_rows], m_rows, 1, SliceAxis::Column);
}

MatrixSlice<const double> Matrix::operator()(std::nullopt_t, std::size_t col) const &
{
    if (m_rows == 0)
        return MatrixSlice<const double>(nullptr, 0, 1, SliceAxis::Column);

    if (m_layout == Layout::RowMajor)
        return MatrixSlice<const double>(&m_data[col], m_rows, m_cols, SliceAxis::Column);

    return MatrixSlice<const double>(&m_data[col * m_rows], m_rows, 1, SliceAxis::Column);
}