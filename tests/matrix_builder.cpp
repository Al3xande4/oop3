#include "matrix_builder.hpp"

const MatrixBuilder &MatrixBuilder::set_rows(size_t rows, double fill)
{
    if (m_rows == rows)
        return *this;

    m_data.resize(rows);

    if (rows > m_rows)
        for (size_t i = m_rows; i < rows; i++)
            m_data[i] = std::vector(m_cols, fill);

    m_rows = rows;
    return *this;
}

const MatrixBuilder &MatrixBuilder::set_cols(size_t cols, double fill)
{
    if (m_cols == cols)
        return *this;

    for (size_t row = 0; row < m_rows; row++)
        m_data[row].resize(cols);

    if (cols > m_cols)
        for (size_t row = 0; row < m_rows; row++)
            for (size_t col = m_cols; col < cols; col++)
                m_data[row][col] = fill;

    m_cols = cols;
    return *this;
}

const MatrixBuilder &MatrixBuilder::set_layout(Layout layout)
{
    m_layout = layout;
    return *this;
}

const MatrixBuilder &MatrixBuilder::add_row(std::span<const double> row)
{
    if (row.size() != m_cols)
        throw std::invalid_argument("MatrixBuilder.add_row: row size mismatches current cols size");

    std::vector<double> vector_row(row.size());
    for (size_t i = 0; i < row.size(); i++)
        vector_row[i] = row[i];

    m_data.emplace_back(std::move(vector_row));
    m_rows++;
    return *this;
}
const MatrixBuilder &MatrixBuilder::add_col(std::span<const double> col)
{
    if (col.size() != m_rows)
        throw std::invalid_argument("MatrixBuilder.add_col: col size mismatches current rows size");

    for (size_t row = 0; row < m_rows; row++)
        m_data[row].push_back(col[row]);

    m_cols++;

    return *this;
}

Matrix MatrixBuilder::build() const
{
    Matrix matrix(m_rows, m_cols, m_layout);

    for (size_t row = 0; row < m_rows; row++)
        for (size_t col = 0; col < m_cols; col++)
            matrix[std::make_pair(row, col)] = m_data[row][col];

    return matrix;
}