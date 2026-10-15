#pragma once

#include <vector>
#include <span>
#include <stdexcept>

enum class Layout
{
    RowMajor,
    ColumnMajor
};

class Matrix
{
private:
    std::vector<double> m_data;

    std::size_t m_rows;
    std::size_t m_cols;

    Layout m_layout;

public:
    Matrix(size_t m = 0, size_t n = 0, Layout layout = Layout::RowMajor) : m_data(m * n, 0.), m_rows{m}, m_cols{n}, m_layout{layout} {};

    size_t get_rows() const { return m_rows; };
    size_t get_cols() const { return m_cols; };

    double &operator[](std::pair<int, int> coords)
    {
        auto [i, j] = coords;

        if (m_layout == Layout::RowMajor)
            return m_data[i * m_cols + j];

        return m_data[j * m_rows + i];
    }

    const double &operator[](std::pair<int, int> coords) const
    {
        auto [i, j] = coords;

        if (m_layout == Layout::RowMajor)
            return m_data[i * m_cols + j];

        return m_data[j * m_rows + i];
    }

    std::span<double> operator[](size_t idx)
    {
        return std::span<double>(m_data.data() + idx * m_cols, m_cols);
    }

    std::span<const double> operator[](size_t idx) const
    {
        return std::span<const double>(m_data.data() + idx * m_cols, m_cols);
    }

    std::span<double> at(size_t idx)
    {
        if (idx >= m_rows)
            throw std::invalid_argument("Matrix: Index out of range");
        return std::span<double>(m_data.data() + idx * m_cols, m_cols);
    }

    friend struct TransformedDataset;
};