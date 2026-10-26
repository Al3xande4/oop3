#pragma once

#include "matrix_slice.hpp"

#include <vector>
#include <span>
#include <stdexcept>
#include <optional>

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
    Matrix(const Matrix &other, std::span<const size_t> row_indexes) : m_rows{row_indexes.size()}, m_cols{other.get_cols()}, m_layout{other.m_layout}
    {
        m_data.resize(m_rows * m_cols);

        for (size_t i = 0; i < row_indexes.size(); i++)
            for (size_t col = 0; col < m_cols; col++)
                (*this)(i, col) = other(row_indexes[i], col);
    };

    size_t get_rows() const { return m_rows; };
    size_t get_cols() const { return m_cols; };

    double &operator()(std::size_t row, std::size_t col) &;
    const double &operator()(std::size_t row, std::size_t col) const &;

    double &operator()(std::size_t row, std::size_t col) && = delete;
    const double &operator()(std::size_t row, std::size_t col) const && = delete;

    MatrixSlice<double> operator()(std::size_t row, std::nullopt_t) &;
    MatrixSlice<const double> operator()(std::size_t row, std::nullopt_t) const &;

    MatrixSlice<double> operator()(std::size_t row, std::nullopt_t) && = delete;
    MatrixSlice<const double> operator()(std::size_t row, std::nullopt_t) const && = delete;

    MatrixSlice<double> operator()(std::nullopt_t, std::size_t col) &;
    MatrixSlice<const double> operator()(std::nullopt_t, std::size_t col) const &;

    MatrixSlice<double> operator()(std::nullopt_t, std::size_t col) && = delete;
    MatrixSlice<const double> operator()(std::nullopt_t, std::size_t col) const && = delete;

    std::span<double> operator[](size_t idx)
    {
        if (m_layout == Layout::ColumnMajor)
            throw std::invalid_argument("Matrix: operator[] doesn't work with column major");

        return std::span<double>(m_data.data() + idx * m_cols, m_cols);
    }

    std::span<const double> operator[](size_t idx) const
    {
        if (m_layout == Layout::ColumnMajor)
            throw std::invalid_argument("Matrix: operator[] doesn't work with column major");

        return std::span<const double>(m_data.data() + idx * m_cols, m_cols);
    }

    std::span<double> at(size_t idx)
    {
        if (idx >= m_rows)
            throw std::out_of_range("Matrix.at: Index out of range");

        if (m_layout == Layout::ColumnMajor)
            throw std::invalid_argument("Matrix: operator[] doesn't work with column major");

        return std::span<double>(m_data.data() + idx * m_cols, m_cols);
    }

    std::span<const double> at(size_t idx) const
    {
        if (idx >= m_rows)
            throw std::out_of_range("Matrix: Index out of range");

        if (m_layout == Layout::ColumnMajor)
            throw std::invalid_argument("Matrix: operator[] doesn't work with column major");

        return std::span<const double>(m_data.data() + idx * m_cols, m_cols);
    }

    friend struct TransformedDataset;
};
