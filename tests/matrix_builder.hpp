#pragma once

#include "dataset/matrix.hpp"

#include <vector>

class MatrixBuilder
{
private:
    size_t m_rows, m_cols;
    Layout m_layout;

    std::vector<std::vector<double>> m_data;

public:
    MatrixBuilder() : m_rows{0}, m_cols{0}, m_layout{Layout::RowMajor} {};

    const MatrixBuilder &set_rows(size_t, double fill = 0.);
    const MatrixBuilder &set_cols(size_t, double fill = 0.);
    const MatrixBuilder &set_layout(Layout);

    const MatrixBuilder &add_row(std::span<const double>);
    const MatrixBuilder &add_col(std::span<const double>);

    Matrix build() const;
};