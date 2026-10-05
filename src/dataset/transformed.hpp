#pragma once

#include <bits/stdc++.h>

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

    void save(const std::filesystem::path &) const;
    void save_txt(const std::filesystem::path &) const;

    static Matrix load(const std::filesystem::path &);

    friend struct TransformedDataset;
};

struct TransformedDataset
{
    Matrix matrix;

    std::vector<std::string> feature_names;

    void save(const std::filesystem::path &) const;

    static TransformedDataset load(const std::filesystem::path &);
};