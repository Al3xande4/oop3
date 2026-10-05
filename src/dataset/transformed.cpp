#include "transformed.hpp"
#include <iostream>

void Matrix::save(const std::filesystem::path &path) const
{
    std::ofstream ofs{path, std::ios::binary};
    if (!ofs.is_open())
        throw std::runtime_error("Saving directory doesn't exist.");

    ofs.write(reinterpret_cast<const char *>(&m_rows), sizeof(m_rows));
    ofs.write(reinterpret_cast<const char *>(&m_cols), sizeof(m_cols));
    ofs.write(reinterpret_cast<const char *>(&m_layout), sizeof(m_layout));

    ofs.write(reinterpret_cast<const char *>(m_data.data()), sizeof(m_data[0]) * m_data.size());
}

void Matrix::save_txt(const std::filesystem::path &path) const
{
    std::ofstream ofs{path};
    if (!ofs.is_open())
        throw std::runtime_error("Saving directory doesn't exist.");

    ofs << m_rows << ' ' << m_cols << ' ' << (int)m_layout << '\n';

    for (const auto &elem : m_data)
        ofs << elem << ' ';
}

Matrix Matrix::load(const std::filesystem::path &path)
{
    Matrix matrix;

    std::ifstream ifs{path, std::ios::binary};
    if (!ifs.is_open())
        throw std::runtime_error("No such file " + std::string(path));

    ifs.read(reinterpret_cast<char *>(&matrix.m_rows), sizeof(matrix.m_rows));
    ifs.read(reinterpret_cast<char *>(&matrix.m_cols), sizeof(matrix.m_cols));
    ifs.read(reinterpret_cast<char *>(&matrix.m_layout), sizeof(matrix.m_layout));

    matrix.m_data.resize(matrix.m_cols * matrix.m_rows);
    ifs.read(reinterpret_cast<char *>(matrix.m_data.data()), sizeof(double) * matrix.m_data.size());

    return matrix;
}

void TransformedDataset::save(const std::filesystem::path &path) const
{
    std::ofstream ofs{path, std::ios::binary};
    if (!ofs.is_open())
        throw std::runtime_error("Saving directory doesn't exist.");

    ofs.write(reinterpret_cast<const char *>(&matrix.m_rows), sizeof(matrix.m_rows));
    ofs.write(reinterpret_cast<const char *>(&matrix.m_cols), sizeof(matrix.m_cols));
    ofs.write(reinterpret_cast<const char *>(&matrix.m_layout), sizeof(matrix.m_layout));

    ofs.write(reinterpret_cast<const char *>(matrix.m_data.data()), sizeof(matrix.m_data[0]) * matrix.m_data.size());

    size_t n = feature_names.size();
    ofs.write(reinterpret_cast<const char *>(&n), sizeof(n));
    for (const auto &s : feature_names)
    {
        size_t l = s.size();

        ofs.write(reinterpret_cast<const char *>(&l), sizeof(l));
        ofs.write(s.data(), l);
    }
}

TransformedDataset TransformedDataset::load(const std::filesystem::path &path)
{
    TransformedDataset ds;

    std::ifstream ifs{path, std::ios::binary};
    if (!ifs.is_open())
        throw std::runtime_error("No such file " + std::string(path));

    ifs.read(reinterpret_cast<char *>(&ds.matrix.m_rows), sizeof(ds.matrix.m_rows));
    ifs.read(reinterpret_cast<char *>(&ds.matrix.m_cols), sizeof(ds.matrix.m_cols));
    ifs.read(reinterpret_cast<char *>(&ds.matrix.m_layout), sizeof(ds.matrix.m_layout));

    ds.matrix.m_data.resize(ds.matrix.m_cols * ds.matrix.m_rows);
    ifs.read(reinterpret_cast<char *>(ds.matrix.m_data.data()), sizeof(double) * ds.matrix.m_data.size());

    size_t n;
    ifs.read(reinterpret_cast<char *>(&n), sizeof(n));
    ds.feature_names.resize(n);
    for (size_t i = 0; i < n; i++)
    {
        size_t l;

        ifs.read(reinterpret_cast<char *>(&l), sizeof(l));
        std::string s(l, 's');
        ifs.read(s.data(), l);
        ds.feature_names[i] = std::move(s);
    }

    return ds;
}