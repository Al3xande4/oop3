#include "transformed.hpp"
#include <iostream>
#include <fstream>

void TransformedDataset::save(const std::filesystem::path &path) const
{
    std::ofstream ofs{path, std::ios::binary};
    if (!ofs.is_open())
        throw std::runtime_error("Saving directory doesn't exist.");

    ofs.write(reinterpret_cast<const char *>(&matrix.m_rows), sizeof(matrix.m_rows));
    ofs.write(reinterpret_cast<const char *>(&matrix.m_cols), sizeof(matrix.m_cols));
    ofs.write(reinterpret_cast<const char *>(&matrix.m_layout), sizeof(matrix.m_layout));

    ofs.write(reinterpret_cast<const char *>(matrix.m_data.data()), sizeof(matrix.m_data[0]) * matrix.m_data.size());

    size_t s = target_values.size();
    ofs.write(reinterpret_cast<const char *>(&s), sizeof(s));
    ofs.write(reinterpret_cast<const char *>(target_values.data()), sizeof(target_values[0]) * s);

    size_t tnl = target_name.size();
    ofs.write(reinterpret_cast<const char *>(&tnl), sizeof(tnl));
    ofs.write(reinterpret_cast<const char *>(target_name.data()), sizeof(char) * tnl);

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

    size_t s;
    ifs.read(reinterpret_cast<char *>(&s), sizeof(s));
    ds.target_values.resize(s);
    ifs.read(reinterpret_cast<char *>(ds.target_values.data()), sizeof(double) * s);

    size_t tnl;
    ifs.read(reinterpret_cast<char *>(&tnl), sizeof(tnl));
    ds.target_name.resize(tnl);
    ifs.read(reinterpret_cast<char *>(ds.target_name.data()), sizeof(char) * tnl);

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

std::pair<TransformedDataset, TransformedDataset> TransformedDataset::split(double s) const
{
    if (s < 0. || s > 1.)
        throw std::invalid_argument("Split argument must be in range [0, 1]");

    std::vector<size_t> indexes(target_values.size());
    for (size_t i = 0; i < indexes.size(); i++)
        indexes[i] = i;

    TransformedDataset train;
    train.feature_names = feature_names;
    train.target_name = target_name;

    size_t n = target_values.size() * s;
    train.matrix = Matrix(matrix, std::span<const size_t>(indexes.begin(), n));
    train.target_values.assign(target_values.begin(), target_values.begin() + n);

    TransformedDataset test;
    test.feature_names = feature_names;
    test.target_name = target_name;

    size_t k = target_values.size() - n;
    test.matrix = Matrix(matrix, std::span<const size_t>(indexes.begin() + n, k));
    test.target_values.assign(target_values.begin() + n, target_values.end());

    return std::make_pair(std::move(train), std::move(test));
}