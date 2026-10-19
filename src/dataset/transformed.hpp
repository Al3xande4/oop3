#pragma once

#include <filesystem>
#include <span>
#include <functional>
#include "matrix.hpp"

struct TransformedDataset
{
    Matrix matrix;

    std::vector<std::string> feature_names;

    std::string target_name;
    std::vector<double> target_values;

    void save(const std::filesystem::path &) const;
    static TransformedDataset load(const std::filesystem::path &);

    std::pair<TransformedDataset, TransformedDataset> split(double) const;
};