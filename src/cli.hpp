#pragma once

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>

struct LoadOptions
{
    bool help = false;
    std::optional<std::filesystem::path> dataset;
    std::optional<std::filesystem::path> dest;
    std::optional<std::size_t> histogram;
    std::optional<std::size_t> target;

    std::string impute = "mean";
    std::string scale = "robust";
    std::string encode = "onehot";

    std::string help_text;
};

struct KnnOptions
{
    std::optional<std::filesystem::path> dataset;
    std::optional<size_t> k;
    std::string distance = "euclid";
    std::string weight = "uniform";

    bool help = false;
    std::string help_text;
};

struct StumpOptions
{
    std::optional<std::filesystem::path> dataset;
    std::optional<size_t> min_samples_leaf;

    bool help = false;
    std::string help_text;
};

KnnOptions parse_knn_cli(int argc, char *argv[]);
LoadOptions parse_load_cli(int argc, char *argv[]);
StumpOptions parse_stump_cli(int argc, char *argv[]);
