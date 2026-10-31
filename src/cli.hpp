#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>

class CliError : public std::runtime_error
{
public:
    CliError(const std::string &message) : std::runtime_error(message) {}
};

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

struct CartOptions
{
    std::optional<std::filesystem::path> dataset;
    std::optional<size_t> min_samples_leaf;
    std::optional<size_t> min_samples_split;
    std::optional<size_t> max_depth;

    bool help = false;
    std::string help_text;
};

struct ForestOptions
{
    std::optional<std::filesystem::path> dataset;
    std::optional<size_t> n_estimators;
    std::optional<bool> bootstrap = true;
    std::optional<size_t> max_samples;
    std::optional<std::string> max_features;
    std::optional<size_t> max_depth;
    std::optional<size_t> min_samples_leaf;
    std::optional<size_t> min_samples_split;
    std::optional<std::uint32_t> random_state;
    std::optional<size_t> n_jobs;

    bool help = false;
    std::string help_text;
};

KnnOptions parse_knn_cli(int argc, char *argv[]);
LoadOptions parse_load_cli(int argc, char *argv[]);
StumpOptions parse_stump_cli(int argc, char *argv[]);
CartOptions parse_cart_cli(int argc, char *argv[]);
ForestOptions parse_forest_cli(int argc, char *argv[]);
