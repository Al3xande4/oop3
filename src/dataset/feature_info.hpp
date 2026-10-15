#pragma once

#include "column.hpp"
#include "config.hpp"

#include <span>
#include <map>
#include <unordered_map>

struct NumericFeatureInfo
{
    std::size_t missing_count;

    double min;
    double max;

    double mean;
    double variance;

    double median;

    double q05;
    double q25;
    double q75;
    double q95;

    std::map<std::pair<double, double>, size_t> histogram;
    size_t max_histogram_freq{1};
};

double mean(std::span<const double> values);
double variance(std::span<const double> values);
double quantile(std::span<const double> values, double q);
double median(std::span<const double> values);

NumericFeatureInfo analyze_numeric_column(const NumericColumn &column);

struct CategoricalFeatureInfo
{
    std::size_t missing_count;

    std::vector<std::string> categories;

    std::unordered_map<
        std::string,
        std::size_t>
        frequencies;

    std::size_t max_freq{1};
};

CategoricalFeatureInfo analyze_categorical_column(const CategoricalColumn &column);

using FeatureInfo =
    std::variant<NumericFeatureInfo, CategoricalFeatureInfo>;

struct DatasetInfo
{
    size_t total_count;
    std::vector<std::string> column_names;

    std::unordered_map<std::string, FeatureInfo> feature_info;
};
