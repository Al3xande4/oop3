#include "feature_info.hpp"
#include "config.hpp"

#include <algorithm>
#include <cmath>
#include <ranges>
#include <stdexcept>
#include <set>

double mean(std::span<const double> values)
{
    if (values.size() == 0)
        throw std::invalid_argument("Mean of an empty array");

    double s = 0.;
    for (const auto &v : values)
        s += v;

    return s / values.size();
}

double variance(std::span<const double> values)
{
    if (values.size() == 0)
        throw std::invalid_argument("Variance of an empty array");

    double m = mean(values);
    double s = 0;
    for (const auto &v : values)
        s += (v - m) * (v - m);

    s /= values.size();
    return std::sqrt(s);
}

double quantile(std::span<const double> values, double q)
{
    if (!std::ranges::is_sorted(values))
        throw std::invalid_argument("Values must be sorted to compute quantile");

    if (values.size() == 0)
        throw std::invalid_argument("Quantile of an empty array");

    if (q < 0 || q > 1)
        throw std::invalid_argument("Qunatile must be in range from 0 to 1");

    size_t n = values.size();

    double pos = (n - 1) * q;
    size_t whole_part = (long long)pos;
    double fraction = pos - whole_part;

    return values[whole_part] + fraction * (values[std::min(whole_part + 1, values.size() - 1)] - values[whole_part]);
}

double median(std::span<const double> values)
{
    if (!std::ranges::is_sorted(values))
        throw std::invalid_argument("Values must be sorted to compute median");

    if (values.size() == 0)
        throw std::invalid_argument("Median of an empty array");

    size_t n = values.size();
    if (n % 2 == 1)
        return values[n / 2];

    return (values[n / 2] + values[n / 2 - 1]) / 2;
}

NumericFeatureInfo analyze_numeric_column(const NumericColumn &column)
{
    NumericFeatureInfo info;
    info.missing_count = 0;

    std::vector<double> values;
    for (const auto &elem : column)
    {
        if (!elem)
        {
            info.missing_count++;
            continue;
        }
        values.push_back(*elem);
    }
    auto [min, max] = std::ranges::minmax(values);
    info.min = min;
    info.max = max;

    info.mean = mean(values);
    info.variance = variance(values);

    std::ranges::sort(values);
    info.median = median(values);
    info.q05 = quantile(values, .05);
    info.q25 = quantile(values, .25);
    info.q75 = quantile(values, .75);
    info.q95 = quantile(values, .95);

    double step = (max - min) / datasetconfig::histogram_intervals;
    for (double start = min; start < max; start += step)
    {
        auto lower = std::ranges::lower_bound(values, start);
        auto upper = std::ranges::upper_bound(values, start + step);
        info.histogram[std::make_pair(start, start + step)] = (size_t)(upper - lower);
        info.max_histogram_freq = std::max(info.max_histogram_freq, (size_t)(upper - lower));
    }

    return info;
};

CategoricalFeatureInfo analyze_categorical_column(const CategoricalColumn &column)
{
    CategoricalFeatureInfo info;

    std::set<std::string> categories;
    info.missing_count = 0;
    for (const auto &elem : column)
    {
        if (!elem)
        {
            info.missing_count++;
            continue;
        }

        if (categories.find(*elem) == categories.end())
            categories.insert(*elem);

        info.frequencies[*elem]++;
        info.max_freq = std::max(info.max_freq, info.frequencies[*elem]);
    }
    for (auto &category : categories)
        info.categories.push_back(std::move(category));

    return info;
}
