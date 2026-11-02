#pragma once

#include <algorithm>
#include <numeric>
#include <optional>
#include <span>
#include <vector>

#include "criteria/search_criteria.hpp"
#include "dataset/matrix.hpp"
#include "models/api/errors.hpp"
#include "split.hpp"

struct FeatureSplitResult
{
    std::optional<Split> split;
    bool is_constant;
};

class SplitFinder
{
public:
    template <SearchCriteria Crit>
    std::optional<Split> find_best(
        const Matrix &features,
        std::span<const double> targets,
        std::span<const std::size_t> object_ids,
        std::span<const std::size_t> feature_ids,
        Crit criteria,
        std::size_t min_samples_leaf,
        std::span<const size_t> counts = {}
    ) const;

    template <SearchCriteria Crit>
    FeatureSplitResult find_for_feature(
        const Matrix &features,
        std::span<const double> targets,
        std::span<const std::size_t> object_ids,
        std::size_t feature_ids,
        Crit criteria,
        std::size_t min_samples_leaf,
        std::span<const size_t> counts = {}
    ) const;
};

template <SearchCriteria Crit>
FeatureSplitResult SplitFinder::find_for_feature(
    const Matrix &features,
    std::span<const double> targets,
    std::span<const std::size_t> object_ids,
    std::size_t feature_id,
    Crit criteria,
    std::size_t min_samples_leaf,
    std::span<const size_t> counts
) const
{
    if (object_ids.size() > targets.size() || object_ids.size() > features.get_rows())
        throw DimensionMismatch("SplitFinder.find_for_feature: objects and features mismatch sizes");

    if (!counts.empty() && features.get_rows() != counts.size())
        throw DimensionMismatch("SplitFinder.find_for_feature: counts and features mismatch sizes");

    std::optional<Split> result = std::nullopt;
    if (criteria.total_count() == 0)
        criteria.calculate_total(targets, object_ids, counts);

    size_t col = feature_id;

    std::vector<size_t> order(object_ids.size());
    for (size_t i = 0; i < object_ids.size(); i++)
        order[i] = object_ids[i];
    std::ranges::sort(order, [&](size_t a, size_t b) { return features(a, col) < features(b, col); });

    criteria.init();
    size_t count_left = 0, count_right = object_ids.size();

    bool is_constant = true;

    for (size_t p = 0; p + 1 < order.size(); p++)
    {
        size_t row = order[p];
        size_t next_row = order[p + 1];

        criteria.update(targets[row], counts.empty() ? 1 : counts[row]);
        count_left++;
        count_right--;

        if (features(row, col) == features(next_row, col))
            continue;

        is_constant = false;

        if (count_right < min_samples_leaf)
            break;

        if (count_left < min_samples_leaf)
            continue;

        double score = criteria.score();
        double t = std::midpoint(features(row, col), features(next_row, col));
        if (t == features(next_row, col))
            t = features(row, col);

        if (!result || score > result.value().score)
            result = {.feature = col, .threshold = t, .score = score};
    }

    return {result, is_constant};
}

template <SearchCriteria Crit>
std::optional<Split> SplitFinder::find_best(
    const Matrix &features,
    std::span<const double> targets,
    std::span<const std::size_t> object_ids,
    std::span<const std::size_t> feature_ids,
    Crit criteria,
    std::size_t min_samples_leaf,
    std::span<const size_t> counts
) const
{
    if (object_ids.size() > targets.size() || object_ids.size() > features.get_rows())
        throw DimensionMismatch("SplitFinder.find_best: objects and features mismatch sizes");

    if (!counts.empty() && features.get_rows() != counts.size())
        throw DimensionMismatch("SplitFinder.find_best: counts and features mismatch sizes");

    std::optional<Split> result = std::nullopt;

    std::vector<size_t> columns(feature_ids.size());
    for (size_t i = 0; i < feature_ids.size(); i++)
        columns[i] = feature_ids[i];
    std::ranges::sort(columns);

    criteria.calculate_total(targets, object_ids, counts);
    for (size_t col : columns)
    {
        auto split = find_for_feature(features, targets, object_ids, col, criteria, min_samples_leaf, counts).split;
        if (!result || (split && split.value().score > result.value().score))
            result = split;
    }

    return result;
}