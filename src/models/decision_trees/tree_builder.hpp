#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>

#include "criteria/search_criteria.hpp"
#include "dataset/matrix.hpp"
#include "leaf/leaf_strategy.hpp"
#include "models/decision_trees/max_feature.hpp"
#include "models/decision_trees/split.hpp"
#include "models/decision_trees/split_finder.hpp"
#include "tree_node.hpp"

template <SearchCriteria Crit, LeafStrategy LeafStrat>
class TreeBuilder
{
private:
    Crit m_criteria;
    LeafStrat m_leaf_strategy;
    std::optional<size_t> m_max_depth;
    size_t m_min_samples_split;
    size_t m_min_samples_leaf;

    MaxFeature m_max_feature;
    std::uint32_t m_random_state;
    std::mt19937 m_gen;

    SplitFinder m_split_finder{};

public:
    using value_type = typename LeafStrat::value_type;

    TreeBuilder(
        Crit criteria,
        LeafStrat leaf_strat,
        std::optional<size_t> max_depth,
        size_t min_samples_split,
        size_t min_samples_leaf,
        MaxFeature max_feature = MaxFeaturesMode::All,
        std::uint32_t random_state = 0
    )
        : m_criteria(criteria), m_leaf_strategy(leaf_strat), m_max_depth(max_depth),
          m_min_samples_split(min_samples_split), m_min_samples_leaf(min_samples_leaf), m_max_feature(max_feature),
          m_random_state(random_state), m_gen(random_state){};

    Node<value_type> build(
        const Matrix &features,
        std::span<const double> targets,
        std::span<std::size_t> object_ids,
        std::span<const std::size_t> feature_ids,
        std::span<const size_t> counts
    );

    Node<value_type> build(
        const Matrix &features,
        std::span<const double> targets,
        std::span<std::size_t> object_ids,
        std::span<const std::size_t> feature_ids
    )
    {
        std::vector<size_t> default_counts;
        default_counts.assign(features.get_rows(), 1);
        return build(features, targets, object_ids, feature_ids, default_counts);
    }

private:
    struct BuildContext
    {
        TreeBuilder &builder;
        const Matrix &features;
        std::span<const double> targets;
        std::span<const size_t> counts;
        std::span<const size_t> feature_ids;
        std::size_t q;

        Node<value_type> build_node(std::span<std::size_t> object_ids, size_t depth = 0);

        bool should_stop(std::span<const std::size_t> object_ids, size_t depth) const;
        std::vector<size_t> select_features();
        std::optional<Split> find_best_split(
            std::span<const std::size_t> object_ids,
            std::span<const size_t> candidate_features
        );
        std::pair<std::span<size_t>, std::span<size_t>> partition(std::span<size_t> object_ids, const Split &split);
        Leaf<value_type> make_leaf(std::span<const std::size_t> object_ids);
    };
};

template <SearchCriteria Crit, LeafStrategy LeafStrat>
Node<typename TreeBuilder<Crit, LeafStrat>::value_type> TreeBuilder<Crit, LeafStrat>::build(
    const Matrix &features,
    std::span<const double> targets,
    std::span<std::size_t> object_ids,
    std::span<const std::size_t> feature_ids,
    std::span<const size_t> counts
)
{
    BuildContext ctx{
        .builder = *this,
        .features = features,
        .targets = targets,
        .counts = counts,
        .feature_ids = feature_ids,
        .q = m_max_feature.calculate(feature_ids.size())
    };

    return ctx.build_node(object_ids);
}

template <SearchCriteria Crit, LeafStrategy LeafStrat>
bool TreeBuilder<Crit, LeafStrat>::BuildContext::should_stop(std::span<const std::size_t> object_ids, size_t depth)
    const
{
    if (builder.m_max_depth && builder.m_max_depth.value() <= depth)
        return true;

    if (object_ids.size() < builder.m_min_samples_split)
        return true;

    double first = targets[object_ids[0]];
    return std::ranges::all_of(object_ids, [&](size_t id) { return std::abs(targets[id] - first) < 1e-9; });
}

template <SearchCriteria Crit, LeafStrategy LeafStrat>
std::vector<size_t> TreeBuilder<Crit, LeafStrat>::BuildContext::select_features()
{
    std::vector<size_t> selected_features(feature_ids.begin(), feature_ids.end());
    if (q < feature_ids.size())
        std::ranges::shuffle(selected_features, builder.m_gen);

    return selected_features;
}

template <SearchCriteria Crit, LeafStrategy LeafStrat>
std::optional<Split> TreeBuilder<Crit, LeafStrat>::BuildContext::find_best_split(
    std::span<const size_t> object_ids,
    std::span<const size_t> candidate_features
)
{
    std::optional<Split> best = std::nullopt;

    size_t budget = q;
    for (size_t feature_id : candidate_features)
    {
        if (budget == 0)
            break;

        FeatureSplitResult split_result = builder.m_split_finder.find_for_feature(
            features, targets, object_ids, feature_id, builder.m_criteria, builder.m_min_samples_leaf, counts
        );
        if (!split_result.is_constant)
            budget--;
        if (split_result.split && (!best || split_result.split->score > best->score))
            best = split_result.split;
    }

    return best;
}

template <SearchCriteria Crit, LeafStrategy LeafStrat>
std::pair<std::span<size_t>, std::span<size_t>> TreeBuilder<Crit, LeafStrat>::BuildContext::partition(
    std::span<size_t> object_ids,
    const Split &split
)
{
    auto mid = std::partition(
        object_ids.begin(), object_ids.end(), [&](size_t id) { return features(id, split.feature) <= split.threshold; }
    );
    size_t left_size = std::distance(object_ids.begin(), mid);

    auto left = object_ids.subspan(0, left_size);
    auto right = object_ids.subspan(left_size);

    return {left, right};
}

template <SearchCriteria Crit, LeafStrategy LeafStrat>
Leaf<typename LeafStrat::value_type> TreeBuilder<Crit, LeafStrat>::BuildContext::make_leaf(
    std::span<const std::size_t> object_ids
)
{
    return Leaf<value_type>(builder.m_leaf_strategy.build(targets, object_ids, counts));
}

template <SearchCriteria Crit, LeafStrategy LeafStrat>
Node<typename TreeBuilder<Crit, LeafStrat>::value_type> TreeBuilder<Crit, LeafStrat>::BuildContext::build_node(
    std::span<std::size_t> object_ids,
    size_t depth
)
{
    if (should_stop(object_ids, depth))
        return make_leaf(object_ids);

    auto candidate_features = select_features();

    std::optional<Split> split = find_best_split(object_ids, candidate_features);

    if (!split)
        return make_leaf(object_ids);

    auto [left, right] = partition(object_ids, split.value());

    Node<value_type> left_node = build_node(left, depth + 1);
    Node<value_type> right_node = build_node(right, depth + 1);

    return std::make_unique<Branch<value_type>>(std::move(left_node), std::move(right_node), split.value());
}