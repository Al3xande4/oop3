#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
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
    size_t m_max_depth;
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
        size_t max_depth,
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
    size_t q = m_max_feature.calculate(feature_ids.size());

    auto build_node =
        [&](auto &self, std::span<std::size_t> object_ids, std::span<const std::size_t> feature_ids, size_t depth
        ) -> Node<value_type>
    {
        if (depth == 0)
            return Leaf<value_type>{m_leaf_strategy.build(targets, object_ids, counts)};

        if (object_ids.size() < m_min_samples_split)
            return Leaf<value_type>{m_leaf_strategy.build(targets, object_ids, counts)};

        bool all_same = true;
        for (size_t i = 1; i < object_ids.size(); ++i)
            if (std::abs(targets[object_ids[i]] - targets[object_ids[0]]) > 1e-9)
            {
                all_same = false;
                break;
            }
        if (all_same)
            return Leaf<value_type>{m_leaf_strategy.build(targets, object_ids, counts)};

        std::vector<size_t> shuffled_feature_ids;
        if (q != feature_ids.size())
        {
            shuffled_feature_ids.resize(feature_ids.size());
            std::ranges::copy(feature_ids, shuffled_feature_ids.begin());
            std::ranges::shuffle(shuffled_feature_ids, m_gen);
            feature_ids = shuffled_feature_ids;
        }

        std::optional<Split> best = std::nullopt;

        size_t budget = q;
        for (size_t feature_id : feature_ids)
        {
            if (budget == 0)
                break;

            FeatureSplitResult split_result = m_split_finder.find_for_feature(
                features, targets, object_ids, feature_id, m_criteria, m_min_samples_leaf, counts
            );
            if (!split_result.is_constant)
                budget--;
            if (!best || (split_result.split && split_result.split.value().score > best.value().score))
                best = split_result.split;
        }

        if (!best)
            return Leaf<value_type>{m_leaf_strategy.build(targets, object_ids, counts)};

        auto it = std::ranges::partition(
            object_ids, [&](size_t index) { return features(index, best.value().feature) <= best.value().threshold; }
        );

        size_t size = it.begin() - object_ids.begin();
        auto left = object_ids.subspan(0, size);
        auto right = object_ids.subspan(size);

        Node<value_type> left_node = self(self, left, feature_ids, depth - 1);
        Node<value_type> right_node = self(self, right, feature_ids, depth - 1);

        return std::make_unique<Branch<value_type>>(std::move(left_node), std::move(right_node), best.value());
    };
    return build_node(build_node, object_ids, feature_ids, m_max_depth);
}
