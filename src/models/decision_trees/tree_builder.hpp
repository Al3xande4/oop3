#pragma once

#include <algorithm>
#include <cstddef>
#include <variant>

#include "criteria/search_criteria.hpp"
#include "dataset/matrix.hpp"
#include "leaf/leaf_strategy.hpp"
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

    SplitFinder m_split_finder{};

public:
    using value_type = typename LeafStrat::value_type;

    TreeBuilder(
        Crit criteria,
        LeafStrat leaf_strat,
        size_t max_depth,
        size_t min_samples_split,
        size_t min_samples_leaf
    )
        : m_criteria(criteria), m_leaf_strategy(leaf_strat), m_max_depth(max_depth),
          m_min_samples_split(min_samples_split), m_min_samples_leaf(min_samples_leaf){};

    Node<value_type> build(
        const Matrix &features,
        std::span<const double> targets,
        std::span<std::size_t> object_ids,
        std::span<const std::size_t> feature_ids
    ) const;

private:
    Node<value_type> _build(
        const Matrix &features,
        std::span<const double> targets,
        std::span<std::size_t> object_ids,
        std::span<const std::size_t> feature_ids,
        size_t depth
    ) const;
};

template <SearchCriteria Crit, LeafStrategy LeafStrat>
Node<typename TreeBuilder<Crit, LeafStrat>::value_type> TreeBuilder<Crit, LeafStrat>::build(
    const Matrix &features,
    std::span<const double> targets,
    std::span<std::size_t> object_ids,
    std::span<const std::size_t> feature_ids
) const
{
    return _build(features, targets, object_ids, feature_ids, m_max_depth);
}

template <SearchCriteria Crit, LeafStrategy LeafStrat>
Node<typename TreeBuilder<Crit, LeafStrat>::value_type> TreeBuilder<Crit, LeafStrat>::_build(
    const Matrix &features,
    std::span<const double> targets,
    std::span<std::size_t> object_ids,
    std::span<const std::size_t> feature_ids,
    size_t depth
) const
{
    if (depth == 0)
        return Leaf<value_type>{m_leaf_strategy.build(targets, object_ids)};

    if (object_ids.size() < m_min_samples_split)
        return Leaf<value_type>{m_leaf_strategy.build(targets, object_ids)};

    bool all_same = true;
    for (size_t i = 1; i < object_ids.size(); ++i)
        if (std::abs(targets[object_ids[i]] - targets[object_ids[0]]) > 1e-9)
        {
            all_same = false;
            break;
        }
    if (all_same)
        return Leaf<value_type>{m_leaf_strategy.build(targets, object_ids)};

    std::optional<Split> split =
        m_split_finder.find_best(features, targets, object_ids, feature_ids, m_criteria, m_min_samples_leaf);

    if (!split)
        return Leaf<value_type>{m_leaf_strategy.build(targets, object_ids)};

    auto it = std::partition(
        object_ids.begin(),
        object_ids.end(),
        [&](size_t index) { return features(index, split.value().feature) <= split.value().threshold; }
    );

    size_t size = it - object_ids.begin();
    auto left = object_ids.subspan(0, size);
    auto right = object_ids.subspan(size);

    Node<value_type> left_node = _build(features, targets, left, feature_ids, depth - 1);
    Node<value_type> right_node = _build(features, targets, right, feature_ids, depth - 1);

    return std::make_unique<Branch<value_type>>(std::move(left_node), std::move(right_node), split.value());
}
