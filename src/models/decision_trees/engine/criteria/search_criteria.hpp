#pragma once

#include <concepts>
#include <span>

template <typename C>
concept SearchCriteria = requires(
    C criteria,
    const C const_criteria,
    std::span<const double> targets,
    std::span<const size_t> objects_ids,
    std::span<const size_t> counts,
    double target,
    double weight
) {
    { criteria.calculate_total(targets, objects_ids, counts) } -> std::same_as<void>;
    { criteria.init() } -> std::same_as<void>;
    { criteria.update(target, weight) } -> std::same_as<void>;
    { const_criteria.score() } -> std::convertible_to<double>;
    { const_criteria.total_count() } -> std::convertible_to<size_t>;
};