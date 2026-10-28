#pragma once

#include <concepts>
#include <span>

template <typename C>
concept SearchCriteria = requires(
    C criteria,
    const C const_criteria,
    std::span<const double> targets,
    std::span<const size_t> objects_ids,
    double target
) {
    { criteria.calculate_total(targets, objects_ids) } -> std::same_as<void>;
    { criteria.init() } -> std::same_as<void>;
    { criteria.update(target) } -> std::same_as<void>;
    { const_criteria.score() } -> std::convertible_to<double>;
};