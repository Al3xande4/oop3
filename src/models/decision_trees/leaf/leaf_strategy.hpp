#pragma once

#include <concepts>
#include <span>

template <typename S>
concept LeafStrategy = requires(
    S strategy,
    const S const_criteria,
    std::span<const double> targets,
    std::span<const size_t> objects_ids,
    std::span<const size_t> counts
) {
    typename S::value_type;
    { strategy.build(targets, objects_ids, counts) } -> std::same_as<typename S::value_type>;
};