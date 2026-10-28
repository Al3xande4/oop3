#pragma once

#include <concepts>
#include <span>

template <typename S>
concept LeafStrategy =
    requires(S strategy, const S const_criteria, std::span<const double> targets, std::span<const size_t> objects_ids) {
        typename S::value_type;
        { strategy.build(targets, objects_ids) } -> std::same_as<typename S::value_type>;
    };