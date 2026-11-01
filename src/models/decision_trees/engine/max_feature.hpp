#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <variant>

#include "models/api/errors.hpp"
#include "models/common/overloaded.hpp"

enum class MaxFeaturesMode
{
    All,
    Sqrt
};

struct MaxFeature
{
    std::variant<size_t, MaxFeaturesMode> value = MaxFeaturesMode::All;

    MaxFeature(MaxFeaturesMode mode) : value{mode} {}
    MaxFeature(size_t count) : value{count} {}

    size_t calculate(size_t total_features) const
    {
        return std::visit(
            Overloaded{
                [&](const MaxFeaturesMode &mode) -> size_t
                {
                    switch (mode)
                    {
                    case MaxFeaturesMode::All:
                        return total_features;
                    case MaxFeaturesMode::Sqrt:
                        return std::max<size_t>(1, std::floor(std::sqrt(total_features)));
                    default:
                        return total_features;
                    }
                },
                [&](size_t count) -> size_t
                {
                    if (count < 1)
                        throw HyperParameterError("MaxFeature must be greater than 0");
                    if (count > total_features)
                        throw HyperParameterError("MaxFeature is greater than total features");
                    return count;
                }
            },
            value
        );
    }
};