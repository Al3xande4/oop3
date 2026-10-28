#pragma once

#include <span>

class MeanLeaf
{
public:
    using value_type = double;

    value_type build(std::span<const double> targets, std::span<const size_t> object_ids) const;
};