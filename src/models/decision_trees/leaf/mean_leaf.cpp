#include "mean_leaf.hpp"

MeanLeaf::value_type MeanLeaf::build(
    std::span<const double> targets,
    std::span<const size_t> object_ids,
    std::span<const size_t> counts
) const
{
    double sum = 0.;
    double w = 0.;
    for (auto index : object_ids)
    {
        sum += counts[index] * targets[index];
        w += counts[index];
    }

    return sum / w;
}