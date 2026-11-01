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
        double c = counts.empty() ? 1.0 : counts[index];
        sum += c * targets[index];
        w += c;
    }
    return w > 0.0 ? (sum / w) : 0.0;
}