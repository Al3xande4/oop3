#include "mean_leaf.hpp"

MeanLeaf::value_type MeanLeaf::build(std::span<const double> targets, std::span<const size_t> object_ids) const
{
    double sum = 0.;
    for (auto index : object_ids)
        sum += targets[index];

    return sum / object_ids.size();
}