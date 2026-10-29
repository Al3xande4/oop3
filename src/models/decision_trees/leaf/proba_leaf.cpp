#include "proba_leaf.hpp"

ProbaLeaf::value_type ProbaLeaf::build(std::span<const double> targets, std::span<const size_t> object_ids) const
{
    value_type probabilities(m_classses_count, 0.0);

    if (object_ids.empty() || m_classses_count == 0)
        return probabilities;

    for (size_t id : object_ids)
        probabilities[(size_t)targets[id]] += 1.0;

    auto total_samples = static_cast<double>(object_ids.size());
    for (double &prob : probabilities)
        prob /= total_samples;

    return probabilities;
}