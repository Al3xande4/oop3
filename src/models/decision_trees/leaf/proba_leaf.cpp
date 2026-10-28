#include "proba_leaf.hpp"

ProbaLeaf::value_type ProbaLeaf::build(std::span<const double> targets, std::span<const size_t> object_ids) const
{
    value_type probabilities(m_classes.size(), 0.0);

    if (object_ids.empty() || m_classes.empty())
        return probabilities;

    for (size_t id : object_ids)
    {
        const double target_val = targets[id];

        auto it = std::ranges::find(m_classes, target_val);
        if (it != m_classes.end())
            probabilities[it - m_classes.begin()] += 1.0;
    }

    auto total_samples = static_cast<double>(object_ids.size());
    for (double &prob : probabilities)
        prob /= total_samples;

    return probabilities;
}