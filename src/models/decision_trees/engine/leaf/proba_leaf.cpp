#include "proba_leaf.hpp"

#include <span>

ProbaLeaf::value_type ProbaLeaf::build(
    std::span<const double> targets,
    std::span<const size_t> object_ids,
    std::span<const size_t> counts
) const
{
    value_type probabilities(m_classses_count, 0.0);

    if (object_ids.empty() || m_classses_count == 0)
        return probabilities;

    double w = 0.;
    for (size_t id : object_ids)
    {
        double c = counts.empty() ? 1.0 : static_cast<double>(counts[id]);
        probabilities[(size_t)targets[id]] += c;
        w += c;
    }

    for (double &prob : probabilities)
        prob /= w;

    return probabilities;
}