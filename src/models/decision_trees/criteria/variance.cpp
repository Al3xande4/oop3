#include "variance.hpp"

void VarianceCriteria::calculate_total(
    std::span<const double> targets,
    std::span<const size_t> object_ids,
    std::span<const size_t> counts
)
{
    m_total_sum = 0.;
    m_total_count = 0.;

    for (size_t object_id : object_ids)
    {
        m_total_sum += counts[object_id] * targets[object_id];
        m_total_count += counts[object_id];
    }
}

void VarianceCriteria::init()
{
    m_sum_left = 0.;
    m_sum_right = m_total_sum;

    m_count_left = 0;
    m_count_right = m_total_count;
}

void VarianceCriteria::update(double target, double weight)
{
    m_sum_left += weight * target;
    m_sum_right -= weight * target;

    m_count_left += weight;
    m_count_right -= weight;
}

double VarianceCriteria::score() const
{
    auto G = [](double s, size_t m) { return s * s / m; };

    return G(m_sum_left, m_count_left) + G(m_sum_right, m_count_right);
}