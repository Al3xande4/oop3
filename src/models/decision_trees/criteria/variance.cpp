#include "variance.hpp"

void VarianceCriteria::calculate_total(std::span<const double> targets, std::span<const size_t> object_ids)
{
    m_total_sum = 0.;
    for (size_t i = 0; i < object_ids.size(); i++)
        m_total_sum += targets[object_ids[i]];

    m_total_count = object_ids.size();
}

void VarianceCriteria::init()
{
    m_sum_left = 0.;
    m_sum_right = m_total_sum;

    m_count_left = 0;
    m_count_right = m_total_count;
}

void VarianceCriteria::update(double target)
{
    m_sum_left += target;
    m_sum_right -= target;

    m_count_left++;
    m_count_right--;
}

double VarianceCriteria::score() const
{
    auto G = [](double s, size_t m) { return s * s / m; };

    return G(m_sum_left, m_count_left) + G(m_sum_right, m_count_right);
}