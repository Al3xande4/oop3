#include "gini.hpp"

#include <algorithm>

void GiniCriteria::calculate_total(
    std::span<const double> targets,
    std::span<const size_t> object_ids,
    std::span<const size_t> counts
)
{
    std::ranges::fill(m_category_count, 0);
    m_total_count = 0;

    for (size_t object_id : object_ids)
    {
        m_category_count[(size_t)targets[object_id]] += counts[object_id];
        m_total_count += counts[object_id];
    }

    m_initial_right = 0.;
    for (auto &freq : m_category_count)
        m_initial_right += freq * freq;
}

void GiniCriteria::init()
{
    std::ranges::copy(m_category_count, m_category_count_right.begin());

    m_left = 0.;
    m_right = m_initial_right;

    m_count_left = 0;
    m_count_right = m_total_count;
}

void GiniCriteria::update(double target, double weight)
{
    auto index = (size_t)target;

    double a = m_category_count[index] - m_category_count_right[index];
    double b = (double)m_category_count_right[index];

    m_left += 2 * a * weight + weight * weight;
    m_right += weight * weight - 2 * b * weight;

    m_category_count_right[index] -= weight;

    m_count_left += weight;
    m_count_right -= weight;
}

double GiniCriteria::score() const
{
    auto H = [](double a, size_t m) { return a / m; };

    return H(m_left, m_count_left) + H(m_right, m_count_right);
}