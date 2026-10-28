#include "gini.hpp"

#include <algorithm>

void GiniCriteria::calculate_total(std::span<const double> targets, std::span<const size_t> object_ids)
{
    std::ranges::fill(m_category_count, 0);

    for (size_t object_id : object_ids)
        m_category_count[(size_t)targets[object_id]]++;

    m_total_count = object_ids.size();
}

void GiniCriteria::init()
{
    std::ranges::copy(m_category_count, m_category_count_right.begin());

    m_left = 0.;
    m_right = 0.;
    for (auto &freq : m_category_count)
        m_right += freq * freq;

    m_count_left = 0;
    m_count_right = m_total_count;
}

void GiniCriteria::update(double target)
{
    size_t index = (size_t)target;
    m_left += 2 * (m_category_count[index] - m_category_count_right[index]) + 1;
    m_right += 1 - 2 * (double)m_category_count_right[index];

    m_category_count_right[index]--;

    m_count_left++;
    m_count_right--;
}

double GiniCriteria::score() const
{
    auto H = [](double a, size_t m) { return a / m; };

    return H(m_left, m_count_left) + H(m_right, m_count_right);
}