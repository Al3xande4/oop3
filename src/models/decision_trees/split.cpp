#include "split.hpp"
#include "models/api/errors.hpp"

#include <numeric>
#include <algorithm>
#include <ranges>

void VarianceCriteria::calculate_total(std::span<const double> targets, std::span<const size_t> object_ids)
{
    m_total_sum = 0.;
    for (size_t i = 0; i < object_ids.size(); i++)
        m_total_sum += targets[object_ids[i]];

    m_total_count = targets.size();
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
    auto G = [](double s, size_t m)
    { return s * s / m; };

    return G(m_sum_left, m_count_left) + G(m_sum_right, m_count_right);
}

void GiniCriteria::calculate_total(std::span<const double> targets, std::span<const size_t> object_ids)
{
    m_category_count.clear();

    for (size_t i = 0; i < object_ids.size(); i++)
        m_category_count[targets[object_ids[i]]]++;

    m_total_count = targets.size();
}

void GiniCriteria::init()
{
    m_category_count_right = m_category_count;

    m_left = 0.;
    m_right = 0.;
    for (auto &[_, freq] : m_category_count)
        m_right += freq * freq;

    m_count_left = 0;
    m_count_right = m_total_count;
}

void GiniCriteria::update(double target)
{
    m_left += 2 * (m_category_count[target] - m_category_count_right[target]) + 1;
    m_right += 1 - 2 * (double)m_category_count_right[target];

    m_category_count_right[target]--;

    m_count_left++;
    m_count_right--;
}

double GiniCriteria::score() const
{
    auto H = [](double a, size_t m)
    { return a / m; };

    return H(m_left, m_count_left) + H(m_right, m_count_right);
}

std::optional<Split> SplitFinder::find_best(
    const Matrix &features,
    std::span<const double> targets,
    std::span<const std::size_t> object_ids,
    std::span<const std::size_t> feature_ids,
    std::unique_ptr<ISearchCriteria> criteria,
    std::size_t min_samples_leaf) const
{
    if (object_ids.size() > targets.size() || object_ids.size() > features.get_rows())
        throw DimensionMismatch("SplitFinder.find_best: objects and features mismatch sizes");

    std::optional<Split> result = std::nullopt;
    criteria->calculate_total(targets, object_ids);

    std::vector<size_t> order(object_ids.size());
    for (size_t i = 0; i < object_ids.size(); i++)
        order[i] = object_ids[i];

    std::vector<size_t> columns(feature_ids.size());
    for (size_t i = 0; i < feature_ids.size(); i++)
        columns[i] = feature_ids[i];
    std::ranges::sort(columns);

    for (size_t i = 0; i < feature_ids.size(); i++)
    {
        size_t col = feature_ids[i];

        std::ranges::sort(order, [&](size_t a, size_t b)
                          { return features[a][col] < features[b][col]; });

        criteria->init();
        size_t count_left = 0, count_right = object_ids.size();

        for (size_t p = 0; p + 1 < order.size(); p++)
        {
            size_t row = order[p];
            size_t next_row = order[p + 1];

            criteria->update(targets[row]);
            count_left++;
            count_right--;

            if (features(row, std::nullopt)(col) == features(next_row, std::nullopt)(col))
                continue;

            if (count_right < min_samples_leaf)
                break;

            if (count_left < min_samples_leaf)
                continue;

            double score = criteria->score();
            double t = std::midpoint(features(row, col), features(next_row, col));

            if (!result || score > result.value().score)
                result = {.feature = col, .threshold = t, .score = score};
        }
    }

    return result;
}