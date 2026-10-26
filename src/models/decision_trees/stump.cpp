#include "stump.hpp"
#include "models/api/errors.hpp"

#include <cmath>
#include <ranges>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <stdexcept>

void VarianceCriteria::calculate_total(std::span<const double> targets)
{
    m_total_sum = 0.;
    for (const auto &t : targets)
        m_total_sum += t;

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

void GiniCriteria::calculate_total(std::span<const double> targets)
{
    m_category_count.clear();

    for (const auto &t : targets)
        m_category_count[t]++;

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

void IDecisionStump::fit(
    const Matrix &features,
    std::span<const double> targets)
{
    if (features.get_rows() != targets.size())
        throw DimensionMismatch("DecisionStump.fit: features and targets mismatch sizes: features - " + std::to_string(features.get_rows()) + ", targets - " + std::to_string(targets.size()));

    if (targets.size() == 0)
        throw EmptyData("DecisionStump.fit: targets size is 0");

    m_threshold = std::nullopt;
    m_feature_index = std::nullopt;

    m_criteria->calculate_total(targets);

    size_t best = 0;
    double threshold = 0;
    double best_score = -std::numeric_limits<double>::infinity();
    bool chosen = false;

    std::vector<size_t> order(features.get_rows());
    std::iota(order.begin(), order.end(), 0ULL);

    for (size_t col = 0; col < features.get_cols(); col++)
    {
        std::ranges::sort(order, [&](size_t a, size_t b)
                          { return features[a][col] < features[b][col]; });

        m_criteria->init();
        size_t count_left = 0, count_right = features.get_rows();

        for (size_t p = 0; p + 1 < features.get_rows(); p++)
        {
            size_t row = order[p];

            m_criteria->update(targets[row]);
            count_left++;
            count_right--;

            if (features[row][col] == features[order[p + 1]][col])
                continue;

            if (count_right < m_min_samples_leaf)
                break;

            if (count_left < m_min_samples_leaf)
                continue;

            double score = m_criteria->score();
            double t = std::midpoint(features[row][col], features[order[p + 1]][col]);

            if (score > best_score)
            {
                best = col;
                chosen = true;
                threshold = t;
                best_score = score;
            }
        }
    }

    m_fitted = true;
    m_fit_feature_size = features.get_cols();

    if (!chosen)
        return;

    m_threshold = threshold;
    m_feature_index = best;
}

void DecisionStumpRegressor::fit(
    const Matrix &features,
    std::span<const double> targets)
{
    IDecisionStump::fit(features, targets);

    if (!m_threshold)
    {
        double sum = 0.;
        for (const auto &t : targets)
            sum += t;

        m_constant = sum / targets.size();
        m_left_mean = m_right_mean = m_constant;

        return;
    }

    size_t count_left = 0, count_right = 0;
    double sum_left = 0., sum_right = 0.;

    for (size_t row = 0; row < features.get_rows(); row++)
    {
        if (features[row][*m_feature_index] <= *m_threshold)
        {
            count_left++;
            sum_left += targets[row];
        }
        else
        {
            count_right++;
            sum_right += targets[row];
        }
    }

    m_left_mean = sum_left / count_left;
    m_right_mean = sum_right / count_right;
}

std::vector<double> DecisionStumpRegressor::predict(
    const Matrix &features) const
{
    if (!m_fitted)
        throw NotFitted("Decision stump: predict called before fit");

    if (features.get_cols() != m_fit_feature_size)
        throw DimensionMismatch("Decision stump: predict features sizes doesn't match fit features size. Predict features size - " + std::to_string(features.get_cols()) + ", training feature size - " + std::to_string(m_fit_feature_size));

    std::vector<double> result(features.get_rows());

    for (size_t row = 0; row < features.get_rows(); row++)
    {
        if (!m_feature_index)
            result[row] = m_constant;
        else if (features[row][*m_feature_index] <= *m_threshold)
            result[row] = m_left_mean;
        else
            result[row] = m_right_mean;
    }

    return result;
}

void DecisionStumpClassifier::fit(
    const Matrix &features,
    std::span<const double> targets)
{
    IDecisionStump::fit(features, targets);

    if (!m_threshold)
    {
        std::unordered_map<double, size_t> occ;
        size_t best = 0;

        for (const auto &t : targets)
        {
            occ[t]++;
            if (occ[t] > best)
            {
                best = occ[t];
                m_constant = t;
            }
        }
        m_left_most_common = m_right_most_common = m_constant;

        return;
    }

    std::unordered_map<double, size_t> left_count;
    std::unordered_map<double, size_t> right_count;
    size_t left_best = 0, right_best = 0;

    for (size_t row = 0; row < features.get_rows(); row++)
    {
        if (features[row][*m_feature_index] <= *m_threshold)
        {
            left_count[targets[row]]++;
            if (left_count[targets[row]] > left_best)
            {
                left_best = left_count[targets[row]];
                m_left_most_common = targets[row];
            }
        }
        else
        {
            right_count[targets[row]]++;
            if (right_count[targets[row]] > right_best)
            {
                right_best = right_count[targets[row]];
                m_right_most_common = targets[row];
            }
        }
    }
}

std::vector<double> DecisionStumpClassifier::predict(
    const Matrix &features) const
{
    if (!m_fitted)
        throw NotFitted("Decision stump: predict called before fit");

    if (features.get_cols() != m_fit_feature_size)
        throw DimensionMismatch("Decision stump: predict features sizes doesn't match fit features size. Predict features size - " + std::to_string(features.get_cols()) + ", training feature size - " + std::to_string(m_fit_feature_size));

    std::vector<double> result(features.get_rows());

    for (size_t row = 0; row < features.get_rows(); row++)
    {
        if (!m_feature_index)
            result[row] = m_constant;
        else if (features[row][*m_feature_index] <= *m_threshold)
            result[row] = m_left_most_common;
        else
            result[row] = m_right_most_common;
    }

    return result;
}