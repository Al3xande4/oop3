#include "stump.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_map>

#include "engine/criteria/gini.hpp"
#include "engine/criteria/variance.hpp"
#include "engine/split_finder.hpp"
#include "models/api/errors.hpp"
#include "models/common/validation.hpp"

void DecisionStumpRegressor::fit(const Matrix &features, std::span<const double> targets)
{
    validation::validate_fit_input(features, targets, "DecisionStumpRegressor");

    m_fitted = true;
    m_fit_feature_size = features.get_cols();

    std::vector<size_t> object_ids(features.get_rows());
    std::iota(object_ids.begin(), object_ids.end(), 0ULL);

    std::vector<size_t> feature_ids(features.get_cols());
    std::iota(feature_ids.begin(), feature_ids.end(), 0ULL);

    VarianceCriteria crit;

    SplitFinder m_split_finder;
    m_split = m_split_finder.find_best(features, targets, object_ids, feature_ids, crit, m_min_samples_leaf);

    if (!m_split)
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
        if (features(row, m_split.value().feature) <= m_split.value().threshold)
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

std::vector<double> DecisionStumpRegressor::predict(const Matrix &features) const
{
    if (!m_fitted)
        throw NotFitted("Decision stump: predict called before fit");

    if (features.get_cols() != m_fit_feature_size)
        throw DimensionMismatch(
            "Decision stump: predict features sizes doesn't match fit features size. Predict features size - " +
            std::to_string(features.get_cols()) + ", training feature size - " + std::to_string(m_fit_feature_size)
        );

    std::vector<double> result(features.get_rows());

    for (size_t row = 0; row < features.get_rows(); row++)
    {
        if (!m_split)
            result[row] = m_constant;
        else if (features(row, m_split.value().feature) <= m_split.value().threshold)
            result[row] = m_left_mean;
        else
            result[row] = m_right_mean;
    }

    return result;
}

void DecisionStumpClassifier::fit(const Matrix &features, std::span<const double> targets)
{

    validation::validate_fit_input(features, targets, "DecisionStumpClassifier");

    std::vector<size_t> object_ids(features.get_rows());
    std::iota(object_ids.begin(), object_ids.end(), 0ULL);

    std::vector<size_t> feature_ids(features.get_cols());
    std::iota(feature_ids.begin(), feature_ids.end(), 0ULL);

    size_t classes_count = std::ranges::max(targets) + 1;
    GiniCriteria crit(classes_count);

    SplitFinder m_split_finder;
    std::vector<size_t> counts(features.get_rows(), 1);
    m_split = m_split_finder.find_best(features, targets, object_ids, feature_ids, crit, m_min_samples_leaf);

    m_fitted = true;
    m_fit_feature_size = features.get_cols();

    if (!m_split)
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
        if (features(row, m_split.value().feature) <= m_split.value().threshold)
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

std::vector<double> DecisionStumpClassifier::predict(const Matrix &features) const
{
    if (!m_fitted)
        throw NotFitted("Decision stump: predict called before fit");

    if (features.get_cols() != m_fit_feature_size)
        throw DimensionMismatch(
            "Decision stump: predict features sizes doesn't match fit features size. Predict features size - " +
            std::to_string(features.get_cols()) + ", training feature size - " + std::to_string(m_fit_feature_size)
        );

    std::vector<double> result(features.get_rows());

    for (size_t row = 0; row < features.get_rows(); row++)
    {
        if (!m_split)
            result[row] = m_constant;
        else if (features(row, m_split.value().feature) <= m_split.value().threshold)
            result[row] = m_left_most_common;
        else
            result[row] = m_right_most_common;
    }

    return result;
}