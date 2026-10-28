#include "cart.hpp"

#include <algorithm>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

#include "models/api/errors.hpp"
#include "models/decision_trees/criteria/gini.hpp"
#include "models/decision_trees/criteria/variance.hpp"
#include "models/decision_trees/leaf/mean_leaf.hpp"
#include "models/decision_trees/leaf/proba_leaf.hpp"
#include "tree_builder.hpp"
#include "tree_searcher.hpp"

void CartRegressor::fit(const Matrix &features, std::span<const double> targets)
{

    if (features.get_rows() != targets.size())
        throw DimensionMismatch(
            "CartRegressor.fit: features and targets mismatch sizes: features - " +
            std::to_string(features.get_rows()) + ", targets - " + std::to_string(targets.size())
        );

    if (targets.size() == 0)
        throw EmptyData("CartRegressor.fit: targets size is 0");

    m_fitted = true;
    m_fit_feature_size = features.get_cols();

    TreeBuilder tree_builder(VarianceCriteria{}, MeanLeaf{}, m_max_depth, m_min_samples_split, m_min_samples_leaf);

    std::vector<size_t> object_ids(features.get_rows());
    std::iota(object_ids.begin(), object_ids.end(), 0ULL);

    std::vector<size_t> feature_ids(features.get_cols());
    std::iota(feature_ids.begin(), feature_ids.end(), 0ULL);

    m_root = tree_builder.build(features, targets, object_ids, feature_ids);
}

std::vector<double> CartRegressor::predict(const Matrix &features) const
{
    if (!m_fitted)
        throw NotFitted("CartRegressor: predict called before fit");

    if (features.get_cols() != m_fit_feature_size)
        throw DimensionMismatch(
            "CartRegressor: predict features sizes doesn't match fit features size. Predict features size - " +
            std::to_string(features.get_cols()) + ", training feature size - " + std::to_string(m_fit_feature_size)
        );

    TreeSearcher tree_searcher;

    std::vector<double> results(features.get_rows());

    for (size_t row = 0; row < features.get_rows(); row++)
        results[row] = tree_searcher.search(m_root, features(row, std::nullopt)).get_value();

    return results;
}

void CartClassifier::fit(const Matrix &features, std::span<const double> targets)
{

    if (features.get_rows() != targets.size())
        throw DimensionMismatch(
            "CartClassifier.fit: features and targets mismatch sizes: features - " +
            std::to_string(features.get_rows()) + ", targets - " + std::to_string(targets.size())
        );

    if (targets.size() == 0)
        throw EmptyData("CartClassifier.fit: targets size is 0");

    m_fitted = true;
    m_fit_feature_size = features.get_cols();

    m_classes.assign(targets.begin(), targets.end());
    std::ranges::sort(m_classes);
    auto [drop_begin, drop_end] = std::ranges::unique(m_classes);
    m_classes.erase(drop_begin, drop_end);

    TreeBuilder tree_builder(
        GiniCriteria{m_classes.size()}, ProbaLeaf{m_classes}, m_max_depth, m_min_samples_split, m_min_samples_leaf
    );

    std::vector<size_t> object_ids(features.get_rows());
    std::iota(object_ids.begin(), object_ids.end(), 0ULL);

    std::vector<size_t> feature_ids(features.get_cols());
    std::iota(feature_ids.begin(), feature_ids.end(), 0ULL);

    m_root = tree_builder.build(features, targets, object_ids, feature_ids);
}

std::vector<double> CartClassifier::predict(const Matrix &features) const
{
    if (!m_fitted)
        throw NotFitted("CartClassifier: predict called before fit");

    if (features.get_cols() != m_fit_feature_size)
        throw DimensionMismatch(
            "CartClassifier: predict features sizes doesn't match fit features size. Predict features size - " +
            std::to_string(features.get_cols()) + ", training feature size - " + std::to_string(m_fit_feature_size)
        );

    TreeSearcher tree_searcher;

    std::vector<double> results(features.get_rows());

    for (size_t row = 0; row < features.get_rows(); row++)
    {
        auto leaf = tree_searcher.search(m_root, features(row, std::nullopt));
        const auto &probabilities = leaf.get_value();

        auto it = std::ranges::max_element(probabilities);
        results[row] = m_classes[it - probabilities.begin()];
    }

    return results;
}

Matrix CartClassifier::predict_proba(const Matrix &features) const
{
    if (!m_fitted)
        throw NotFitted("CartClassifier: predict called before fit");

    if (features.get_cols() != m_fit_feature_size)
        throw DimensionMismatch(
            "CartClassifier: predict features sizes doesn't match fit features size. Predict features size - " +
            std::to_string(features.get_cols()) + ", training feature size - " + std::to_string(m_fit_feature_size)
        );

    TreeSearcher tree_searcher;

    Matrix result(features.get_rows(), m_classes.size());

    for (size_t row = 0; row < features.get_rows(); row++)
    {
        auto leaf = tree_searcher.search(m_root, features(row, std::nullopt));
        const auto &probabilities = leaf.get_value();

        for (size_t i = 0; i < probabilities.size(); i++)
            result(row, i) = probabilities[i];
    }

    return result;
}