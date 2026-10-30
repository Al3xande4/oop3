#include "cart.hpp"

#include <optional>
#include <vector>

#include "dataset/matrix.hpp"
#include "models/common/validation.hpp"
#include "models/decision_trees/criteria/gini.hpp"
#include "models/decision_trees/criteria/variance.hpp"
#include "models/decision_trees/leaf/mean_leaf.hpp"
#include "models/decision_trees/leaf/proba_leaf.hpp"
#include "tree_builder.hpp"
#include "tree_searcher.hpp"
#include "tree_stats.hpp"

void CartRegressor::fit(const Matrix &features, std::span<const double> targets)
{
    std::vector<size_t> counts(features.get_rows(), 1);
    fit_with_counts(features, targets, counts);
}

void CartRegressor::fit_with_counts(
    const Matrix &features,
    std::span<const double> targets,
    std::span<const size_t> counts
)
{
    validation::validate_fit_input(features, targets, "CartRegressor");

    m_fitted = true;
    m_fit_feature_size = features.get_cols();

    TreeBuilder tree_builder(
        VarianceCriteria{},
        MeanLeaf{},
        m_max_depth,
        m_min_samples_split,
        m_min_samples_leaf,
        m_max_feature,
        m_random_state
    );

    std::vector<size_t> object_ids(features.get_rows());
    std::iota(object_ids.begin(), object_ids.end(), 0ULL);

    std::vector<size_t> feature_ids(features.get_cols());
    std::iota(feature_ids.begin(), feature_ids.end(), 0ULL);

    m_root = tree_builder.build(features, targets, object_ids, feature_ids, counts);

    TreeInspector inspector;
    m_info = inspector.info(m_root);
}

std::vector<double> CartRegressor::predict(const Matrix &features) const
{
    validation::validate_predict_input(features, m_fitted, m_fit_feature_size, "CartRegressor");

    TreeSearcher tree_searcher;

    std::vector<double> results(features.get_rows());

    for (size_t row = 0; row < features.get_rows(); row++)
        results[row] = tree_searcher.search(m_root, features(row, std::nullopt)).get_value();

    return results;
}

void CartClassifier::fit(const Matrix &features, std::span<const double> targets)
{
    std::vector<size_t> counts(features.get_rows(), 1);
    fit_with_counts(features, targets, counts);
}

void CartClassifier::fit_with_counts(
    const Matrix &features,
    std::span<const double> targets,
    std::span<const size_t> counts
)
{
    validation::validate_fit_input(features, targets, "CartClassifier");

    m_fitted = true;
    m_fit_feature_size = features.get_cols();

    m_mapper.fit(targets);

    TreeBuilder tree_builder(
        GiniCriteria{m_mapper.num_classes()},
        ProbaLeaf{m_mapper.num_classes()},
        m_max_depth,
        m_min_samples_split,
        m_min_samples_leaf,
        m_max_feature,
        m_random_state
    );

    std::vector<size_t> object_ids(features.get_rows());
    std::iota(object_ids.begin(), object_ids.end(), 0ULL);

    std::vector<size_t> feature_ids(features.get_cols());
    std::iota(feature_ids.begin(), feature_ids.end(), 0ULL);

    m_root = tree_builder.build(features, m_mapper.encode(targets), object_ids, feature_ids, counts);

    TreeInspector inspector;
    m_info = inspector.info(m_root);
}

Matrix CartClassifier::predict_proba(const Matrix &features) const
{
    validation::validate_predict_input(features, m_fitted, m_fit_feature_size, "CartClassifier");

    TreeSearcher tree_searcher;

    Matrix result(features.get_rows(), m_mapper.num_classes());

    for (size_t row = 0; row < features.get_rows(); row++)
    {
        auto leaf = tree_searcher.search(m_root, features(row, std::nullopt));
        const auto &probabilities = leaf.get_value();

        for (size_t i = 0; i < probabilities.size(); i++)
            result(row, i) = probabilities[i];
    }

    return result;
}