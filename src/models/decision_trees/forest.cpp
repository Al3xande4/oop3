#include "forest.hpp"

#include <vector>

#include "dataset/matrix.hpp"
#include "models/common/validation.hpp"
#include "models/decision_trees/cart.hpp"
#include "models/decision_trees/forest_trainer.hpp"

void ForestRegressor::fit(const Matrix &features, std::span<const double> targets)
{
    m_fitted = false;
    validation::validate_fit_input(features, targets, "ForestRegressor");

    ForestTrainer<CartRegressor> forest_trainer(
        {m_estimators,
         m_bootstrap,
         m_max_samples,
         m_max_features,
         m_max_depth,
         m_min_samples_split,
         m_min_samples_leaf,
         m_random_state,
         m_njobs}
    );

    m_trees = forest_trainer.train(features, targets);

    m_fitted = true;
    m_features_fit_size = features.get_cols();
}

void ForestClassifier::fit(const Matrix &features, std::span<const double> targets)
{
    m_fitted = false;
    validation::validate_fit_input(features, targets, "ForestClassifier");

    m_mapper.fit(targets);

    ForestTrainer<CartClassifier> forest_trainer(
        {m_estimators,
         m_bootstrap,
         m_max_samples,
         m_max_features,
         m_max_depth,
         m_min_samples_split,
         m_min_samples_leaf,
         m_random_state,
         m_njobs}
    );

    m_trees = forest_trainer.train(features, targets);

    m_fitted = true;
    m_features_fit_size = features.get_cols();
}

std::vector<double> ForestRegressor::predict(const Matrix &features) const
{
    validation::validate_predict_input(features, m_fitted, m_features_fit_size, "ForestRegressor");

    std::vector<double> result(features.get_rows());

    std::vector<std::vector<double>> all_predicts(m_estimators);

    size_t num_workers = std::min(m_njobs, m_estimators);
    ParallelWorker worker{num_workers};
    worker.perform(
        [&](size_t worker_id, const std::stop_source &stop)
        {
            for (size_t i = worker_id; i < m_estimators && !stop.stop_requested(); i += num_workers)
                all_predicts[i] = m_trees[i].predict(features);
        }
    );

    for (size_t i = 0; i < m_estimators; i++)
        for (size_t row = 0; row < features.get_rows(); row++)
            result[row] += all_predicts[i][row];

    for (double &value : result)
        value /= m_estimators;

    return result;
}

Matrix ForestClassifier::predict_proba(const Matrix &features) const
{
    validation::validate_predict_input(features, m_fitted, m_features_fit_size, "ForestClassifier");

    std::vector<Matrix> all_predicts(m_estimators);

    size_t num_workers = std::min(m_njobs, m_estimators);
    ParallelWorker worker{num_workers};
    worker.perform(
        [&](size_t worker_id, const std::stop_source &stop)
        {
            for (size_t i = worker_id; i < m_estimators && !stop.stop_requested(); i += num_workers)
                all_predicts[i] = m_trees[i].predict_proba(features);
        }
    );

    Matrix result(features.get_rows(), m_mapper.num_classes());

    for (size_t i = 0; i < m_estimators; i++)
        for (size_t row = 0; row < features.get_rows(); row++)
            for (size_t cls = 0; cls < m_mapper.num_classes(); cls++)
                result(row, cls) += all_predicts[i](row, cls);

    for (size_t row = 0; row < features.get_rows(); row++)
        for (size_t cls = 0; cls < m_mapper.num_classes(); cls++)
            result(row, cls) /= m_estimators;

    return result;
}