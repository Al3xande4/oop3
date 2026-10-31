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

template <typename Func>
auto parallel_predict(size_t total_tasks, size_t n_jobs, Func &&task_fn)
    -> std::vector<std::invoke_result_t<Func, size_t>>
{
    using ResultType = std::invoke_result_t<Func, size_t>;
    std::vector<ResultType> results(total_tasks);

    size_t num_workers = std::min(n_jobs, total_tasks);
    if (num_workers <= 1)
    {
        for (size_t i = 0; i < total_tasks; ++i)
            results[i] = task_fn(i);
        return results;
    }

    std::vector<std::exception_ptr> errors(num_workers, nullptr);
    std::stop_source global_stop;

    {
        std::vector<std::jthread> workers;
        workers.reserve(num_workers);

        for (size_t worker_id = 0; worker_id < num_workers; ++worker_id)
        {
            workers.emplace_back(
                [&, worker_id]()
                {
                    try
                    {
                        for (size_t i = worker_id; i < total_tasks && !global_stop.stop_requested(); i += num_workers)
                            results[i] = task_fn(i);
                    }
                    catch (...)
                    {
                        errors[worker_id] = std::current_exception();
                        global_stop.request_stop();
                    }
                }
            );
        }
    }

    for (const auto &err : errors)
        if (err)
            std::rethrow_exception(err);

    return results;
}

std::vector<double> ForestRegressor::predict(const Matrix &features) const
{
    validation::validate_predict_input(features, m_fitted, m_features_fit_size, "ForestRegressor");

    auto all_predicts = parallel_predict(
        m_estimators, m_njobs, [&](size_t i) -> std::vector<double> { return m_trees[i].predict(features); }
    );
    std::vector<double> result(features.get_rows());

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

    auto all_predicts =
        parallel_predict(m_estimators, m_njobs, [&](size_t i) -> Matrix { return m_trees[i].predict_proba(features); });
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