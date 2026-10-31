#pragma once

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <random>
#include <span>
#include <stop_token>
#include <thread>
#include <vector>

#include "dataset/matrix.hpp"
#include "models/common/bootstrap_sampler.hpp"
#include "models/decision_trees/cart.hpp"
#include "models/decision_trees/forest.hpp"
#include "models/decision_trees/max_feature.hpp"

template <class TreeType>
concept CartTree =
    std::constructible_from<TreeType, TreeParams, MaxFeature, std::uint32_t> &&
    requires(TreeType tree, const Matrix &features, std::span<const double> targets, std::span<const size_t> counts) {
        { tree.fit_with_counts(features, targets, counts) } -> std::same_as<void>;
    };

template <CartTree Tree>
class ForestTrainer
{
private:
    RandomForestParams m_params;

public:
    ForestTrainer(const RandomForestParams &params) : m_params(params){};

    std::vector<Tree> train(const Matrix &features, std::span<const double> targets);

private:
    std::vector<std::uint32_t> _generate_seeds()
    {
        std::vector<std::uint32_t> seeds(m_params.estimators);
        std::mt19937 gen(m_params.random_state.value_or(std::random_device{}()));

        for (auto &seed : seeds)
            seed = static_cast<std::uint32_t>(gen());

        return seeds;
    }

    void _train_one(const Matrix &features, std::span<const double> targets, Tree &tree, std::uint32_t seed);
};

template <CartTree Tree>
void ForestTrainer<Tree>::_train_one(
    const Matrix &features,
    std::span<const double> targets,
    Tree &tree,
    std::uint32_t seed
)
{
    BootstrapSampler sampler(m_params.bootstrap, m_params.max_samples);

    std::mt19937 gen{seed};

    BootstrapResult bootstrap = sampler.sample(features.get_rows(), gen);

    tree.fit_with_counts(features, targets, bootstrap.counts);
};

template <CartTree Tree>
std::vector<Tree> ForestTrainer<Tree>::train(const Matrix &features, std::span<const double> targets)
{
    std::vector<Tree> trees;
    trees.reserve(m_params.estimators);

    auto seeds = _generate_seeds();

    TreeParams tree_params = {
        .max_depth = m_params.max_depth,
        .min_samples_split = m_params.min_samples_split,
        .min_samples_leaf = m_params.min_samples_leaf,
    };

    for (auto seed : seeds)
        trees.emplace_back(tree_params, m_params.max_features.value(), seed);

    size_t num_workers = std::min(m_params.n_jobs, m_params.estimators);
    if (num_workers <= 1)
    {
        for (size_t i = 0; i < m_params.estimators; i++)
            _train_one(features, targets, trees[i], seeds[i]);

        return trees;
    }

    std::vector<std::exception_ptr> errors(num_workers, nullptr);
    std::stop_source global_stop;

    {
        std::vector<std::jthread> workers;
        workers.reserve(num_workers);

        for (size_t worker_id = 0; worker_id < num_workers; worker_id++)
        {
            workers.emplace_back(
                [&, worker_id]()
                {
                    try
                    {
                        for (size_t i = worker_id; i < m_params.estimators && !global_stop.stop_requested();
                             i += num_workers)
                        {
                            _train_one(features, targets, trees[i], seeds[i]);
                        }
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

    for (const auto &error : errors)
        if (error)
            std::rethrow_exception(error);

    return trees;
}