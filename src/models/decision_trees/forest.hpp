#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "models/api/errors.hpp"
#include "models/api/model.hpp"
#include "models/common/class_mapper.hpp"
#include "models/decision_trees/cart.hpp"
#include "models/decision_trees/max_feature.hpp"

struct RandomForestParams
{
    size_t estimators = 100;
    bool bootstrap = true;
    std::optional<size_t> max_samples = std::nullopt;
    std::optional<MaxFeature> max_features = std::nullopt;
    std::optional<size_t> max_depth = std::nullopt;
    size_t min_samples_split = 2;
    size_t min_samples_leaf = 1;
    std::optional<std::uint32_t> random_state = std::nullopt;
    size_t n_jobs = 1;
};

class IRandomForest
{
protected:
    size_t m_estimators;
    std::optional<size_t> m_max_samples;
    std::optional<MaxFeature> m_max_features;

    std::optional<size_t> m_max_depth;
    size_t m_min_samples_split;
    size_t m_min_samples_leaf;

    std::optional<std::uint32_t> m_random_state;

    size_t m_njobs;

    size_t m_features_fit_size = 0;

    bool m_bootstrap;

    bool m_fitted = false;

public:
    IRandomForest(const RandomForestParams &params)
        : m_estimators(params.estimators), m_max_samples(params.max_samples), m_max_features(params.max_features),
          m_max_depth(params.max_depth), m_min_samples_split(params.min_samples_split),
          m_min_samples_leaf(params.min_samples_leaf), m_random_state(params.random_state), m_njobs(params.n_jobs),
          m_bootstrap(params.bootstrap)
    {
        if (params.estimators < 1)
            throw HyperParameterError("RandomForest: estimators must be greater than 0");

        if (params.max_samples && params.max_samples.value() == 0)
            throw HyperParameterError("RandomForest: max samples must be greater than 0");

        if (params.min_samples_split < 2)
            throw HyperParameterError("RandomForest: min samples split must be at least 2");

        if (params.min_samples_leaf == 0)
            throw HyperParameterError("RandomForest: min samples leaf must be at least 1");

        if (params.n_jobs < 1)
            throw HyperParameterError("RandomForest: n_jobs must be greater than 0");

        if (!params.bootstrap && params.max_samples)
            throw HyperParameterError("RandomForest: bootstrap is false and max_samples must not be provided");
    };
};

class ForestRegressor : public IRandomForest, public IRegressor
{
private:
    std::vector<CartRegressor> m_trees;

public:
    ForestRegressor(const RandomForestParams &params) : IRandomForest(params)
    {
        if (!params.max_features)
            m_max_features = MaxFeaturesMode::All;
    }

    void fit(const Matrix &features, std::span<const double> targets) override;
    std::vector<double> predict(const Matrix &features) const override;
};

class ForestClassifier : public IRandomForest, public IClassifier
{
private:
    std::vector<CartClassifier> m_trees;

    ClassMapper m_mapper;

public:
    ForestClassifier(const RandomForestParams &params) : IRandomForest(params)
    {
        if (!params.max_features)
            m_max_features = MaxFeaturesMode::Sqrt;
    }

    void fit(const Matrix &features, std::span<const double> targets) override;
    Matrix predict_proba(const Matrix &) const override;

    std::span<const double> classes() const override
    {
        if (!m_fitted)
            throw NotFitted("ForestClassifier.classes: classes called before fit");

        return m_mapper.classes();
    };
};