#pragma once

#include <cstdint>
#include <vector>

#include "dataset/matrix.hpp"
#include "models/api/errors.hpp"
#include "models/api/model.hpp"
#include "models/common/class_mapper.hpp"
#include "models/decision_trees/engine/max_feature.hpp"
#include "models/decision_trees/engine/tree_node.hpp"
#include "models/decision_trees/engine/tree_stats.hpp"

struct TreeParams
{
    std::optional<size_t> max_depth;
    size_t min_samples_split;
    size_t min_samples_leaf;
};

class ICartModel
{
protected:
    std::optional<size_t> m_max_depth;
    size_t m_min_samples_split;
    size_t m_min_samples_leaf;

    size_t m_fit_feature_size{0}; ///< Число столбцов обучающей матрицы
    bool m_fitted{false};         ///< Признак успешного fit

    MaxFeature m_max_feature;
    std::uint32_t m_random_state;

    TreeInfo m_info;

public:
    ICartModel(TreeParams tree_params, MaxFeature max_feature = MaxFeaturesMode::All, std::uint32_t random_state = 0)
        : m_max_depth{tree_params.max_depth}, m_min_samples_split{tree_params.min_samples_split},
          m_min_samples_leaf{tree_params.min_samples_leaf}, m_max_feature(max_feature), m_random_state(random_state)
    {
        if (tree_params.min_samples_split < 2)
            throw HyperParameterError("CartModel: min samples split must be greater than 1");

        if (tree_params.min_samples_leaf < 1)
            throw HyperParameterError("CartModel: min samples leaf must be greater than 0");
    }

    virtual void fit_with_counts(
        const Matrix &features,
        std::span<const double> targets,
        std::span<const size_t> counts
    ) = 0;

    TreeInfo get_info() const { return m_info; }
};

class CartRegressor : public IRegressor, public ICartModel
{
private:
    Node<double> m_root{nullptr};

public:
    using ICartModel::ICartModel;

    void fit(const Matrix &features, std::span<const double> targets) override;

    void fit_with_counts(const Matrix &features, std::span<const double> targets, std::span<const size_t> counts)
        override;

    std::vector<double> predict(const Matrix &features) const override;
};

class CartClassifier : public IClassifier, public ICartModel
{
private:
    Node<std::vector<double>> m_root{nullptr};

    ClassMapper m_mapper;

public:
    using ICartModel::ICartModel;

    Matrix predict_proba(const Matrix &features) const override;

    void fit(const Matrix &features, std::span<const double> targets) override;

    void fit_with_counts(const Matrix &features, std::span<const double> targets, std::span<const size_t> counts)
        override;

    std::span<const double> classes() const override
    {
        if (!m_fitted)
            throw NotFitted("CartClassifier.classes: called before fit.");
        return m_mapper.classes();
    }
};