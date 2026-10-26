#pragma once

#include "models/api/model.hpp"
#include "models/api/errors.hpp"

#include <span>
#include <memory>
#include <optional>
#include <unordered_map>

class ISearchCriteria
{
public:
    virtual void calculate_total(std::span<const double>) = 0;
    virtual void init() = 0;
    virtual void update(double) = 0;
    virtual double score() const = 0;

    virtual ~ISearchCriteria() = default;
};

class VarianceCriteria : public ISearchCriteria
{
private:
    double m_sum_left, m_sum_right, m_total_sum;
    size_t m_count_left, m_count_right, m_total_count;

public:
    void calculate_total(std::span<const double>) override;
    void init() override;
    void update(double) override;
    double score() const override;
};

class GiniCriteria : public ISearchCriteria
{
private:
    std::unordered_map<double, size_t> m_category_count;
    std::unordered_map<double, size_t> m_category_count_right;
    size_t m_count_left, m_count_right, m_total_count;
    double m_left, m_right;

public:
    void calculate_total(std::span<const double>) override;
    void init() override;
    void update(double) override;
    double score() const override;
};

class IDecisionStump : public IModel
{
protected:
    std::optional<size_t> m_feature_index;
    std::optional<double> m_threshold;

    std::unique_ptr<ISearchCriteria> m_criteria;
    size_t m_min_samples_leaf;

    bool m_fitted;
    size_t m_fit_feature_size;

public:
    IDecisionStump(std::unique_ptr<ISearchCriteria> criteria, size_t min_samples_leaf) : m_feature_index{std::nullopt}, m_threshold{std::nullopt}, m_criteria(std::move(criteria)), m_min_samples_leaf{min_samples_leaf}, m_fitted{false}, m_fit_feature_size{0}
    {
        if (min_samples_leaf < 1)
            throw HyperParameterError("DecisionStump: min samples leaf must be greater than 0");
    };

    virtual void fit(
        const Matrix &features,
        std::span<const double> target);

    std::optional<size_t> split_feature() const
    {
        if (!m_fitted)
            throw NotFitted("DesionStump: split_feature before fit");

        return m_feature_index;
    };
    std::optional<double> split_threshold() const
    {
        if (!m_fitted)
            throw NotFitted("DesionStump: split_threshold before fit");

        return m_threshold;
    };

    virtual double get_left_leaf() const = 0;
    virtual double get_right_leaf() const = 0;

    virtual ~IDecisionStump() = default;
};

class DecisionStumpRegressor : public IDecisionStump
{
protected:
    double m_left_mean;
    double m_right_mean;
    double m_constant;

public:
    DecisionStumpRegressor(size_t min_samples_leaf) : IDecisionStump(std::make_unique<VarianceCriteria>(), min_samples_leaf), m_left_mean{0.}, m_right_mean{0.}, m_constant{0.} {};

    void fit(
        const Matrix &features,
        std::span<const double> target) override;

    std::vector<double> predict(
        const Matrix &features) const override;

    double get_left_leaf() const override
    {
        if (!m_fitted)
            throw NotFitted("DesionStumpRegressor: get_left_leaf before fit");

        return m_left_mean;
    }
    double get_right_leaf() const override
    {
        if (!m_fitted)
            throw NotFitted("DesionStumpRegressor: get_right_leaf before fit");

        return m_right_mean;
    }
};

class DecisionStumpClassifier : public IDecisionStump
{
protected:
    double m_left_most_common;
    double m_right_most_common;
    double m_constant;

public:
    DecisionStumpClassifier(size_t min_samples_leaf) : IDecisionStump(std::make_unique<GiniCriteria>(), min_samples_leaf), m_left_most_common{0.}, m_right_most_common{0.}, m_constant{0.} {};

    void fit(
        const Matrix &features,
        std::span<const double> target) override;

    std::vector<double> predict(
        const Matrix &features) const override;

    double get_left_leaf() const override
    {
        if (!m_fitted)
            throw NotFitted("DesionStumpClassifier: get_left_leaf before fit");

        return m_left_most_common;
    }
    double get_right_leaf() const override
    {
        if (!m_fitted)
            throw NotFitted("DesionStumpClassifier: get_right_leaf before fit");

        return m_right_most_common;
    }
};