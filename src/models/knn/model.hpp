#pragma once

#include <vector>
#include <memory>

#include "models/api/model.hpp"
#include "models/api/distance.hpp"
#include "models/api/errors.hpp"

class IKnnWeight
{
public:
    virtual ~IKnnWeight() = default;
    virtual double weight(double) const = 0;
};

class UniformWeight : public IKnnWeight
{
private:
    double m_weight;

public:
    UniformWeight(double w = 1.) : m_weight{w} {};

    double weight(double) const override;
};

class InverseDistanceWeight : public IKnnWeight
{
public:
    double weight(double) const override;
};

class KnnModel : public IModel
{
protected:
    Matrix m_data;
    std::vector<double> m_target;

    size_t m_k;

    std::shared_ptr<IDistance> m_distance;
    std::shared_ptr<IKnnWeight> m_weight_strategy;

    bool m_fitted;

public:
    KnnModel(size_t k, std::shared_ptr<IDistance> distance, std::shared_ptr<IKnnWeight> weight_strategy) : m_k{k}, m_distance{distance}, m_weight_strategy{weight_strategy}, m_fitted{false}
    {
        if (k == 0)
            throw HyperParameterError("KnnModel: k must be greater than 0");
    };

    void fit(
        const Matrix &features,
        std::span<const double> target) override;

protected:
    struct Neighbor
    {
        size_t index;
        double distance;
    };

    std::vector<Neighbor> find_k_nearest(std::span<const double> object) const;
};

class KnnClassifier : public KnnModel
{
public:
    using KnnModel::KnnModel;

    std::vector<double> predict(const Matrix &features) const override;
};

class KnnRegressor : public KnnModel
{
public:
    using KnnModel::KnnModel;

    std::vector<double> predict(const Matrix &features) const override;
};
