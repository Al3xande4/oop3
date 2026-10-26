#include "models/knn/model.hpp"
#include "models/api/errors.hpp"

#include <functional>
#include <map>
#include <queue>
#include <iostream>
#include <algorithm>

double UniformWeight::weight(double) const
{
    return m_weight;
};

double InverseDistanceWeight::weight(double dist) const
{
    return 1. / (1e-9 + dist);
};

void KnnModel::fit(
    const Matrix &features,
    std::span<const double> target)
{
    if (features.get_rows() != target.size())
        throw DimensionMismatch("KnnModel.fit: features and targets mismatch: " + std::to_string(features.get_rows()) + ", " + std::to_string(target.size()));

    if (features.get_rows() == 0)
        throw EmptyData("KnnModel.fit: empty data");

    if (m_k > features.get_rows())
        throw std::invalid_argument("KnnModel.fit: k is bigger than training data");

    m_data = features;
    m_target.assign(target.begin(), target.end());

    m_fitted = true;
}

std::vector<KnnModel::Neighbor> KnnModel::find_k_nearest(std::span<const double> object) const
{
    auto comparator = [](const Neighbor &a, const Neighbor &b)
    { return a.distance < b.distance; };

    std::priority_queue<Neighbor, std::vector<Neighbor>, decltype(comparator)> nearest(comparator);

    for (size_t i = 0; i < m_data.get_rows(); i++)
    {
        double dist = m_distance->compute(m_data[i], object);

        if (nearest.size() < m_k)
        {
            nearest.emplace(i, dist);
            continue;
        }

        if (nearest.top().distance > dist)
        {
            nearest.pop();
            nearest.emplace(i, dist);
        }
    }

    std::vector<Neighbor> result;
    while (!nearest.empty())
    {
        result.emplace_back(std::move(nearest.top()));
        nearest.pop();
    }

    std::ranges::sort(result, comparator);
    return result;
};

std::vector<double> KnnClassifier::predict(const Matrix &features) const
{
    if (!m_fitted)
        throw NotFitted("KnnClassifier.predict called before fit");

    if (features.get_cols() != m_data.get_cols())
        throw DimensionMismatch("KnnClassifier.predict features sizes doesn't match fit features size. Predict features size - " + std::to_string(features.get_cols()) + ", training feature size - " + std::to_string(m_data.get_cols()));

    std::vector<double> predictions;
    predictions.reserve(features.get_rows());

    for (size_t i = 0; i < features.get_rows(); i++)
    {
        auto nearest = find_k_nearest(features[i]);

        std::map<double, double> frequencies;
        double prediction = 0;
        for (auto [elem_index, elem_dist] : nearest)
        {
            double elem_ans = m_target[elem_index];
            frequencies[elem_ans] += m_weight_strategy->weight(elem_dist);

            if (frequencies[elem_ans] > frequencies[prediction])
                prediction = elem_ans;
        }

        predictions.push_back(prediction);
    }

    return predictions;
};

std::vector<double> KnnRegressor::predict(const Matrix &features) const
{
    if (!m_fitted)
        throw NotFitted("KnnRegressor.predict called before fit");

    if (features.get_cols() != m_data.get_cols())
        throw DimensionMismatch("KnnRegressor.predict features sizes doesn't match fit features size. Predict features size - " + std::to_string(features.get_cols()) + ", training feature size - " + std::to_string(m_data.get_cols()));

    std::vector<double> predictions;
    predictions.reserve(features.get_rows());

    for (size_t i = 0; i < features.get_rows(); i++)
    {
        auto nearest = find_k_nearest(features[i]);

        double weighted_sum = 0.;
        double weights = 0.;

        for (auto [elem_index, elem_dist] : nearest)
        {
            double weight = m_weight_strategy->weight(elem_dist);
            double elem_ans = weight * m_target[elem_index];

            weighted_sum += elem_ans;
            weights += weight;
        }

        predictions.push_back(weighted_sum / weights);
    }

    return predictions;
};