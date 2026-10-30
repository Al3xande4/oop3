#pragma once

#include <span>
#include <string>

#include "dataset/matrix.hpp"
#include "models/api/errors.hpp"
namespace validation
{
inline void validate_fit_input(const Matrix &features, std::span<const double> targets, const std::string &model_name)
{
    if (features.get_rows() != targets.size())
        throw DimensionMismatch(
            model_name + ".fit: features and targets mismatch sizes: features - " +
            std::to_string(features.get_rows()) + ", targets - " + std::to_string(targets.size())
        );

    if (targets.size() == 0)
        throw EmptyData(model_name + ".fit: targets size is 0");

    if (features.get_cols() == 0)
        throw EmptyData(model_name + ".fit: features has 0 columns");

    if (features.get_rows() == 0)
        throw EmptyData(model_name + ".fit: features is empty");
}

inline void validate_predict_input(const Matrix &features, bool fitted, size_t fit_feature_size, const std::string &model_name)
{
    if (!fitted)
        throw NotFitted(model_name + ".predict: predict called before fit");

    if (features.get_cols() != fit_feature_size)
        throw DimensionMismatch(
            model_name + ".predict: predict features sizes doesn't match fit features size. Predict features size - " +
            std::to_string(features.get_cols()) + ", training feature size - " + std::to_string(fit_feature_size)
        );
}

} // namespace validation