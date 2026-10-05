#pragma once

#include "feature_info.hpp"

class NumericImputationStrategy
{
public:
    virtual double impute(const NumericFeatureInfo &) const = 0;
    virtual ~NumericImputationStrategy() = default;
};

class MeanImputation : public NumericImputationStrategy
{
public:
    virtual double impute(const NumericFeatureInfo &info) const override
    {
        return info.mean;
    }
};

class ZeroImputation : public NumericImputationStrategy
{
public:
    virtual double impute(const NumericFeatureInfo &) const override
    {
        return 0.0;
    }
};

class MedianImputation : public NumericImputationStrategy
{
public:
    virtual double impute(const NumericFeatureInfo &info) const override
    {
        return info.median;
    }
};