#pragma once

#include "feature_info.hpp"

class ScalingStrategy
{
public:
    virtual double scale(double x, const NumericFeatureInfo &) const = 0;
    virtual ~ScalingStrategy() = default;
};

class MinMaxScaling : public ScalingStrategy
{
public:
    double scale(double x, const NumericFeatureInfo &info) const override
    {
        if (info.max == info.min)
            return 1;

        return (x - info.min) / (info.max - info.min);
    }
};

class StandartScaling : public ScalingStrategy
{
public:
    double scale(double x, const NumericFeatureInfo &info) const override
    {

        if (info.variance == 0)
            return 1;

        return (x - info.mean) / info.variance;
    }
};

class RobustScaling : public ScalingStrategy
{
public:
    double scale(double x, const NumericFeatureInfo &info) const override
    {

        if (info.q25 == info.q75)
            return 1;

        return (x - info.median) / (info.q75 - info.q25);
    }
};
