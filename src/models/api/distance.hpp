#pragma once
#include <span>
#include "dataset/matrix.hpp"

class IDistance
{
public:
    virtual ~IDistance() = default;

    virtual double compute(
        std::span<const double> a,
        std::span<const double> b) const = 0;

    virtual double compute(
        const MatrixSlice<const double> &a,
        const MatrixSlice<const double> &b) const = 0;
};

class EuclidDistance : public IDistance
{
public:
    double compute(
        std::span<const double> a,
        std::span<const double> b) const override;

    virtual double compute(
        const MatrixSlice<const double> &a,
        const MatrixSlice<const double> &b) const override;
};

class ManhattenDistance : public IDistance
{
public:
    double compute(
        std::span<const double> a,
        std::span<const double> b) const override;

    virtual double compute(
        const MatrixSlice<const double> &a,
        const MatrixSlice<const double> &b) const override;
};

class ChebyshevDistance : public IDistance
{
public:
    double compute(
        std::span<const double> a,
        std::span<const double> b) const override;

    virtual double compute(
        const MatrixSlice<const double> &a,
        const MatrixSlice<const double> &b) const override;
};