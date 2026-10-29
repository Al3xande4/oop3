#pragma once

#include "dataset/matrix_slice.hpp"

class IDistance
{
public:
    virtual ~IDistance() = default;

    virtual double compute(const MatrixSlice<const double> &a, const MatrixSlice<const double> &b) const = 0;
};

class EuclidDistance : public IDistance
{
public:
    double compute(const MatrixSlice<const double> &a, const MatrixSlice<const double> &b) const override;
};

class ManhattenDistance : public IDistance
{
public:
    double compute(const MatrixSlice<const double> &a, const MatrixSlice<const double> &b) const override;
};

class ChebyshevDistance : public IDistance
{
public:
    double compute(const MatrixSlice<const double> &a, const MatrixSlice<const double> &b) const override;
};