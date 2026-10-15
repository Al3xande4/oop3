#include "models/api/distance.hpp"

#include <ranges>
#include <algorithm>
#include <cmath>

double EuclidDistance::compute(
    std::span<const double> a,
    std::span<const double> b) const
{
    if (a.size() != b.size())
        throw std::runtime_error("Euclid distance: sizes don't match");

    double res = 0.;
    for (size_t i = 0; i < a.size(); i++)
        res += std::pow(a[i] - b[i], 2);

    return std::sqrt(res);
}

double ManhattenDistance::compute(
    std::span<const double> a,
    std::span<const double> b) const
{
    if (a.size() != b.size())
        throw std::runtime_error("Manhatten distance: sizes don't match");

    double res = 0.;
    for (size_t i = 0; i < a.size(); i++)
        res += std::abs(a[i] - b[i]);

    return res;
}

double ChebyshevDistance::compute(
    std::span<const double> a,
    std::span<const double> b) const
{
    if (a.size() != b.size())
        throw std::runtime_error("Chebyshev distance: sizes don't match");

    double res = 0.;
    for (size_t i = 0; i < a.size(); i++)
        res = std::max(res, std::abs(a[i] - b[i]));

    return res;
}