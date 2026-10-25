#include "models/api/distance.hpp"
#include "models/api/errors.hpp"

#include <ranges>
#include <algorithm>
#include <cmath>

double EuclidDistance::compute(
    std::span<const double> a,
    std::span<const double> b) const
{
    if (a.size() != b.size())
        throw DimensionMismatch("Euclid distance: vectors sizes are not equal. First - " + std::to_string(a.size()) + ", second - " + std::to_string(b.size()) + ".");

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
        throw DimensionMismatch("ManhattenDistance: vectors sizes are not equal. First - " + std::to_string(a.size()) + ", second - " + std::to_string(b.size()) + ".");

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
        throw DimensionMismatch("Chebyshev distance: vectors sizes are not equal. First - " + std::to_string(a.size()) + ", second - " + std::to_string(b.size()) + ".");

    double res = 0.;
    for (size_t i = 0; i < a.size(); i++)
        res = std::max(res, std::abs(a[i] - b[i]));

    return res;
}