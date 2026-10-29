#include "models/api/distance.hpp"

#include <algorithm>
#include <cmath>

#include "models/api/errors.hpp"

double EuclidDistance::compute(const MatrixSlice<const double> &a, const MatrixSlice<const double> &b) const
{
    if (a.size() != b.size())
        throw DimensionMismatch(
            "Euclid distance: vectors sizes are not equal. First - " + std::to_string(a.size()) + ", second - " +
            std::to_string(b.size()) + "."
        );

    double res = 0.;
    if (a.is_contiguous() && b.is_contiguous())
    {
        auto a_s = a.as_span();
        auto b_s = b.as_span();
        for (size_t i = 0; i < a.size(); i++)
            res += (a_s[i] - b_s[i]) * (a_s[i] - b_s[i]);
    }
    else
    {
        for (size_t i = 0; i < a.size(); i++)
            res += (a(i) - b(i)) * (a(i) - b(i));
    }

    return std::sqrt(res);
}

double ManhattenDistance::compute(const MatrixSlice<const double> &a, const MatrixSlice<const double> &b) const
{
    if (a.size() != b.size())
        throw DimensionMismatch(
            "ManhattenDistance: vectors sizes are not equal. First - " + std::to_string(a.size()) + ", second - " +
            std::to_string(b.size()) + "."
        );

    double res = 0.;
    if (a.is_contiguous() && b.is_contiguous())
    {
        auto a_s = a.as_span();
        auto b_s = b.as_span();
        for (size_t i = 0; i < a.size(); i++)
            res += std::abs(a_s[i] - b_s[i]);
    }
    else
    {
        for (size_t i = 0; i < a.size(); i++)
            res += std::abs(a(i) - b(i));
    }

    return res;
}

double ChebyshevDistance::compute(const MatrixSlice<const double> &a, const MatrixSlice<const double> &b) const
{
    if (a.size() != b.size())
        throw DimensionMismatch(
            "Chebyshev distance: vectors sizes are not equal. First - " + std::to_string(a.size()) + ", second - " +
            std::to_string(b.size()) + "."
        );

    double res = 0.;
    if (a.is_contiguous() && b.is_contiguous())
    {
        auto a_s = a.as_span();
        auto b_s = b.as_span();
        for (size_t i = 0; i < a.size(); i++)
            res = std::max(res, std::abs(a_s[i] - b_s[i]));
    }
    else
    {
        for (size_t i = 0; i < a.size(); i++)
            res = std::max(res, std::abs(a(i) - b(i)));
    }

    return res;
}
