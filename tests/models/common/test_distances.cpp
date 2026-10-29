#include "test_util.hpp"
#include "models/api/distance.hpp"

#include <array>
#include <cmath>
#include <vector>

template <class Container>
MatrixSlice<const double> to_slice(const Container &c)
{
    return MatrixSlice<const double>(c.data(), c.size(), 1, SliceAxis::Row);
}

struct DistanceTestObject
{
    std::vector<double> a;
    std::vector<double> b;

    std::array<double, 3> expected; // euclid, manh, cheb
};

int main()
{
    EuclidDistance euclid_dist;
    ManhattenDistance manh_dist;
    ChebyshevDistance cheb_dist;

    const std::array<const IDistance *, 3> metrics = {
        &euclid_dist, &manh_dist, &cheb_dist};

    // Arrange: known input/expected triples for every metric
    const std::vector<DistanceTestObject> objects = {
        {{4., 3.}, {0., 0.}, {5., 7., 4.}},
        {{0., 0.}, {0., 0.}, {0., 0., 0.}},
        {{0., 0.}, {3., 4.}, {5., 7., 4.}},
        {{1., -1.}, {-1., 1.}, {std::sqrt(8.), 4., 2.}},
        {{1., 5., 2.}, {4., 1., 8.}, {std::sqrt(61.), 13., 6.}},
    };

    // Act + Assert: table-driven checks against every metric
    for (size_t m = 0; m < metrics.size(); m++)
        for (const auto &obj : objects)
            CHECK_NEAR(
                metrics[m]->compute(to_slice(obj.a), to_slice(obj.b)),
                obj.expected[m],
                1e-12);

    // Properties: identity and symmetry hold for every metric
    for (const IDistance *dist : metrics)
        for (const auto &obj : objects)
        {
            CHECK_NEAR(dist->compute(to_slice(obj.a), to_slice(obj.a)), 0., 1e-12);
            CHECK_NEAR(
                dist->compute(to_slice(obj.a), to_slice(obj.b)),
                dist->compute(to_slice(obj.b), to_slice(obj.a)),
                1e-12);
        }

    // Triangle inequality: (0,0) -> (3,0) -> (3,4)
    const std::vector<double> p0 = {0., 0.};
    const std::vector<double> p1 = {3., 0.};
    const std::vector<double> p2 = {3., 4.};
    for (const IDistance *dist : metrics)
        CHECK(dist->compute(to_slice(p0), to_slice(p2)) <=
              dist->compute(to_slice(p0), to_slice(p1)) + dist->compute(to_slice(p1), to_slice(p2)) + 1e-12);

    // Mismatched sizes must be rejected by every metric
    const std::vector<double> longer = {1., 1., 2.};
    const std::array<double, 2> shorter = {1., 1.};
    for (const IDistance *dist : metrics)
        CHECK_THROWS(dist->compute(to_slice(longer), to_slice(shorter)););

    return finish("distances");
}
