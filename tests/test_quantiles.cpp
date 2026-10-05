#include "dataset/feature_info.hpp"
#include "test_util.hpp"

#include <vector>

int main()
{
    const std::vector<double> sorted{1., 2., 3., 4., 5., 6., 7., 8., 9., 10.};
    const std::span<const double> values(sorted);

    CHECK_NEAR(mean(values), 5.5, 1e-12);
    CHECK_NEAR(median(values), 5.5, 1e-12);

    CHECK_NEAR(quantile(values, 0.05), 1.45, 1e-12);
    CHECK_NEAR(quantile(values, 0.25), 3.25, 1e-12);
    CHECK_NEAR(quantile(values, 0.50), 5.50, 1e-12);
    CHECK_NEAR(quantile(values, 0.75), 7.75, 1e-12);
    CHECK_NEAR(quantile(values, 0.95), 9.55, 1e-12);

    CHECK_NEAR(variance(values), std::sqrt(8.25), 1e-12);

    const std::vector<double> odd{4., 1., 3., 2., 5.};
    const std::span<const double> odd_values(odd);
    (void)odd_values;

    const std::vector<double> odd_sorted_vec{1., 2., 3., 4., 5.};
    const std::span<const double> odd_sorted(odd_sorted_vec);
    CHECK_NEAR(median(odd_sorted), 3.0, 1e-12);
    CHECK_NEAR(quantile(odd_sorted, 0.25), 2.0, 1e-12);

    bool threw = false;
    try
    {
        quantile(values, 1.5);
    }
    catch (const std::invalid_argument &)
    {
        threw = true;
    }
    CHECK(threw);

    threw = false;
    try
    {
        quantile(odd_values, 0.5);
    }
    catch (const std::invalid_argument &)
    {
        threw = true;
    }
    CHECK(threw);

    threw = false;
    try
    {
        mean(std::span<const double>{});
    }
    catch (const std::invalid_argument &)
    {
        threw = true;
    }
    CHECK(threw);

    return finish("quantiles");
}
