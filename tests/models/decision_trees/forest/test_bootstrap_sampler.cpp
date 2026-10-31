#include <random>

#include "models/api/errors.hpp"
#include "models/common/bootstrap_sampler.hpp"
#include "test_util.hpp"

void test_bootstrap_off()
{
    BootstrapSampler sampler(false);
    std::mt19937 gen(42);

    auto res = sampler.sample(10, gen);
    CHECK(res.counts.size() == 10);
    CHECK(res.object_ids.size() == 10);

    for (size_t i = 0; i < 10; ++i)
    {
        CHECK(res.counts[i] == 1);
        CHECK(res.object_ids[i] == i);
    }
}

void test_bootstrap_on_default_samples()
{
    BootstrapSampler sampler(true);
    std::mt19937 gen(42);

    auto res = sampler.sample(20, gen);
    CHECK(res.counts.size() == 20);

    size_t total_samples = 0;
    for (size_t c : res.counts)
        total_samples += c;

    CHECK(total_samples == 20);

    // object_ids contains only indices with positive counts
    for (size_t id : res.object_ids)
        CHECK(res.counts[id] > 0);
}

void test_bootstrap_custom_max_samples()
{
    BootstrapSampler sampler(true, 7);
    std::mt19937 gen(123);

    auto res = sampler.sample(20, gen);
    CHECK(res.counts.size() == 20);

    size_t total_samples = 0;
    for (size_t c : res.counts)
        total_samples += c;

    CHECK(total_samples == 7);
}

void test_bootstrap_exceptions()
{
    // max_samples with bootstrap=false throws HyperParameterError
    CHECK_THROWS_AS(BootstrapSampler(false, 5), HyperParameterError);

    // max_samples > rows throws HyperParameterError
    BootstrapSampler sampler(true, 30);
    std::mt19937 gen(0);
    CHECK_THROWS_AS(sampler.sample(10, gen), HyperParameterError);

    // max_samples = 0 throws HyperParameterError
    BootstrapSampler sampler_zero(true, 0);
    CHECK_THROWS_AS(sampler_zero.sample(10, gen), HyperParameterError);
}

int main()
{
    test_bootstrap_off();
    test_bootstrap_on_default_samples();
    test_bootstrap_custom_max_samples();
    test_bootstrap_exceptions();
    return 0;
}
