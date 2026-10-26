#include "test_util.hpp"
#include "models/decision_trees/stump.hpp"

#include <random>

class NaiveDecisionStumpRegressor : public DecisionStumpRegressor
{
public:
    using DecisionStumpRegressor::DecisionStumpRegressor;

    void fit(
        const Matrix &features,
        std::span<const double> targets) override
    {
        m_fitted = true;
        m_fit_feature_size = features.get_cols();
        m_constant = -43124;

        double best = std::numeric_limits<double>::infinity();
        for (size_t col = 0; col < features.get_cols(); col++)
        {
            for (size_t row = 0; row < features.get_rows(); row++)
            {
                double cur = features[row][col];
                double next = std::numeric_limits<double>::infinity();

                for (size_t i = 0; i < features.get_rows(); i++)
                {
                    if (features[i][col] > cur && features[i][col] < next)
                        next = features[i][col];
                }

                double t = std::midpoint(cur, next);

                size_t count_left = 0, count_right = 0;
                double sum_left = 0., sum_right = 0.;

                for (size_t i = 0; i < features.get_rows(); i++)
                {
                    if (features[i][col] <= t)
                    {
                        count_left++;
                        sum_left += targets[i];
                    }
                    else
                    {
                        count_right++;
                        sum_right += targets[i];
                    }
                }

                if (count_left < m_min_samples_leaf || count_right < m_min_samples_leaf)
                    continue;

                double mean_left = sum_left / count_left, mean_right = sum_right / count_right;
                double sse = 0.;

                for (size_t i = 0; i < features.get_rows(); i++)
                {
                    double error = 0.;
                    if (features[i][col] <= t)
                        error = targets[i] - mean_left;
                    else
                        error = targets[i] - mean_right;

                    sse += error * error;
                }

                if (sse < best)
                {
                    best = sse;
                    m_threshold = t;
                    m_feature_index = col;
                    m_left_mean = mean_left;
                    m_right_mean = mean_right;
                }
                else if (std::abs(sse - best) < 1e-9)
                {
                    if (col < m_feature_index)
                    {
                        m_feature_index = col;
                        m_threshold = t;
                        m_left_mean = mean_left;
                        m_right_mean = mean_right;
                    }
                    else if (col == m_feature_index && t < m_threshold)
                    {
                        m_threshold = t;
                        m_left_mean = mean_left;
                        m_right_mean = mean_right;
                    }
                }
            }
        }
    }
};

void test_regression_matches_naive()
{
    std::mt19937 rng(100);
    std::uniform_real_distribution<double> feat_dist(-20.0, 20.0);
    std::uniform_real_distribution<double> target_dist(-100.0, 100.0);
    std::uniform_int_distribution<int> dup_dist(-4, 4);

    const std::vector<size_t> leaf_sizes = {1, 3, 8, 15};

    for (int trial = 0; trial < 20; ++trial)
    {
        const size_t n = 40;
        const size_t d = 4;
        Matrix X(n, d);
        std::vector<double> y(n);

        for (size_t i = 0; i < n; ++i)
        {
            for (size_t j = 0; j < d; ++j)
                X[i][j] = (trial % 2 == 0) ? feat_dist(rng) : (dup_dist(rng));
            y[i] = target_dist(rng);
        }

        for (size_t min_leaf : leaf_sizes)
        {
            DecisionStumpRegressor fast(min_leaf);
            NaiveDecisionStumpRegressor naive(min_leaf);

            fast.fit(X, y);
            naive.fit(X, y);

            CHECK(fast.split_feature().has_value() == naive.split_feature().has_value());
            if (fast.split_feature().has_value() && naive.split_feature().has_value())
            {
                CHECK(*fast.split_feature() == *naive.split_feature());
                CHECK_NEAR(*fast.split_threshold(), *naive.split_threshold(), 1e-9);
                CHECK_NEAR(fast.get_left_leaf(), naive.get_left_leaf(), 1e-9);
                CHECK_NEAR(fast.get_right_leaf(), naive.get_right_leaf(), 1e-9);
            }

            std::vector<double> pred_fast = fast.predict(X);
            std::vector<double> pred_naive = naive.predict(X);
            for (size_t i = 0; i < n; ++i)
                CHECK_NEAR(pred_fast[i], pred_naive[i], 1e-9);
        }
    }
}

void benchmark_fast_vs_naive()
{
    const size_t n = 2000;
    const size_t d = 20;
    std::mt19937 rng(100);
    std::uniform_real_distribution<double> dist(-100.0, 100.0);

    Matrix X(n, d);
    std::vector<double> y(n);
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < d; ++j)
            X[i][j] = dist(rng);
        y[i] = dist(rng);
    }

    DecisionStumpRegressor fast(5);
    NaiveDecisionStumpRegressor naive(5);

    auto t0 = std::chrono::steady_clock::now();
    fast.fit(X, y);
    auto t1 = std::chrono::steady_clock::now();

    auto t2 = std::chrono::steady_clock::now();
    naive.fit(X, y);
    auto t3 = std::chrono::steady_clock::now();

    double fast_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double naive_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    std::printf("Comparison n=%zu, d=%zu. Fast O(d*n log n): %.3f ms | Naive O(d*n^2): %.3f ms\n",
                n, d, fast_ms, naive_ms);
}

int main()
{
    test_regression_matches_naive();
    // benchmark_fast_vs_naive();

    return finish("Naive Stump vs Core Stump");
}