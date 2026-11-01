#include <cstddef>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <span>
#include <vector>

#include "cart_test_data.hpp"
#include "models/decision_trees/engine/criteria/gini.hpp"
#include "models/decision_trees/engine/criteria/variance.hpp"
#include "models/decision_trees/engine/split_finder.hpp"
#include "test_util.hpp"

namespace
{

/// Наивный полный перебор всех пар (признак, позиция между различающимися
/// значениями) с явным подсчётом оценки. Порог — std::midpoint, как в ядре.
/// Победитель: больше оценка, при равенстве — меньший признак, затем меньший порог.
std::optional<Split> naive_find_best(
    const Matrix &features,
    std::span<const double> targets,
    std::span<const std::size_t> object_ids,
    std::span<const std::size_t> feature_ids,
    bool classification,
    std::size_t classes_count,
    std::size_t min_samples_leaf
)
{
    std::optional<Split> best = std::nullopt;

    for (std::size_t col : feature_ids)
    {
        std::vector<std::size_t> order(object_ids.begin(), object_ids.end());
        std::ranges::sort(order, [&](std::size_t a, std::size_t b) { return features(a, col) < features(b, col); });

        for (std::size_t p = 0; p + 1 < order.size(); p++)
        {
            double a = features(order[p], col);
            double b = features(order[p + 1], col);
            if (a == b)
                continue;

            double threshold = std::midpoint(a, b);

            std::vector<std::size_t> left, right;
            std::vector<std::size_t> left_classes, right_classes;
            if (classification)
            {
                left_classes.assign(classes_count, 0);
                right_classes.assign(classes_count, 0);
            }
            double sum_left = 0., sum_right = 0.;

            for (std::size_t id : object_ids)
            {
                if (features(id, col) <= threshold)
                {
                    left.push_back(id);
                    sum_left += targets[id];
                    if (classification)
                        left_classes[static_cast<std::size_t>(targets[id])]++;
                }
                else
                {
                    right.push_back(id);
                    sum_right += targets[id];
                    if (classification)
                        right_classes[static_cast<std::size_t>(targets[id])]++;
                }
            }

            if (left.size() < min_samples_leaf || right.size() < min_samples_leaf)
                continue;

            double score = 0.;
            if (classification)
            {
                double a_sum = 0., b_sum = 0.;
                for (std::size_t c = 0; c < classes_count; c++)
                {
                    a_sum += static_cast<double>(left_classes[c]) * static_cast<double>(left_classes[c]);
                    b_sum += static_cast<double>(right_classes[c]) * static_cast<double>(right_classes[c]);
                }
                score = a_sum / static_cast<double>(left.size()) + b_sum / static_cast<double>(right.size());
            }
            else
            {
                score = sum_left * sum_left / static_cast<double>(left.size()) +
                        sum_right * sum_right / static_cast<double>(right.size());
            }

            if (!best || score > best->score ||
                (score == best->score &&
                 (col < best->feature || (col == best->feature && threshold < best->threshold))))
            {
                best = Split{.feature = col, .threshold = threshold, .score = score};
            }
        }
    }

    return best;
}

std::vector<std::size_t> iota_ids(std::size_t n)
{
    std::vector<std::size_t> ids(n);
    std::iota(ids.begin(), ids.end(), 0);
    return ids;
}

} // namespace

void test_matches_naive_regression()
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> feat_dist(-10.0, 10.0);
    std::uniform_real_distribution<double> target_dist(-50.0, 50.0);
    std::uniform_int_distribution<int> dup_dist(-3, 3);

    for (int trial = 0; trial < 25; trial++)
    {
        const std::size_t n = 12 + static_cast<std::size_t>(trial % 7);
        const std::size_t d = 3;
        Matrix X(n, d);
        std::vector<double> y(n);

        for (std::size_t i = 0; i < n; i++)
        {
            for (std::size_t j = 0; j < d; j++)
                X(i, j) = (trial % 2 == 0) ? feat_dist(rng) : static_cast<double>(dup_dist(rng));
            y[i] = target_dist(rng);
        }

        for (std::size_t min_leaf : {std::size_t{1}, std::size_t{2}, std::size_t{4}})
        {
            auto object_ids = iota_ids(n);
            auto feature_ids = iota_ids(d);

            SplitFinder finder;
            auto fast = finder.find_best(X, y, object_ids, feature_ids, VarianceCriteria{}, min_leaf);
            auto slow = naive_find_best(X, y, object_ids, feature_ids, false, 0, min_leaf);

            CHECK(fast.has_value() == slow.has_value());
            if (fast && slow)
            {
                CHECK(fast->feature == slow->feature);
                CHECK_NEAR(fast->threshold, slow->threshold, 1e-12);
                CHECK_NEAR(fast->score, slow->score, 1e-9);
            }
        }
    }
}

void test_matches_naive_classification()
{
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> feat_dist(-5.0, 5.0);
    std::uniform_int_distribution<int> class_dist(0, 2);

    for (int trial = 0; trial < 20; trial++)
    {
        const std::size_t n = 15;
        const std::size_t d = 3;
        const std::size_t classes_count = 3;
        Matrix X(n, d);
        std::vector<double> y(n);

        for (std::size_t i = 0; i < n; i++)
        {
            for (std::size_t j = 0; j < d; j++)
                X(i, j) = feat_dist(rng);
            y[i] = static_cast<double>(class_dist(rng));
        }

        for (std::size_t min_leaf : {std::size_t{1}, std::size_t{3}})
        {
            auto object_ids = iota_ids(n);
            auto feature_ids = iota_ids(d);

            SplitFinder finder;
            auto fast = finder.find_best(X, y, object_ids, feature_ids, GiniCriteria{classes_count}, min_leaf);
            auto slow = naive_find_best(X, y, object_ids, feature_ids, true, classes_count, min_leaf);

            CHECK(fast.has_value() == slow.has_value());
            if (fast && slow)
            {
                CHECK(fast->feature == slow->feature);
                CHECK_NEAR(fast->threshold, slow->threshold, 1e-12);
                CHECK_NEAR(fast->score, slow->score, 1e-9);
            }
        }
    }
}

void test_object_subset()
{
    // Разбиение ищется только среди переданных объектов.
    Matrix X = make_matrix({{0.}, {10.}, {1.}, {11.}, {2.}, {12.}});
    std::vector<double> y = {0., 100., 0., 100., 0., 100.};
    std::vector<std::size_t> even_ids = {0, 2, 4}; // только левые значения
    std::vector<std::size_t> feature_ids = {0};

    SplitFinder finder;
    auto split = finder.find_best(X, y, even_ids, feature_ids, VarianceCriteria{}, 1);

    CHECK(split.has_value());
    if (split)
    {
        CHECK(split->feature == 0);
        // Все отобранные объекты лежат слева от исходных 10, 11, 12.
        CHECK(split->threshold < 10.);
    }
}

void test_feature_subset()
{
    // Признак 0 идеально разделяет классы, признак 1 — случайный шум.
    Matrix X = make_matrix({{0., 5.}, {1., 3.}, {2., 8.}, {8., 1.}, {9., 7.}, {10., 4.}});
    std::vector<double> y = {0., 0., 0., 1., 1., 1.};
    std::vector<std::size_t> object_ids = iota_ids(6);
    std::vector<std::size_t> only_noisy = {1};

    SplitFinder finder;
    auto split = finder.find_best(X, y, object_ids, only_noisy, GiniCriteria{2}, 1);

    CHECK(split.has_value());
    if (split)
        CHECK(split->feature == 1);
}

void test_no_admissible_split()
{
    // min_samples_leaf не даёт получить допустимое разбиение из четырёх объектов.
    Matrix X = make_matrix({{0.}, {1.}, {2.}, {3.}});
    std::vector<double> y = {0., 1., 0., 1.};
    std::vector<std::size_t> object_ids = iota_ids(4);
    std::vector<std::size_t> feature_ids = {0};

    SplitFinder finder;
    auto split = finder.find_best(X, y, object_ids, feature_ids, VarianceCriteria{}, 3);
    CHECK(!split.has_value());
}

void test_constant_feature()
{
    Matrix X = make_matrix({{2., 0.}, {2., 1.}, {2., 2.}, {2., 3.}});
    std::vector<double> y = {0., 1., 2., 3.};
    std::vector<std::size_t> object_ids = iota_ids(4);
    std::vector<std::size_t> only_constant = {0};

    SplitFinder finder;
    auto split = finder.find_best(X, y, object_ids, only_constant, VarianceCriteria{}, 1);
    CHECK(!split.has_value());
}

void test_deterministic_tie()
{
    // Оба признака разделяют одинаково: при равных оценках берётся меньший индекс.
    Matrix X = make_matrix({{0., 0.}, {1., 1.}, {2., 2.}, {3., 3.}});
    std::vector<double> y = {0., 0., 10., 10.};
    std::vector<std::size_t> object_ids = iota_ids(4);
    std::vector<std::size_t> feature_ids = {0, 1};

    SplitFinder finder;
    auto first = finder.find_best(X, y, object_ids, feature_ids, VarianceCriteria{}, 1);
    auto second = finder.find_best(X, y, object_ids, feature_ids, VarianceCriteria{}, 1);

    CHECK(first.has_value());
    if (first)
    {
        CHECK(first->feature == 0);
        CHECK(first->threshold == second->threshold);
        CHECK(first->score == second->score);
    }
}

void test_dimension_mismatch()
{
    Matrix X = make_matrix({{0.}, {1.}, {2.}});
    std::vector<double> y = {0., 1.}; // меньше, чем объектов
    std::vector<std::size_t> object_ids = iota_ids(3);
    std::vector<std::size_t> feature_ids = {0};

    SplitFinder finder;
    CHECK_THROWS_AS(finder.find_best(X, y, object_ids, feature_ids, VarianceCriteria{}, 1), DimensionMismatch);
}

int main()
{
    test_matches_naive_regression();
    test_matches_naive_classification();
    test_object_subset();
    test_feature_subset();
    test_no_admissible_split();
    test_constant_feature();
    test_deterministic_tie();
    test_dimension_mismatch();

    return finish("SplitFinder vs naive");
}
