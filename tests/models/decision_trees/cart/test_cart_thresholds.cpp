#include <cmath>
#include <cstddef>
#include <numeric>
#include <span>
#include <vector>

#include "cart_test_data.hpp"
#include "models/api/errors.hpp"
#include "models/decision_trees/cart.hpp"
#include "models/decision_trees/engine/criteria/variance.hpp"
#include "models/decision_trees/engine/leaf/mean_leaf.hpp"
#include "models/decision_trees/engine/split_finder.hpp"
#include "models/decision_trees/engine/tree_builder.hpp"
#include "models/decision_trees/engine/tree_searcher.hpp"
#include "test_util.hpp"

namespace
{

Split build_root_split(const Matrix &X, const std::vector<double> &y, std::size_t min_samples_leaf)
{
    std::vector<std::size_t> object_ids(y.size());
    std::iota(object_ids.begin(), object_ids.end(), 0);
    std::vector<std::size_t> feature_ids = {0};

    SplitFinder finder;
    std::optional<Split> split = finder.find_best(X, y, object_ids, feature_ids, VarianceCriteria{}, min_samples_leaf);
    if (!split)
        throw std::runtime_error("expected a split, got nullopt");
    return *split;
}

} // namespace

void test_threshold_splits_between_distinct_values()
{
    // Значения 2 и 5: порог в [2, 5), объект со значением 2 идёт влево, 5 — вправо.
    Matrix X = make_matrix({{0.}, {2.}, {5.}, {7.}});
    std::vector<double> y = {0., 0., 10., 10.};

    Split split = build_root_split(X, y, 1);
    CHECK(split.feature == 0);
    CHECK(split.threshold >= 2.);
    CHECK(split.threshold < 5.);

    // Маршрутизация согласована с порогом.
    TreeSearcher searcher;
    std::vector<std::size_t> object_ids = {0, 1, 2, 3};
    std::vector<std::size_t> feature_ids = {0};
    TreeBuilder<VarianceCriteria, MeanLeaf> builder(VarianceCriteria{}, MeanLeaf{}, 1, 2, 1);
    Node<double> root = builder.build(X, y, object_ids, feature_ids);

    CHECK_NEAR(searcher.search(root, X(1, std::nullopt).as_const()).get_value(), 0., 1e-12);  // 2 <= t
    CHECK_NEAR(searcher.search(root, X(2, std::nullopt).as_const()).get_value(), 10., 1e-12); // 5 > t
}

void test_equal_values_not_split_inside_group()
{
    // Группа одинаковых значений (три двойки) обязана уйти в одну сторону.
    Matrix X = make_matrix({{1.}, {2.}, {2.}, {2.}, {5.}, {6.}});
    std::vector<double> y = {0., 10., 10., 10., 20., 20.};

    SplitFinder finder;
    std::vector<std::size_t> object_ids = {0, 1, 2, 3, 4, 5};
    std::vector<std::size_t> feature_ids = {0};

    auto split = finder.find_best(X, y, object_ids, feature_ids, VarianceCriteria{}, 2);
    CHECK(split.has_value());
    if (split)
    {
        // Лучший порог стоит между группой двоек и пятёркой.
        CHECK(split->threshold >= 2.);
        CHECK(split->threshold < 5.);

        // Ни одна пара одинаковых значений не разделена порогом.
        CHECK((X(1, 0) <= split->threshold) == (X(2, 0) <= split->threshold));
        CHECK((X(2, 0) <= split->threshold) == (X(3, 0) <= split->threshold));

        // Все три двойки слева, пятёрки справа.
        CHECK(X(1, 0) <= split->threshold);
        CHECK(X(2, 0) <= split->threshold);
        CHECK(X(3, 0) <= split->threshold);
        CHECK(!(X(4, 0) <= split->threshold));
        CHECK(!(X(5, 0) <= split->threshold));
    }
}

void test_threshold_close_representable_pair()
{
    // Соседние представимые double a < b: любой корректный порог
    // обязан оставить a слева, а b — справа (так считал критерий).
    double a = std::nextafter(1.0, 2.0); // 1.000...002, мантисса нечётная
    double b = std::nextafter(a, 2.0);   // следующее представимое

    Matrix X = make_matrix({{a}, {b}});
    std::vector<double> y = {0., 10.};

    SplitFinder finder;
    std::vector<std::size_t> object_ids = {0, 1};
    std::vector<std::size_t> feature_ids = {0};
    auto split = finder.find_best(X, y, object_ids, feature_ids, VarianceCriteria{}, 1);

    CHECK(split.has_value());
    if (!split)
        return;

    // Критерий учитывал: a слева, b справа. Порог должен маршрутизировать так же.
    CHECK(X(0, 0) <= split->threshold);
    CHECK(!(X(1, 0) <= split->threshold));

    // И через predict: разные листья — разные ответы.
    TreeBuilder<VarianceCriteria, MeanLeaf> builder(VarianceCriteria{}, MeanLeaf{}, 1, 2, 1);
    std::vector<std::size_t> ids = {0, 1};
    std::vector<std::size_t> feats = {0};
    Node<double> root = builder.build(X, y, ids, feats);

    TreeSearcher searcher;
    CHECK_NEAR(searcher.search(root, X(0, std::nullopt).as_const()).get_value(), 0., 1e-12);
    CHECK_NEAR(searcher.search(root, X(1, std::nullopt).as_const()).get_value(), 10., 1e-12);
}

void test_threshold_is_midpoint()
{
    // Между 2 и 5 удобный порог — std::midpoint = 3.5.
    Matrix X = make_matrix({{0.}, {2.}, {5.}, {7.}});
    std::vector<double> y = {0., 0., 10., 10.};

    Split split = build_root_split(X, y, 1);
    CHECK_NEAR(split.threshold, std::midpoint(2., 5.), 0.);
}

void test_model_routing_consistent_with_criterion()
{
    // На вложенных данных предсказания обучающихся равен средним
    // по листьям, на которые их отправляет найденный корневой порог.
    LabeledDataset data = make_nested_regression();
    const Matrix &X = data.X;
    const std::vector<double> &y = data.y;

    Split split = build_root_split(X, y, 1);

    double sum_left = 0., sum_right = 0.;
    std::size_t n_left = 0, n_right = 0;
    for (std::size_t i = 0; i < y.size(); i++)
    {
        if (X(i, split.feature) <= split.threshold)
        {
            sum_left += y[i];
            n_left++;
        }
        else
        {
            sum_right += y[i];
            n_right++;
        }
    }

    // Дерево глубины 1: предсказания обязаны совпадать с этими средними.
    CartRegressor model({1, 2, 1});
    model.fit(X, y);
    std::vector<double> predictions = model.predict(X);

    for (std::size_t i = 0; i < y.size(); i++)
    {
        double expected = X(i, split.feature) <= split.threshold ? sum_left / n_left : sum_right / n_right;
        CHECK_NEAR(predictions[i], expected, 1e-12);
    }
}

int main()
{
    test_threshold_splits_between_distinct_values();
    test_equal_values_not_split_inside_group();
    test_threshold_close_representable_pair();
    test_threshold_is_midpoint();
    test_model_routing_consistent_with_criterion();

    return finish("Cart thresholds");
}
