#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <vector>

#include "cart_test_data.hpp"
#include "models/decision_trees/cart.hpp"
#include "models/decision_trees/criteria/variance.hpp"
#include "models/decision_trees/leaf/mean_leaf.hpp"
#include "models/decision_trees/tree_builder.hpp"
#include "test_util.hpp"

namespace
{

std::vector<std::size_t> iota_ids(std::size_t n)
{
    std::vector<std::size_t> ids(n);
    std::iota(ids.begin(), ids.end(), 0);
    return ids;
}

} // namespace

void test_tree_differs_from_stump()
{
    // Вложенные пороги: дерево глубже одного уровня отличается от пня.
    LabeledDataset data = make_nested_regression();

    CartRegressor stump_level(1, 2, 1);
    CartRegressor tree(4, 2, 1);

    stump_level.fit(data.X, data.y);
    tree.fit(data.X, data.y);

    std::vector<double> shallow = stump_level.predict(data.X);
    std::vector<double> deep = tree.predict(data.X);

    bool differ = false;
    for (std::size_t i = 0; i < data.y.size(); i++)
        if (std::abs(shallow[i] - deep[i]) > 1e-9)
            differ = true;
    CHECK(differ);

    // Глубокое дерево воспроизводит кусочно-постоянный отклик точно.
    for (std::size_t i = 0; i < data.y.size(); i++)
        CHECK_NEAR(deep[i], data.y[i], 1e-12);
}

void test_max_depth_zero_is_single_leaf()
{
    std::vector<std::size_t> object_ids = iota_ids(8);
    std::vector<std::size_t> feature_ids = {0};

    LabeledDataset data = make_nested_regression();
    TreeBuilder<VarianceCriteria, MeanLeaf> builder(VarianceCriteria{}, MeanLeaf{}, 0, 2, 1);
    Node<double> root = builder.build(data.X, data.y, object_ids, feature_ids);

    TreeStats stats = compute_tree_stats(root);
    CHECK(stats.nodes == 1);
    CHECK(stats.leaves == 1);
    CHECK(stats.branches == 0);
    CHECK(stats.depth == 0);
}

void test_max_depth_one_is_stump_shape()
{
    std::vector<std::size_t> feature_ids = {0};

    LabeledDataset data = make_step_regression();
    std::vector<std::size_t> object_ids = iota_ids(data.y.size());

    TreeBuilder<VarianceCriteria, MeanLeaf> builder(VarianceCriteria{}, MeanLeaf{}, 1, 2, 1);
    Node<double> root = builder.build(data.X, data.y, object_ids, feature_ids);

    TreeStats stats = compute_tree_stats(root);
    CHECK(stats.nodes == 3);
    CHECK(stats.leaves == 2);
    CHECK(stats.branches == 1);
    CHECK(stats.depth == 1);
}

void test_grow_two_levels()
{
    // Вложенные данные требуют двух разбиений: структура больше пня.
    std::vector<std::size_t> feature_ids = {0};

    LabeledDataset data = make_nested_regression();
    std::vector<std::size_t> object_ids = iota_ids(data.y.size());

    TreeBuilder<VarianceCriteria, MeanLeaf> builder(VarianceCriteria{}, MeanLeaf{}, 4, 2, 1);
    Node<double> root = builder.build(data.X, data.y, object_ids, feature_ids);

    TreeStats stats = compute_tree_stats(root);
    CHECK(stats.depth == 3);
    CHECK(stats.leaves == 4);
    CHECK(stats.branches == 3);
    CHECK(stats.nodes == 7);

    std::vector<Split> splits;
    collect_splits(root, splits);
    CHECK(splits.size() == 3);

    // Все разбиения по одному признаку и множество порогов: {1.5, 3.5, 5.5}.
    for (const Split &split : splits)
        CHECK(split.feature == 0);
    if (splits.size() == 3)
    {
        std::vector<double> thresholds;
        thresholds.reserve(splits.size());
        for (const Split &split : splits)
            thresholds.push_back(split.threshold);
        std::ranges::sort(thresholds);
        CHECK_NEAR(thresholds[0], 1.5, 1e-12);
        CHECK_NEAR(thresholds[1], 3.5, 1e-12);
        CHECK_NEAR(thresholds[2], 5.5, 1e-12);
    }
}

void test_depth_limit_bounds_height()
{
    // Фактическая высота не превышает max_depth.
    LabeledDataset data = make_nested_regression();

    for (std::size_t max_depth : {std::size_t{0}, std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{5}})
    {
        std::vector<std::size_t> object_ids = iota_ids(data.y.size());
        std::vector<std::size_t> feature_ids = {0};

        TreeBuilder<VarianceCriteria, MeanLeaf> builder(VarianceCriteria{}, MeanLeaf{}, max_depth, 2, 1);
        Node<double> root = builder.build(data.X, data.y, object_ids, feature_ids);

        TreeStats stats = compute_tree_stats(root);
        CHECK(stats.depth <= max_depth);
        CHECK(stats.leaves <= std::size_t{1} << max_depth);
    }
}

void test_predictions_improve_with_depth()
{
    // Чем больше разрешённая глубина, тем меньше SSE на обучении (не растёт).
    LabeledDataset data = make_nested_regression();

    double previous = std::numeric_limits<double>::infinity();
    for (std::size_t max_depth : {std::size_t{0}, std::size_t{1}, std::size_t{2}, std::size_t{3}})
    {
        CartRegressor model(max_depth, 2, 1);
        model.fit(data.X, data.y);

        std::vector<double> predictions = model.predict(data.X);
        double sse = 0.;
        for (std::size_t i = 0; i < predictions.size(); i++)
        {
            double error = predictions[i] - data.y[i];
            sse += error * error;
        }

        CHECK(sse <= previous + 1e-12);
        previous = sse;
    }

    CHECK(previous < 1e-12); // глубина 3 уже восстанавливает отклик точно
}

int main()
{
    test_tree_differs_from_stump();
    test_max_depth_zero_is_single_leaf();
    test_max_depth_one_is_stump_shape();
    test_grow_two_levels();
    test_depth_limit_bounds_height();
    test_predictions_improve_with_depth();

    return finish("Cart growth");
}
