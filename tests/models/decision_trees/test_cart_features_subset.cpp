#include <cstddef>
#include <numeric>
#include <span>
#include <vector>

#include "cart_test_data.hpp"
#include "models/decision_trees/criteria/gini.hpp"
#include "models/decision_trees/criteria/variance.hpp"
#include "models/decision_trees/leaf/mean_leaf.hpp"
#include "models/decision_trees/leaf/proba_leaf.hpp"
#include "models/decision_trees/split_finder.hpp"
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

void test_split_finder_picks_only_allowed_feature()
{
    // Признак 0 сепарирует идеально, признак 1 — шум.
    // В подмножестве разрешён только признак 1 — выбирается он, несмотря на худшее качество.
    Matrix X = make_matrix({{0., 5.}, {1., 3.}, {2., 8.}, {8., 1.}, {9., 7.}, {10., 4.}});
    std::vector<double> y = {0., 0., 0., 1., 1., 1.};

    SplitFinder finder;
    auto object_ids = iota_ids(6);

    std::vector<std::size_t> only_noisy = {1};
    auto split = finder.find_best(X, y, object_ids, only_noisy, VarianceCriteria{}, 1);
    CHECK(split.has_value());
    if (split)
        CHECK(split->feature == 1);

    // Для сравнения: с обоими признаками берётся сепарирующий признак 0.
    std::vector<std::size_t> both = {0, 1};
    auto split_all = finder.find_best(X, y, object_ids, both, VarianceCriteria{}, 1);
    CHECK(split_all.has_value());
    if (split_all)
        CHECK(split_all->feature == 0);
}

void test_split_finder_classification_subset()
{
    // То же для критерия Джини.
    Matrix X = make_matrix({{0., 5.}, {1., 3.}, {2., 8.}, {8., 1.}, {9., 7.}, {10., 4.}});
    std::vector<double> y = {0., 0., 0., 1., 1., 1.};

    SplitFinder finder;
    auto object_ids = iota_ids(6);
    std::vector<std::size_t> only_noisy = {1};

    auto split = finder.find_best(X, y, object_ids, only_noisy, GiniCriteria{2}, 1);
    CHECK(split.has_value());
    if (split)
        CHECK(split->feature == 1);
}

void test_tree_builder_never_uses_forbidden_feature()
{
    // Дерево, растущее только по подмножеству признаков, не трогает запрещённые
    // даже если они более информативны.
    Matrix X = make_matrix({
        {0., 10.},
        {1., 11.},
        {2., 12.},
        {3., 13.},
        {7., 14.},
        {8., 15.},
        {9., 16.},
        {10., 17.},
    });
    // Признак 0 сепарирует идеально; признак 1 разрывает классы.
    std::vector<double> y = {0., 0., 0., 0., 1., 1., 1., 1.};

    std::vector<std::size_t> object_ids = iota_ids(8);
    std::vector<std::size_t> only_noise = {1};

    TreeBuilder<VarianceCriteria, MeanLeaf> builder(VarianceCriteria{}, MeanLeaf{}, 3, 2, 1);
    Node<double> root = builder.build(X, y, object_ids, only_noise);

    std::vector<Split> splits;
    collect_splits(root, splits);

    // Шумовой признак монотонен: дерево может разбиться один раз по нему
    // и больше не найти разбиений. Главное — берётся только признак 1.
    for (const Split &split : splits)
        CHECK(split.feature == 1);
}

void test_tree_builder_all_features_still_work()
{
    // Список всех признаков ведёт к использованию информативного признака 0.
    Matrix X = make_matrix({
        {0., 10.},
        {1., 11.},
        {2., 12.},
        {3., 13.},
        {7., 14.},
        {8., 15.},
        {9., 16.},
        {10., 17.},
    });
    std::vector<double> y = {0., 0., 0., 0., 1., 1., 1., 1.};

    std::vector<std::size_t> object_ids = iota_ids(8);
    std::vector<std::size_t> all_features = iota_ids(2);

    TreeBuilder<VarianceCriteria, MeanLeaf> builder(VarianceCriteria{}, MeanLeaf{}, 1, 2, 1);
    Node<double> root = builder.build(X, y, object_ids, all_features);

    std::vector<Split> splits;
    collect_splits(root, splits);

    CHECK(splits.size() == 1);
    if (!splits.empty())
        CHECK(splits[0].feature == 0);
}

void test_classifier_feature_subset()
{
    // Подклассификатор через TreeBuilder с подмножеством признаков.
    Matrix X = make_matrix({{0., 5.}, {1., 3.}, {2., 8.}, {8., 1.}, {9., 7.}, {10., 4.}});
    std::vector<double> y = {0., 0., 0., 1., 1., 1.};
    std::vector<double> classes = {0., 1.};

    std::vector<std::size_t> object_ids = iota_ids(6);
    std::vector<std::size_t> only_noisy = {1};

    TreeBuilder<GiniCriteria, ProbaLeaf> builder(GiniCriteria{2}, ProbaLeaf{classes.size()}, 3, 2, 1);
    Node<std::vector<double>> root = builder.build(X, y, object_ids, only_noisy);

    std::vector<Split> splits;
    collect_splits(root, splits);
    for (const Split &split : splits)
        CHECK(split.feature == 1);
}

int main()
{
    test_split_finder_picks_only_allowed_feature();
    test_split_finder_classification_subset();
    test_tree_builder_never_uses_forbidden_feature();
    test_tree_builder_all_features_still_work();
    test_classifier_feature_subset();

    return finish("Cart feature subset");
}
