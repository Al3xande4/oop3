#include <cstddef>
#include <memory>
#include <numeric>
#include <span>
#include <vector>

#include "cart_test_data.hpp"
#include "models/decision_trees/cart.hpp"
#include "models/decision_trees/stump.hpp"
#include "test_util.hpp"

void test_max_depth_zero_regression()
{
    // max_depth = 0 — один лист: общее среднее по обучающей выборке.
    LabeledDataset data = make_step_regression();

    CartRegressor model(TreeParams{0, 2, 1});
    model.fit(data.X, data.y);

    std::vector<double> predictions = model.predict(data.X);
    CHECK(predictions.size() == data.y.size());

    double mean = 5.; // (0*4 + 10*4) / 8
    for (double p : predictions)
        CHECK_NEAR(p, mean, 1e-12);
}

void test_max_depth_zero_classification()
{
    // max_depth = 0 — один лист с общими частотами классов.
    LabeledDataset data = make_three_class();

    CartClassifier model(TreeParams{0, 2, 1});
    model.fit(data.X, data.y);

    Matrix proba = model.predict_proba(data.X);
    CHECK(proba.get_rows() == data.y.size());
    CHECK(proba.get_cols() == 3);

    for (std::size_t row = 0; row < proba.get_rows(); row++)
    {
        CHECK_NEAR(proba(row, 0), 2. / 6., 1e-12);
        CHECK_NEAR(proba(row, 1), 2. / 6., 1e-12);
        CHECK_NEAR(proba(row, 2), 2. / 6., 1e-12);
    }

    // Ничья максимальных вероятностей: побеждает меньшая метка.
    std::vector<double> predictions = model.predict(data.X);
    for (double p : predictions)
        CHECK_NEAR(p, 0., 1e-12);
}

void test_depth_one_equals_stump_regression()
{
    // При max_depth = 1 дерево совпадает с пнём из работы 2.
    const std::vector<LabeledDataset> datasets = {make_step_regression(), make_nested_regression()};

    for (const LabeledDataset &data : datasets)
    {
        for (std::size_t min_leaf : {std::size_t{1}, std::size_t{2}, std::size_t{3}})
        {
            DecisionStumpRegressor stump(min_leaf);
            CartRegressor tree(TreeParams{1, 2, min_leaf});

            stump.fit(data.X, data.y);
            tree.fit(data.X, data.y);

            std::vector<double> pred_stump = stump.predict(data.X);
            std::vector<double> pred_tree = tree.predict(data.X);

            for (std::size_t i = 0; i < data.y.size(); i++)
                CHECK_NEAR(pred_tree[i], pred_stump[i], 1e-12);
        }
    }
}

void test_depth_one_equals_stump_classification()
{
    // Метки 0/1 без ничьих в листьях: режим пня равен аргмаксу вероятностей.
    const std::vector<LabeledDataset> datasets = {make_binary_classification(), make_three_class()};

    for (const LabeledDataset &data : datasets)
    {
        for (std::size_t min_leaf : {std::size_t{1}, std::size_t{2}})
        {
            DecisionStumpClassifier stump(min_leaf);
            CartClassifier tree(TreeParams{1, 2, min_leaf});

            stump.fit(data.X, data.y);
            tree.fit(data.X, data.y);

            std::vector<double> pred_stump = stump.predict(data.X);
            std::vector<double> pred_tree = tree.predict(data.X);

            for (std::size_t i = 0; i < data.y.size(); i++)
                CHECK_NEAR(pred_tree[i], pred_stump[i], 1e-12);
        }
    }
}

void test_separable_regression_exact()
{
    // Сепарируемые данные: дерево глубины 1 восстанавливает оба уровня точно.
    LabeledDataset data = make_step_regression();

    CartRegressor model(TreeParams{3, 2, 1});
    model.fit(data.X, data.y);

    std::vector<double> predictions = model.predict(data.X);
    for (std::size_t i = 0; i < data.y.size(); i++)
        CHECK_NEAR(predictions[i], data.y[i], 1e-12);

    // Точки между порогом уходят влево, после порога — вправо.
    Matrix queries = make_matrix({{3.5}, {100.}, {-10.}});
    std::vector<double> q = model.predict(queries);
    CHECK_NEAR(q[0], 0., 1e-12); // 3.5 <= 3.5
    CHECK_NEAR(q[1], 10., 1e-12);
    CHECK_NEAR(q[2], 0., 1e-12);
}

void test_separable_classification_exact()
{
    LabeledDataset data = make_binary_classification();

    CartClassifier model(TreeParams{3, 2, 1});
    model.fit(data.X, data.y);

    std::vector<double> predictions = model.predict(data.X);
    for (std::size_t i = 0; i < data.y.size(); i++)
        CHECK_NEAR(predictions[i], data.y[i], 1e-12);

    Matrix proba = model.predict_proba(data.X);
    for (std::size_t i = 0; i < data.y.size(); i++)
    {
        double expected_one = data.y[i];
        for (std::size_t c = 0; c < 2; c++)
            CHECK_NEAR(proba(i, c), c == static_cast<std::size_t>(expected_one) ? 1. : 0., 1e-12);
    }
}

void test_via_interface_polymorphism()
{
    // Модель подставляется через IRegressor / IClassifier без проверок конкретного класса.
    LabeledDataset reg = make_nested_regression();
    LabeledDataset clf = make_binary_classification();

    std::unique_ptr<IRegressor> regressor = std::make_unique<CartRegressor>(TreeParams{3, 2, 1});
    std::unique_ptr<IClassifier> classifier = std::make_unique<CartClassifier>(TreeParams{3, 2, 1});

    regressor->fit(reg.X, reg.y);
    classifier->fit(clf.X, clf.y);

    std::vector<double> pred_reg = regressor->predict(reg.X);
    CHECK(pred_reg.size() == reg.y.size());

    std::vector<double> pred_clf = classifier->predict(clf.X);
    for (std::size_t i = 0; i < clf.y.size(); i++)
        CHECK_NEAR(pred_clf[i], clf.y[i], 1e-12);

    Matrix proba = classifier->predict_proba(clf.X);
    CHECK(proba.get_cols() == classifier->classes().size());

    std::span<const double> classes = classifier->classes();
    CHECK(classes.size() == 2);
    CHECK_NEAR(classes[0], 0., 1e-12);
    CHECK_NEAR(classes[1], 1., 1e-12);
}

int main()
{
    test_max_depth_zero_regression();
    test_max_depth_zero_classification();
    test_depth_one_equals_stump_regression();
    test_depth_one_equals_stump_classification();
    test_separable_regression_exact();
    test_separable_classification_exact();
    test_via_interface_polymorphism();

    return finish("Cart basic");
}
