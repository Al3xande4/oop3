#include <cstddef>
#include <vector>

#include "cart_test_data.hpp"
#include "models/decision_trees/cart.hpp"
#include "models/decision_trees/tree_stats.hpp"
#include "test_util.hpp"

void test_constant_feature()
{
    // Константные признаки не дают разбиения: дерево вырождается в лист.
    Matrix X = make_matrix({{2., 7.}, {2., 7.}, {2., 7.}, {2., 7.}, {2., 7.}, {2., 7.}});
    std::vector<double> y = {0., 1., 2., 3., 4., 5.};

    CartRegressor model(5, 2, 1);
    model.fit(X, y);

    std::vector<double> predictions = model.predict(X);
    double mean = 2.5;
    for (double p : predictions)
        CHECK_NEAR(p, mean, 1e-12);
}

void test_all_targets_constant()
{
    // Одинаковые отклики: лист со значением отклика, разбиение не нужно.
    Matrix X = make_matrix({{0.}, {1.}, {2.}, {3.}});
    std::vector<double> y = {7., 7., 7., 7.};

    CartRegressor model(5, 2, 1);
    model.fit(X, y);

    TreeInfo stats = model.get_info();
    CHECK(stats.depth == 0);
    CHECK(stats.branches == 0);
    CHECK(stats.leaves == 1);

    std::vector<double> predictions = model.predict(X);
    for (double p : predictions)
        CHECK_NEAR(p, 7., 1e-12);
}

void test_single_object()
{
    // Один объект: лист с его откликом.
    Matrix X = make_matrix({{1.5}});
    std::vector<double> y = {42.};

    CartRegressor model(5, 2, 1);
    model.fit(X, y);

    std::vector<double> predictions = model.predict(X);
    CHECK(predictions.size() == 1);
    CHECK_NEAR(predictions[0], 42., 1e-12);

    CartClassifier classifier(5, 2, 1);
    classifier.fit(X, y);

    std::span<const double> classes = classifier.classes();
    CHECK(classes.size() == 1);
    CHECK_NEAR(classes[0], 42., 1e-12);

    Matrix proba = classifier.predict_proba(X);
    CHECK(proba.get_rows() == 1);
    CHECK(proba.get_cols() == 1);
    CHECK_NEAR(proba(0, 0), 1., 1e-12);
}

void test_single_class()
{
    // Все объекты одного класса: вероятность этого класса равна единице.
    Matrix X = make_matrix({{0.}, {1.}, {2.}, {3.}, {4.}});
    std::vector<double> y = {3., 3., 3., 3., 3.};

    CartClassifier model(5, 2, 1);
    model.fit(X, y);

    CHECK(model.classes().size() == 1);

    std::vector<double> predictions = model.predict(X);
    for (double p : predictions)
        CHECK_NEAR(p, 3., 1e-12);

    Matrix proba = model.predict_proba(X);
    for (std::size_t row = 0; row < proba.get_rows(); row++)
        CHECK_NEAR(proba(row, 0), 1., 1e-12);
}

void test_repeated_values_stay_together()
{
    // Одинаковые значения признака обязаны уйти в одну сторону разбиения.
    Matrix X = make_matrix({{1.}, {1.}, {1.}, {2.}, {2.}, {2.}});
    std::vector<double> y = {0., 0., 0., 10., 10., 10.};

    CartRegressor model(3, 2, 1);
    model.fit(X, y);

    std::vector<double> predictions = model.predict(X);
    for (std::size_t i = 0; i < 3; i++)
        CHECK_NEAR(predictions[i], 0., 1e-12);
    for (std::size_t i = 3; i < 6; i++)
        CHECK_NEAR(predictions[i], 10., 1e-12);
}

void test_min_samples_leaf_blocks_split()
{
    // min_samples_leaf = 3 при четырёх объектах: допустимого разбиения нет.
    Matrix X = make_matrix({{0.}, {1.}, {2.}, {3.}});
    std::vector<double> y = {0., 0., 10., 10.};

    CartRegressor model(5, 2, 3);
    model.fit(X, y);

    std::vector<double> predictions = model.predict(X);
    for (double p : predictions)
        CHECK_NEAR(p, 5., 1e-12); // общее среднее
}

void test_min_samples_split_blocks_split()
{
    // min_samples_split больше размера узла: разбиение не пробуется.
    Matrix X = make_matrix({{0.}, {1.}, {2.}});
    std::vector<double> y = {0., 5., 10.};

    CartRegressor model(5, 4, 1);
    model.fit(X, y);

    std::vector<double> predictions = model.predict(X);
    for (double p : predictions)
        CHECK_NEAR(p, 5., 1e-12);
}

void test_depth_limits_growth()
{
    // Одни и те же данные: глубина 0, 1 и 2 дают разную точность воспроизведения.
    LabeledDataset data = make_nested_regression();

    CartRegressor depth0(0, 2, 1);
    CartRegressor depth1(1, 2, 1);
    CartRegressor depth2(2, 2, 1);

    depth0.fit(data.X, data.y);
    depth1.fit(data.X, data.y);
    depth2.fit(data.X, data.y);

    std::vector<double> p0 = depth0.predict(data.X);
    std::vector<double> p1 = depth1.predict(data.X);
    std::vector<double> p2 = depth2.predict(data.X);

    // Чем больше глубина, тем меньше сумма квадратов ошибок на обучении.
    auto sse = [&](const std::vector<double> &pred)
    {
        double s = 0.;
        for (std::size_t i = 0; i < pred.size(); i++)
        {
            double e = pred[i] - data.y[i];
            s += e * e;
        }
        return s;
    };

    CHECK(sse(p0) > sse(p1));
    CHECK(sse(p1) > sse(p2));
}

void test_two_features_both_usable()
{
    // Разбиения могут использовать разные признаки в разных узлах.
    Matrix X = make_matrix({
        {0., 10.},
        {0., 11.},
        {0., 12.},
        {0., 13.},
        {10., 0.},
        {10., 1.},
        {10., 2.},
        {10., 3.},
    });
    std::vector<double> y = {0., 0., 0., 0., 10., 10., 10., 10.};

    CartRegressor model(2, 2, 1);
    model.fit(X, y);

    std::vector<double> predictions = model.predict(X);
    for (std::size_t i = 0; i < 4; i++)
        CHECK_NEAR(predictions[i], 0., 1e-12);
    for (std::size_t i = 4; i < 8; i++)
        CHECK_NEAR(predictions[i], 10., 1e-12);
}

int main()
{
    test_constant_feature();
    test_all_targets_constant();
    test_single_object();
    test_single_class();
    test_repeated_values_stay_together();
    test_min_samples_leaf_blocks_split();
    test_min_samples_split_blocks_split();
    test_depth_limits_growth();
    test_two_features_both_usable();

    return finish("Cart edge cases");
}
