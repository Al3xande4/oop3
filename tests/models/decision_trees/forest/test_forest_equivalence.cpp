#include "cart_test_data.hpp"
#include "models/decision_trees/cart.hpp"
#include "models/decision_trees/forest.hpp"
#include "test_util.hpp"

// Проверка пункта 8:
// bootstrap=false, один элемент ансамбля, max_features=All дают обычное CART
// с теми же ограничениями на данных с однозначным лучшим разбиением.
void test_equivalence_regressor()
{
    LabeledDataset data = make_step_regression();

    TreeParams tree_params{.max_depth = 4, .min_samples_split = 2, .min_samples_leaf = 1};
    CartRegressor cart(tree_params);
    cart.fit(data.X, data.y);
    auto cart_pred = cart.predict(data.X);

    RandomForestParams forest_params{
        .estimators = 1,
        .bootstrap = false,
        .max_features = MaxFeaturesMode::All,
        .max_depth = 4,
        .min_samples_split = 2,
        .min_samples_leaf = 1,
        .random_state = 42,
    };
    ForestRegressor forest(forest_params);
    forest.fit(data.X, data.y);
    auto forest_pred = forest.predict(data.X);

    CHECK(cart_pred.size() == forest_pred.size());
    for (size_t i = 0; i < cart_pred.size(); ++i)
        CHECK_NEAR(cart_pred[i], forest_pred[i], 1e-12);
}

void test_equivalence_classifier()
{
    LabeledDataset data = make_three_class();

    TreeParams tree_params{.max_depth = 4, .min_samples_split = 2, .min_samples_leaf = 1};
    CartClassifier cart(tree_params);
    cart.fit(data.X, data.y);
    auto cart_proba = cart.predict_proba(data.X);
    auto cart_pred = cart.predict(data.X);

    RandomForestParams forest_params{
        .estimators = 1,
        .bootstrap = false,
        .max_features = MaxFeaturesMode::All,
        .max_depth = 4,
        .min_samples_split = 2,
        .min_samples_leaf = 1,
        .random_state = 42,
    };
    ForestClassifier forest(forest_params);
    forest.fit(data.X, data.y);
    auto forest_proba = forest.predict_proba(data.X);
    auto forest_pred = forest.predict(data.X);

    CHECK(cart_proba.get_rows() == forest_proba.get_rows());
    CHECK(cart_proba.get_cols() == forest_proba.get_cols());

    for (size_t r = 0; r < cart_proba.get_rows(); ++r)
    {
        CHECK_NEAR(cart_pred[r], forest_pred[r], 1e-12);
        for (size_t c = 0; c < cart_proba.get_cols(); ++c)
            CHECK_NEAR(cart_proba(r, c), forest_proba(r, c), 1e-12);
    }
}

// Проверка пункта 11 (краевые случаи):
// пустой пакет запросов
void test_empty_predict_batch()
{
    LabeledDataset data = make_step_regression();
    ForestRegressor regressor(RandomForestParams{.estimators = 3, .random_state = 42});
    regressor.fit(data.X, data.y);

    Matrix empty_test(0, data.X.get_cols());
    auto reg_pred = regressor.predict(empty_test);
    CHECK(reg_pred.empty());

    LabeledDataset cls_data = make_binary_classification();
    ForestClassifier classifier(RandomForestParams{.estimators = 3, .random_state = 42});
    classifier.fit(cls_data.X, cls_data.y);

    auto cls_pred = classifier.predict(empty_test);
    CHECK(cls_pred.empty());

    auto cls_proba = classifier.predict_proba(empty_test);
    CHECK(cls_proba.get_rows() == 0);
    CHECK(cls_proba.get_cols() == 2);
}

// Проверка пункта 11:
// один объект, один признак
void test_single_sample_and_feature()
{
    Matrix X = make_matrix({{42.0}});
    std::vector<double> y_reg = {100.0};

    ForestRegressor regressor(RandomForestParams{.estimators = 3, .random_state = 42});
    regressor.fit(X, y_reg);
    auto pred_reg = regressor.predict(X);
    CHECK(pred_reg.size() == 1);
    CHECK_NEAR(pred_reg[0], 100.0, 1e-9);

    std::vector<double> y_cls = {1.0};
    ForestClassifier classifier(RandomForestParams{.estimators = 3, .random_state = 42});
    classifier.fit(X, y_cls);
    auto pred_cls = classifier.predict(X);
    CHECK(pred_cls.size() == 1);
    CHECK_NEAR(pred_cls[0], 1.0, 1e-9);
}

// Проверка пункта 11:
// константная матрица признаков / константный отклик
void test_constant_matrix_and_target()
{
    Matrix X = make_matrix({{1.0, 2.0}, {1.0, 2.0}, {1.0, 2.0}, {1.0, 2.0}});
    std::vector<double> y = {5.0, 5.0, 5.0, 5.0};

    ForestRegressor regressor(RandomForestParams{.estimators = 5, .random_state = 42});
    regressor.fit(X, y);
    auto pred = regressor.predict(X);

    CHECK(pred.size() == 4);
    for (double p : pred)
        CHECK_NEAR(p, 5.0, 1e-9);
}

int main()
{
    test_equivalence_regressor();
    test_equivalence_classifier();
    test_empty_predict_batch();
    test_single_sample_and_feature();
    test_constant_matrix_and_target();
    return 0;
}
