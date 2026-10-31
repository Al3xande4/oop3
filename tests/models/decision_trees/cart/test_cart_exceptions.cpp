#include <cmath>
#include <span>
#include <vector>

#include "cart_test_data.hpp"
#include "models/api/errors.hpp"
#include "models/decision_trees/cart.hpp"
#include "test_util.hpp"

void test_bad_hyperparameters()
{
    // min_samples_split должен быть не меньше 2, min_samples_leaf — не меньше 1.
    CHECK_THROWS_AS((CartRegressor(TreeParams{5, 1, 1})), HyperParameterError);
    CHECK_THROWS_AS((CartRegressor(TreeParams{5, 0, 1})), HyperParameterError);
    CHECK_THROWS_AS((CartRegressor(TreeParams{5, 2, 0})), HyperParameterError);
    CHECK_THROWS_AS((CartClassifier(TreeParams{5, 1, 1})), HyperParameterError);
    CHECK_THROWS_AS((CartClassifier(TreeParams{5, 2, 0})), HyperParameterError);

    // Граничные допустимые значения не бросают (падение теста = непойманное исключение).
    CartRegressor valid_regressor(TreeParams{0, 2, 1});
    CartClassifier valid_classifier(TreeParams{1, 2, 1});
    (void)valid_regressor;
    (void)valid_classifier;
}

void test_predict_before_fit()
{
    CartRegressor regressor(TreeParams{3, 2, 1});
    Matrix X = make_matrix({{0.}, {1.}});

    CHECK_THROWS_AS(regressor.predict(X), NotFitted);

    CartClassifier classifier(TreeParams{3, 2, 1});
    CHECK_THROWS_AS(classifier.predict(X), NotFitted);
    CHECK_THROWS_AS(classifier.predict_proba(X), NotFitted);
    CHECK_THROWS_AS(classifier.classes(), NotFitted);
}

void test_fit_dimension_mismatch()
{
    LabeledDataset data = make_step_regression();
    std::span<const double> shorter(data.y.data(), data.y.size() - 1);

    CartRegressor regressor(TreeParams{3, 2, 1});
    CHECK_THROWS_AS(regressor.fit(data.X, shorter), DimensionMismatch);

    CartClassifier classifier(TreeParams{3, 2, 1});
    CHECK_THROWS_AS(classifier.fit(data.X, shorter), DimensionMismatch);
}

void test_fit_empty()
{
    Matrix X(0, 2);
    std::vector<double> y;

    CartRegressor regressor(TreeParams{3, 2, 1});
    CHECK_THROWS_AS(regressor.fit(X, y), EmptyData);

    CartClassifier classifier(TreeParams{3, 2, 1});
    CHECK_THROWS_AS(classifier.fit(X, y), EmptyData);
}

void test_fit_zero_features()
{
    // Ненулевое число признаков обязательно проверяется при fit.
    Matrix X(5, 0);
    std::vector<double> y = {1., 2., 3., 4., 5.};

    CartRegressor regressor(TreeParams{3, 2, 1});
    CHECK_THROWS(regressor.fit(X, y););

    CartClassifier classifier(TreeParams{3, 2, 1});
    CHECK_THROWS(classifier.fit(X, y););
}

void test_predict_feature_mismatch()
{
    LabeledDataset data = make_step_regression();
    Matrix wrong = make_matrix({{0., 1.}, {1., 2.}}); // два столбца вместо одного

    CartRegressor regressor(TreeParams{3, 2, 1});
    regressor.fit(data.X, data.y);
    CHECK_THROWS_AS(regressor.predict(wrong), DimensionMismatch);

    CartClassifier classifier(TreeParams{3, 2, 1});
    classifier.fit(data.X, data.y);
    CHECK_THROWS_AS(classifier.predict(wrong), DimensionMismatch);
    CHECK_THROWS_AS(classifier.predict_proba(wrong), DimensionMismatch);
}

void test_zero_test_objects()
{
    // Ноль тестовых объектов допустим: пустой вектор или матрица 0 × C.
    LabeledDataset data = make_binary_classification();
    Matrix empty(0, 1);

    CartRegressor regressor(TreeParams{3, 2, 1});
    regressor.fit(data.X, data.y);
    std::vector<double> pred = regressor.predict(empty);
    CHECK(pred.empty());

    CartClassifier classifier(TreeParams{3, 2, 1});
    classifier.fit(data.X, data.y);
    std::vector<double> pred_clf = classifier.predict(empty);
    CHECK(pred_clf.empty());

    Matrix proba = classifier.predict_proba(empty);
    CHECK(proba.get_rows() == 0);
    CHECK(proba.get_cols() == classifier.classes().size());
}

void test_refit_replaces_state()
{
    // Повторный успешный fit полностью заменяет прежнее обученное состояние.
    LabeledDataset first = make_step_regression(); // один признак
    LabeledDataset second = make_binary_classification();

    CartRegressor regressor(TreeParams{3, 2, 1});
    regressor.fit(first.X, first.y);
    regressor.fit(second.X, second.y); // другой набор и то же число признаков

    std::vector<double> pred = regressor.predict(second.X);
    for (std::size_t i = 0; i < second.y.size(); i++)
        CHECK_NEAR(pred[i], second.y[i], 1e-12);

    // Классификатор: после повторного fit классы соответствуют новым меткам.
    CartClassifier classifier(TreeParams{3, 2, 1});
    classifier.fit(second.X, second.y);

    std::vector<double> multi_y = {10., 10., 20., 20., 30., 30.};
    classifier.fit(second.X, multi_y);

    std::span<const double> classes = classifier.classes();
    CHECK(classes.size() == 3);
    CHECK_NEAR(classes[0], 10., 1e-12);
    CHECK_NEAR(classes[1], 20., 1e-12);
    CHECK_NEAR(classes[2], 30., 1e-12);
}

int main()
{
    test_bad_hyperparameters();
    test_predict_before_fit();
    test_fit_dimension_mismatch();
    test_fit_empty();
    test_fit_zero_features();
    test_predict_feature_mismatch();
    test_zero_test_objects();
    test_refit_replaces_state();

    return finish("Cart exceptions");
}
