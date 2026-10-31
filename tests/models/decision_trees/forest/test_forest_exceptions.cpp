#include "cart_test_data.hpp"
#include "models/api/errors.hpp"
#include "models/decision_trees/forest.hpp"
#include "test_util.hpp"

void test_bad_hyperparameters()
{
    // estimators < 1
    CHECK_THROWS_AS(ForestRegressor(RandomForestParams{.estimators = 0}), HyperParameterError);
    CHECK_THROWS_AS(ForestClassifier(RandomForestParams{.estimators = 0}), HyperParameterError);

    // bootstrap = false with max_samples
    CHECK_THROWS_AS(ForestRegressor(RandomForestParams{.bootstrap = false, .max_samples = 10}), HyperParameterError);
    CHECK_THROWS_AS(ForestClassifier(RandomForestParams{.bootstrap = false, .max_samples = 10}), HyperParameterError);

    // max_samples = 0
    CHECK_THROWS_AS(ForestRegressor(RandomForestParams{.max_samples = 0}), HyperParameterError);
    CHECK_THROWS_AS(ForestClassifier(RandomForestParams{.max_samples = 0}), HyperParameterError);

    // min_samples_split < 2
    CHECK_THROWS_AS(ForestRegressor(RandomForestParams{.min_samples_split = 1}), HyperParameterError);
    CHECK_THROWS_AS(ForestClassifier(RandomForestParams{.min_samples_split = 0}), HyperParameterError);

    // min_samples_leaf == 0
    CHECK_THROWS_AS(ForestRegressor(RandomForestParams{.min_samples_leaf = 0}), HyperParameterError);
    CHECK_THROWS_AS(ForestClassifier(RandomForestParams{.min_samples_leaf = 0}), HyperParameterError);

    // n_jobs == 0
    CHECK_THROWS_AS(ForestRegressor(RandomForestParams{.n_jobs = 0}), HyperParameterError);
    CHECK_THROWS_AS(ForestClassifier(RandomForestParams{.n_jobs = 0}), HyperParameterError);
}

void test_predict_before_fit()
{
    Matrix X = make_matrix({{1.0, 2.0}, {3.0, 4.0}});

    ForestRegressor regressor({});
    CHECK_THROWS_AS(regressor.predict(X), NotFitted);

    ForestClassifier classifier({});
    CHECK_THROWS_AS(classifier.predict(X), NotFitted);
    CHECK_THROWS_AS(classifier.predict_proba(X), NotFitted);
    CHECK_THROWS_AS(classifier.classes(), NotFitted);
}

void test_dimension_mismatch_fit()
{
    LabeledDataset data = make_step_regression();
    std::vector<double> shorter_y(data.y.begin(), data.y.end() - 1);

    ForestRegressor regressor(RandomForestParams{.estimators = 5});
    CHECK_THROWS_AS(regressor.fit(data.X, shorter_y), DimensionMismatch);

    ForestClassifier classifier(RandomForestParams{.estimators = 5});
    CHECK_THROWS_AS(classifier.fit(data.X, shorter_y), DimensionMismatch);
}

void test_empty_data_fit()
{
    Matrix empty_X(0, 0);
    std::vector<double> empty_y;

    ForestRegressor regressor(RandomForestParams{.estimators = 5});
    CHECK_THROWS_AS(regressor.fit(empty_X, empty_y), EmptyData);

    ForestClassifier classifier(RandomForestParams{.estimators = 5});
    CHECK_THROWS_AS(classifier.fit(empty_X, empty_y), EmptyData);

    // 0 признаков (cols == 0) при ненулевом числе строк
    Matrix zero_cols(5, 0);
    std::vector<double> y5 = {1., 2., 3., 4., 5.};
    CHECK_THROWS_AS(regressor.fit(zero_cols, y5), EmptyData);
    CHECK_THROWS_AS(classifier.fit(zero_cols, y5), EmptyData);
}

void test_predict_feature_mismatch()
{
    LabeledDataset data = make_step_regression();
    ForestRegressor regressor(RandomForestParams{.estimators = 5, .random_state = 42});
    regressor.fit(data.X, data.y);

    Matrix wrong_cols = make_matrix({{1.0, 2.0}});
    CHECK_THROWS_AS(regressor.predict(wrong_cols), DimensionMismatch);

    LabeledDataset cls_data = make_binary_classification();
    ForestClassifier classifier(RandomForestParams{.estimators = 5, .random_state = 42});
    classifier.fit(cls_data.X, cls_data.y);

    CHECK_THROWS_AS(classifier.predict(wrong_cols), DimensionMismatch);
    CHECK_THROWS_AS(classifier.predict_proba(wrong_cols), DimensionMismatch);
}

void test_refit_failure_leaves_not_fitted()
{
    LabeledDataset data = make_step_regression();
    ForestRegressor regressor(RandomForestParams{.estimators = 5, .random_state = 42});
    regressor.fit(data.X, data.y);
    CHECK(regressor.predict(data.X).size() == data.y.size());

    // Failed refit
    std::vector<double> bad_y;
    CHECK_THROWS_AS(regressor.fit(data.X, bad_y), DimensionMismatch);

    // Must not be considered fitted now
    CHECK_THROWS_AS(regressor.predict(data.X), NotFitted);
}

int main()
{
    test_bad_hyperparameters();
    test_predict_before_fit();
    test_dimension_mismatch_fit();
    test_empty_data_fit();
    test_predict_feature_mismatch();
    test_refit_failure_leaves_not_fitted();
    return 0;
}
