#include "test_util.hpp"
#include "dataset/matrix.hpp"
#include "models/decision_trees/stump.hpp"

Matrix make_matrix(std::initializer_list<std::initializer_list<double>> rows)
{
    const size_t n_rows = rows.size();
    const size_t n_cols = n_rows > 0 ? rows.begin()->size() : 0;
    Matrix m(n_rows, n_cols);

    size_t i = 0;
    for (const auto &row : rows)
    {
        size_t j = 0;
        for (double val : row)
        {
            m[i][j] = val;
            ++j;
        }
        ++i;
    }
    return m;
}

void test_separable_data_regression()
{
    Matrix X = make_matrix({
        {1.0, 1.0},
        {9.0, 2.0},
        {2.0, 8.0},
        {8.0, 9.0},
    });
    std::vector<double> y = {10.0, 10.0, 50.0, 50.0};
    DecisionStumpRegressor stump(1);

    stump.fit(X, y);
    std::vector<double> preds = stump.predict(make_matrix({{0.0, 1.5}, {0.0, 8.5}}));

    CHECK(stump.split_feature().has_value());
    CHECK(*stump.split_feature() == 1);
    CHECK_NEAR(*stump.split_threshold(), 5.0, 1e-12);
    CHECK_NEAR(stump.get_left_leaf(), 10.0, 1e-12);
    CHECK_NEAR(stump.get_right_leaf(), 50.0, 1e-12);
    CHECK_NEAR(preds[0], 10.0, 1e-12);
    CHECK_NEAR(preds[1], 50.0, 1e-12);
}

void test_separable_data_classification()
{
    Matrix X = make_matrix({
        {5.0, 100.0},
        {10.0, -50.0},
        {20.0, 100.0},
        {25.0, -50.0},
    });
    std::vector<double> y = {0.0, 0.0, 1.0, 1.0};
    DecisionStumpClassifier stump(1);

    stump.fit(X, y);
    std::vector<double> preds = stump.predict(make_matrix({{12.0, 0.0}, {18.0, 0.0}}));

    CHECK(stump.split_feature().has_value());
    CHECK(*stump.split_feature() == 0);
    CHECK_NEAR(*stump.split_threshold(), 15.0, 1e-12);
    CHECK_NEAR(stump.get_left_leaf(), 0.0, 1e-12);
    CHECK_NEAR(stump.get_right_leaf(), 1.0, 1e-12);
    CHECK_NEAR(preds[0], 0.0, 1e-12);
    CHECK_NEAR(preds[1], 1.0, 1e-12);
}

void test_model_interface()
{
    Matrix X = make_matrix({
        {5.0, 100.0},
        {10.0, -50.0},
        {20.0, 100.0},
        {25.0, -50.0},
    });
    std::vector<double> y = {0.0, 0.0, 1.0, 1.0};

    std::unique_ptr<IModel> model = std::make_unique<DecisionStumpClassifier>(1);

    model->fit(X, y);
    std::vector<double> preds = model->predict(make_matrix({{12.0, 0.0}, {18.0, 0.0}}));
}

int main()
{
    test_separable_data_classification();
    test_separable_data_regression();
    test_model_interface();

    return finish("Decision Stump Basic tests");
}