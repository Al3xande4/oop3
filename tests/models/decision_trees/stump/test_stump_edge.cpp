#include "models/decision_trees/stump.hpp"
#include "models/api/model.hpp"
#include "test_util.hpp"

#include <initializer_list>
#include <memory>
#include <vector>

namespace
{
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

    void test_constant_feature_ignored()
    {
        Matrix X = make_matrix({
            {7.0, 1.0},
            {7.0, 3.0},
            {7.0, 7.0},
            {7.0, 9.0},
        });
        std::vector<double> y = {-5.0, -5.0, 5.0, 5.0};
        DecisionStumpRegressor stump(1);

        stump.fit(X, y);

        CHECK(stump.split_feature().has_value());
        CHECK(*stump.split_feature() == 1);
        CHECK_NEAR(*stump.split_threshold(), 5.0, 1e-12);

        Matrix matrix = make_matrix({
            {1., 7.},
            {3., 7.},
            {7., 7.},
            {9., 7.},
        });
        DecisionStumpRegressor stump2(1);
        stump2.fit(matrix, y);
        CHECK(stump2.split_feature().has_value());
        CHECK(*stump2.split_feature() == 0);
        CHECK_NEAR(*stump2.split_threshold(), 5.0, 1e-12);
    }

    void test_all_features_constant_regression()
    {
        Matrix X = make_matrix({
            {4.0, 2.0},
            {4.0, 2.0},
            {4.0, 2.0},
            {4.0, 2.0},
        });
        std::vector<double> y = {10.0, 20.0, 30.0, 40.0};
        DecisionStumpRegressor stump(1);

        stump.fit(X, y);
        std::vector<double> preds = stump.predict(X);

        CHECK(!stump.split_feature().has_value());
        CHECK(!stump.split_threshold().has_value());
        CHECK_NEAR(stump.get_left_leaf(), 25.0, 1e-12);
        for (double p : preds)
            CHECK_NEAR(p, 25.0, 1e-12);
    }

    void test_all_features_constant_classification()
    {
        Matrix X = make_matrix({
            {1.0, 1.0},
            {1.0, 1.0},
            {1.0, 1.0},
            {1.0, 1.0},
        });
        std::vector<double> y = {3.0, 1.0, 3.0, 3.0};
        DecisionStumpClassifier stump(1);

        stump.fit(X, y);
        std::vector<double> preds = stump.predict(X);

        CHECK(!stump.split_feature().has_value());
        CHECK(!stump.split_threshold().has_value());
        CHECK_NEAR(stump.get_left_leaf(), 3.0, 1e-12);
        for (double p : preds)
            CHECK_NEAR(p, 3.0, 1e-12);
    }

    void test_min_samples_leaf_blocks_all_splits()
    {
        Matrix X = make_matrix({
            {1.0},
            {2.0},
            {3.0},
            {4.0},
            {5.0},
        });
        std::vector<double> y = {10.0, 20.0, 30.0, 40.0, 50.0};
        DecisionStumpRegressor stump(3);

        stump.fit(X, y);
        std::vector<double> preds = stump.predict(X);

        CHECK(!stump.split_feature().has_value());
        CHECK(!stump.split_threshold().has_value());
        for (double p : preds)
            CHECK_NEAR(p, 30.0, 1e-12);
    }

    void test_deterministic_tie_breaking()
    {
        Matrix X = make_matrix({
            {1.0, 10.0},
            {2.0, 20.0},
            {3.0, 30.0},
            {4.0, 40.0},
        });
        std::vector<double> y = {0.0, 0.0, 10.0, 10.0};
        DecisionStumpRegressor stump(1);

        stump.fit(X, y);

        CHECK(stump.split_feature().has_value());
        CHECK(*stump.split_feature() == 0);
        CHECK_NEAR(*stump.split_threshold(), 2.5, 1e-12);
    }

    void test_single_sample_dataset()
    {
        Matrix X = make_matrix({{42.0, -7.0}});
        std::vector<double> y = {123.0};
        DecisionStumpRegressor reg(1);
        DecisionStumpClassifier clf(1);

        reg.fit(X, y);
        clf.fit(X, y);

        CHECK(!reg.split_feature().has_value());
        CHECK(!clf.split_feature().has_value());
        CHECK_NEAR(reg.predict(X)[0], 123.0, 1e-12);
        CHECK_NEAR(clf.predict(X)[0], 123.0, 1e-12);
    }

    void test_single_class_dataset()
    {
        Matrix X = make_matrix({
            {1.0},
            {2.0},
            {3.0},
            {4.0},
        });
        std::vector<double> y = {5.0, 5.0, 5.0, 5.0};
        DecisionStumpClassifier clf(1);

        clf.fit(X, y);
        std::vector<double> preds = clf.predict(X);

        for (double p : preds)
            CHECK_NEAR(p, 5.0, 1e-12);
    }

}

int main()
{
    test_constant_feature_ignored();
    test_all_features_constant_regression();
    test_all_features_constant_classification();
    test_min_samples_leaf_blocks_all_splits();
    test_deterministic_tie_breaking();
    test_single_sample_dataset();
    test_single_class_dataset();

    return finish("test_stump_edge");
}