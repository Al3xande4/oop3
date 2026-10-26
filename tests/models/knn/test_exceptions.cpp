#include "models/knn/model.hpp"
#include "models/api/errors.hpp"
#include "dataset/matrix.hpp"
#include "test_util.hpp"

#include <stdexcept>
#include <vector>

namespace
{
    void test_invalid_k()
    {
        CHECK_THROWS_AS(KnnRegressor(0, std::make_shared<EuclidDistance>(), std::make_shared<UniformWeight>());, HyperParameterError);
        CHECK_THROWS_AS(KnnClassifier(0, std::make_shared<EuclidDistance>(), std::make_shared<UniformWeight>());, HyperParameterError);

        KnnClassifier reg(3, std::make_shared<EuclidDistance>(), std::make_shared<UniformWeight>());
        KnnRegressor clf(3, std::make_shared<EuclidDistance>(), std::make_shared<UniformWeight>());
        Matrix X(2, 2);

        CHECK_THROWS(reg.predict(X););
        CHECK_THROWS(clf.predict(X););
    }

    void test_methods_before_fit_throw()
    {
        KnnClassifier reg(1, std::make_shared<EuclidDistance>(), std::make_shared<UniformWeight>());
        KnnRegressor clf(1, std::make_shared<EuclidDistance>(), std::make_shared<UniformWeight>());
        Matrix X(2, 2);

        CHECK_THROWS_AS(reg.predict(X), NotFitted);
        CHECK_THROWS_AS(clf.predict(X), NotFitted);
    }

    void test_fit_dimension_mismatch_throws()
    {
        KnnClassifier reg(1, std::make_shared<EuclidDistance>(), std::make_shared<UniformWeight>());
        KnnRegressor clf(1, std::make_shared<EuclidDistance>(), std::make_shared<UniformWeight>());

        Matrix X(4, 2);
        std::vector<double> y_short = {1.0, 2.0};
        std::vector<double> y_long = {1.0, 2.0, 3.0, 4.0, 5.0};

        CHECK_THROWS_AS(reg.fit(X, y_short), DimensionMismatch);
        CHECK_THROWS_AS(reg.fit(X, y_long), DimensionMismatch);
        CHECK_THROWS_AS(clf.fit(X, y_short), DimensionMismatch);
        CHECK_THROWS_AS(clf.fit(X, y_long), DimensionMismatch);
    }

    void test_fit_empty_data_throws()
    {
        KnnClassifier reg(1, std::make_shared<EuclidDistance>(), std::make_shared<UniformWeight>());
        KnnRegressor clf(1, std::make_shared<EuclidDistance>(), std::make_shared<UniformWeight>());

        Matrix X_empty(0, 3);
        std::vector<double> y_empty;

        CHECK_THROWS_AS(reg.fit(X_empty, y_empty), EmptyData);
        CHECK_THROWS_AS(clf.fit(X_empty, y_empty), EmptyData);
    }

    void test_predict_feature_count_mismatch_throws()
    {
        KnnClassifier reg(1, std::make_shared<EuclidDistance>(), std::make_shared<UniformWeight>());
        KnnRegressor clf(1, std::make_shared<EuclidDistance>(), std::make_shared<UniformWeight>());

        Matrix X_train(4, 2);
        std::vector<double> y = {0.0, 0.0, 1.0, 1.0};
        reg.fit(X_train, y);
        clf.fit(X_train, y);

        Matrix X_fewer_cols(2, 1);
        Matrix X_more_cols(2, 5);

        CHECK_THROWS_AS(reg.predict(X_fewer_cols), DimensionMismatch);
        CHECK_THROWS_AS(reg.predict(X_more_cols), DimensionMismatch);
        CHECK_THROWS_AS(clf.predict(X_fewer_cols), DimensionMismatch);
        CHECK_THROWS_AS(clf.predict(X_more_cols), DimensionMismatch);
    }

}

int main()
{
    test_invalid_k();
    test_methods_before_fit_throw();
    test_fit_dimension_mismatch_throws();
    test_fit_empty_data_throws();
    test_predict_feature_count_mismatch_throws();

    return finish("test_stump_exceptions");
}