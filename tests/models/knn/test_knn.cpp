#include "test_util.hpp"
#include "models/knn/model.hpp"

#include <memory>
#include <utility>
#include <vector>

namespace
{
    Matrix make_column(const std::vector<double> &values)
    {
        Matrix m(values.size(), 1);
        for (size_t i = 0; i < values.size(); i++)
            m[i][0] = values[i];
        return m;
    }

    struct SpyDistance : IDistance
    {
        std::shared_ptr<IDistance> real;
        mutable size_t calls = 0;

        explicit SpyDistance(std::shared_ptr<IDistance> d) : real{std::move(d)} {}

        double compute(std::span<const double> a, std::span<const double> b) const override
        {
            ++calls;
            return real->compute(a, b);
        }
    };

} // namespace

int main()
{
    auto euclid = std::make_shared<EuclidDistance>();
    auto uniform = std::make_shared<UniformWeight>(1.0);
    auto inverse = std::make_shared<InverseDistanceWeight>();

    Matrix train = make_column({0., 100.});
    std::vector<double> targets = {10., 20.};

    KnnRegressor bad_fit(1, euclid, inverse);
    CHECK_THROWS(bad_fit.fit(make_column({1., 2., 3.}), std::vector<double>{1., 2.}););

    KnnRegressor one_nn(1, euclid, inverse);
    one_nn.fit(train, targets);
    std::vector<double> pred_one = one_nn.predict(make_column({1.}));
    CHECK(pred_one.size() == 1);
    CHECK_NEAR(pred_one[0], 10., 1e-9);

    Matrix train4 = make_column({1., 2., 3., 4.});
    std::vector<double> targets4 = {10., 20., 30., 60.};
    KnnRegressor mean_model(4, euclid, uniform);
    mean_model.fit(train4, targets4);
    std::vector<double> pred_mean = mean_model.predict(make_column({0.}));
    CHECK(pred_mean.size() == 1);
    CHECK_NEAR(pred_mean[0], 30., 1e-12);

    KnnRegressor big_k(10, euclid, uniform);
    big_k.fit(train4, targets4);
    std::vector<double> pred_big = big_k.predict(make_column({0.}));
    CHECK(pred_big.size() == 1);
    CHECK_NEAR(pred_big[0], 30., 1e-12);

    KnnClassifier label_model(1, euclid, uniform);
    label_model.fit(train, std::vector<double>{1., 0.});
    std::vector<double> labels_one = label_model.predict(make_column({1.}));
    CHECK(labels_one.size() == 1);
    CHECK_NEAR(labels_one[0], 1., 1e-12);

    KnnClassifier majority(3, euclid, inverse);
    majority.fit(make_column({0., 1., 2., 10.}), std::vector<double>{1., 1., 0., 0.});
    std::vector<double> labels3 = majority.predict(make_column({0.5}));
    CHECK(labels3.size() == 1);
    CHECK_NEAR(labels3[0], 1., 1e-12);

    auto spy = std::make_shared<SpyDistance>(std::make_shared<EuclidDistance>());
    KnnRegressor spy_model(2, spy, uniform);
    spy_model.fit(train4, targets4);
    CHECK(spy->calls == 0);
    std::vector<double> spy_pred = spy_model.predict(make_column({0., 5.}));
    CHECK(spy_pred.size() == 2);
    CHECK(spy->calls == 8);

    return finish("knn");
}
