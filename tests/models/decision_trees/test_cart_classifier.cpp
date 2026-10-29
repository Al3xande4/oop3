#include <cstddef>
#include <span>
#include <vector>

#include "cart_test_data.hpp"
#include "models/decision_trees/cart.hpp"
#include "test_util.hpp"

void test_classes_sorted()
{
    // classes() возвращает метки по возрастанию в порядке столбцов predict_proba.
    LabeledDataset data = make_three_class();

    CartClassifier model(3, 2, 1);
    model.fit(data.X, data.y);

    std::span<const double> classes = model.classes();
    CHECK(classes.size() == 3);
    CHECK_NEAR(classes[0], 0., 1e-12);
    CHECK_NEAR(classes[1], 1., 1e-12);
    CHECK_NEAR(classes[2], 2., 1e-12);
}

void test_proba_shape_and_sums()
{
    LabeledDataset data = make_three_class();

    CartClassifier model(3, 2, 1);
    model.fit(data.X, data.y);

    Matrix proba = model.predict_proba(data.X);
    CHECK(proba.get_rows() == data.y.size());
    CHECK(proba.get_cols() == model.classes().size());

    for (std::size_t row = 0; row < proba.get_rows(); row++)
    {
        double sum = 0.;
        for (std::size_t c = 0; c < proba.get_cols(); c++)
        {
            CHECK(proba(row, c) >= 0.);
            sum += proba(row, c);
        }
        CHECK_NEAR(sum, 1., 1e-12);
    }
}

void test_leaf_missing_class_has_zero_probability()
{
    // В чистых листьях нет части классов: соответствующие вероятности нулевые.
    // Метки сдвинуты: 1 и 2, нулевого класса в данных нет.
    Matrix X = make_matrix({{0.}, {1.}, {2.}, {3.}, {4.}, {5.}});
    std::vector<double> y = {1., 1., 1., 2., 2., 2.};

    CartClassifier model(3, 2, 1);
    model.fit(X, y);

    std::span<const double> classes = model.classes();
    CHECK(classes.size() == 2);
    CHECK_NEAR(classes[0], 1., 1e-12);
    CHECK_NEAR(classes[1], 2., 1e-12);

    Matrix proba = model.predict_proba(X);
    for (std::size_t row = 0; row < 3; row++)
    {
        CHECK_NEAR(proba(row, 0), 1., 1e-12); // класс 1
        CHECK_NEAR(proba(row, 1), 0., 1e-12); // класс 2 отсутствует в листе
    }
    for (std::size_t row = 3; row < 6; row++)
    {
        CHECK_NEAR(proba(row, 0), 0., 1e-12);
        CHECK_NEAR(proba(row, 1), 1., 1e-12);
    }
}

void test_predict_agrees_with_proba()
{
    LabeledDataset data = make_three_class();

    CartClassifier model(3, 2, 1);
    model.fit(data.X, data.y);

    std::span<const double> classes = model.classes();
    Matrix proba = model.predict_proba(data.X);
    std::vector<double> predictions = model.predict(data.X);

    for (std::size_t row = 0; row < data.y.size(); row++)
    {
        // argmax с выбором меньшей метки при равенстве.
        std::size_t best = 0;
        for (std::size_t c = 1; c < classes.size(); c++)
            if (proba(row, c) > proba(row, best))
                best = c;

        CHECK_NEAR(predictions[row], classes[best], 1e-12);
        CHECK_NEAR(predictions[row], data.y[row], 1e-12);
    }
}

void test_tie_prefers_smaller_label()
{
    // Все объекты одного признака: единственный лист, частоты классов поровну.
    // При равенстве максимальных вероятностей предсказывается меньшая метка.
    Matrix X = make_matrix({{1.}, {1.}, {1.}, {1.}});
    std::vector<double> y = {0., 1., 0., 1.};

    CartClassifier model(3, 2, 1);
    model.fit(X, y);

    Matrix proba = model.predict_proba(X);
    CHECK_NEAR(proba(0, 0), 0.5, 1e-12);
    CHECK_NEAR(proba(0, 1), 0.5, 1e-12);

    std::vector<double> predictions = model.predict(X);
    for (double p : predictions)
        CHECK_NEAR(p, 0., 1e-12); // меньшая метка
}

void test_mixed_leaf_frequencies()
{
    // При одинаковых признаках дальнейшее разбиение невозможно:
    // лист остаётся смешанным с точными частотами классов 2/5 и 3/5.
    Matrix X = make_matrix({{0.}, {10.}, {10.}, {10.}, {10.}, {10.}});
    std::vector<double> y = {0., 0., 0., 1., 1., 1.};

    CartClassifier model(5, 2, 1);
    model.fit(X, y);

    Matrix proba = model.predict_proba(X);
    // Слева (x = 0): чистый класс 0.
    CHECK_NEAR(proba(0, 0), 1., 1e-12);
    CHECK_NEAR(proba(0, 1), 0., 1e-12);

    // Справа (x = 10): класс 0 — 2 объекта, класс 1 — 3 объекта.
    for (std::size_t row = 1; row < 6; row++)
    {
        CHECK_NEAR(proba(row, 0), 2. / 5., 1e-12);
        CHECK_NEAR(proba(row, 1), 3. / 5., 1e-12);
    }

    std::vector<double> predictions = model.predict(X);
    CHECK_NEAR(predictions[0], 0., 1e-12);
    for (std::size_t i = 1; i < 6; i++)
        CHECK_NEAR(predictions[i], 1., 1e-12);
}

int main()
{
    test_classes_sorted();
    test_proba_shape_and_sums();
    test_leaf_missing_class_has_zero_probability();
    test_predict_agrees_with_proba();
    test_tie_prefers_smaller_label();
    test_mixed_leaf_frequencies();

    return finish("Cart classifier");
}
